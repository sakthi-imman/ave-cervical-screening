using Flux, Metalhead, Images, CUDA, BSON, Glob, Statistics, ProgressMeter, 
      MLUtils, CSV, DataFrames, Optimisers, Zygote, Functors
using Flux: crossentropy, onehotbatch, onecold

# --- Constants ---
const CLASSES = ["Normal", "HPV", "Precancer"]
const IMG_SIZE = (224, 224)

# --- GPU Check ---
function gpu_available()
    try
        return CUDA.functional()
    catch e
        @warn "GPU unavailable: $e"
        return false
    end
end

# --- Model Architecture ---
function build_ave_model(;n_classes=length(CLASSES))
    cnn = Chain(
        Conv((3,3), 3=>64, relu, pad=(1,1)),
        BatchNorm(64),
        MaxPool((2,2)),
        Conv((3,3), 64=>128, relu, pad=(1,1)),
        BatchNorm(128),
        MaxPool((2,2)),
        Conv((3,3), 128=>256, relu, pad=(1,1)),
        BatchNorm(256),
        MaxPool((2,2)),
        Conv((3,3), 256=>512, relu, pad=(1,1)),
        BatchNorm(512),
        MaxPool((2,2))
    )
    
    classifier = Chain(
        Flux.flatten,
        Dense(512*14*14, 1024, relu),
        Dropout(0.5),
        Dense(1024, n_classes),
        softmax
    )
    
    model = Chain(cnn, classifier)
    return gpu_available() ? model |> gpu : model
end

# --- Data Loading ---
function load_labeled_data(data_dir; max_samples_per_class=500)
    data = []
    labels = []
    
    for (class_idx, class_name) in enumerate(CLASSES)
        paths = Glob.glob("$data_dir/Type_$(class_idx)/*.jpg")
        isempty(paths) && @warn "No images found for $class_name in $data_dir"
        
        paths = paths[1:min(end, max_samples_per_class)]
        @showprogress "Loading $class_name..." for path in paths
            try
                img = load(path) |> x->imresize(x, IMG_SIZE) |> x->channelview(x)/255f0
                push!(data, Float32.(permutedims(img, (3,2,1))))
                push!(labels, class_idx)
            catch e
                @warn "Skipping $path: $e"
            end
        end
    end
    
    if isempty(data)
        error("No valid images found in $data_dir")
    end
    
    # Stack all images into a single array (width × height × channels × samples)
    data = cat(data..., dims=4)
    labels = Int.(labels)
    
    shuffle_idx = randperm(length(labels))
    return (data[:,:,:,shuffle_idx], labels[shuffle_idx])
end

function load_unlabeled_data(data_dir)
    paths = Glob.glob("$data_dir/*.jpg")
    isempty(paths) && error("No images found in $data_dir")
    
    data = []
    filenames = []
    
    @showprogress "Loading test images..." for path in paths
        try
            img = load(path) |> x->imresize(x, IMG_SIZE) |> x->channelview(x)/255f0
            push!(data, Float32.(permutedims(img, (3,2,1))))
            push!(filenames, basename(path))
        catch e
            @warn "Skipping $path: $e"
        end
    end
    
    isempty(data) && error("No valid images could be loaded from $data_dir")
    data = cat(data..., dims=4)
    return (data, filenames)
end

# --- Training ---
function train_ave_model(;epochs=30, batch_size=32, lr=0.001)
    model = build_ave_model()
    model_cpu = cpu(model)  # CPU version for saving
    
    # Load labeled training data
    train_data, train_labels = try
        load_labeled_data("images/train")
    catch e
        @error "Training data loading failed" exception=e
        return nothing
    end
    
    # Create validation split (20%)
    split_idx = floor(Int, 0.8 * size(train_data, 4))
    val_data = train_data[:,:,:,split_idx+1:end]
    val_labels = train_labels[split_idx+1:end]
    train_data = train_data[:,:,:,1:split_idx]
    train_labels = train_labels[1:split_idx]
    
    # Create data loaders
    train_loader = DataLoader((train_data, train_labels), 
                            batchsize=min(batch_size, size(train_data, 4)), 
                            shuffle=true)
    val_loader = DataLoader((val_data, val_labels), 
                          batchsize=min(batch_size, size(val_data, 4)), 
                          shuffle=false)
    
    # Define loss function
    
    function loss(x, y)
        y_hat = model(x)
        reg = sum(p -> sum(abs, p), params(model))  # L1 regularization
        return crossentropy(y_hat, onehotbatch(y, 1:length(CLASSES))) + 0.0001f0 * reg
    end
    
    # Set up optimizer - using new explicit style
    opt = Optimisers.Adam(lr)
    state = Optimisers.setup(opt, model)
    
    # Training loop
    for epoch in 1:epochs
        for (x, y) in train_loader
            x = gpu_available() ? gpu(x) : x
            grads = gradient(model) do m
                loss(x, y)
            end
            state, model = Optimisers.update(state, model, grads[1])
        end
        
        # Validation
        val_acc = 0.0
        for (x, y) in val_loader
            x = gpu_available() ? gpu(x) : x
            preds = model(x)
            val_acc += mean(onecold(preds, 1:length(CLASSES)) .== y)
        end
        val_acc /= length(val_loader)
        
        @info "Epoch $epoch: Val Acc = $(round(val_acc*100, digits=2))%"
        gpu_available() && CUDA.reclaim()
    end
    
    # Save both GPU and CPU versions
    model_cpu = cpu(model)
    BSON.@save "ave_model.bson" model model_cpu
    return model
end

# --- Prediction on Unlabeled Data ---
function predict_unlabeled(model, test_dir; output_csv="predictions.csv")
    test_data, filenames = try
        load_unlabeled_data(test_dir)
    catch e
        @error "Test data loading failed" exception=e
        return nothing
    end
    
    predictions = []
    for i in 1:size(test_data, 4)
        img = test_data[:,:,:,i:i]  # Keep batch dimension
        img = gpu_available() ? gpu(img) : img
        probs = model(img)
        pred_class = CLASSES[onecold(probs)[1]]
        confidence = maximum(probs)
        push!(predictions, (filename=filenames[i], class=pred_class, confidence))
    end
    
    # Save to CSV
    df = DataFrame(predictions)
    CSV.write(output_csv, df)
    @info "Predictions saved to $output_csv"
    
    return df
end

# --- Main Execution ---
try
    # 1. Train model
    model = train_ave_model()
    isnothing(model) && exit(1)
    
    # 2. Predict on unlabeled test data
    test_results = predict_unlabeled(model, "images/test")
    isnothing(test_results) && exit(1)
    
    # 3. Print sample predictions
    println("\nSample predictions:")
    for row in eachrow(test_results[1:min(5, nrow(test_results))])
        println("$(row.filename): $(row.class) ($(round(row.confidence*100))% confidence)")
    end
    
catch e
    @error "AVE System Error" exception=(e, catch_backtrace())
finally
    gpu_available() && CUDA.reclaim()
end