# AVE Paper Replication - Full Julia Script with 10-Fold Cross-Validation

using Images, FileIO, Glob, MLUtils
using MLUtils: kfolds
using Flux
using Flux: onehotbatch, onecold, crossentropy
using BSON: @save
using Random, StatsBase
using MLDataPattern: batchview


function ensure_dataset_structure()
    base_path = "images/train"
    
    # Create base train folder if not exists 
    if !isdir(base_path)
        println("📂 Creating base path: $base_path")
        mkpath(base_path)
    end

end

# Call before loading images
ensure_dataset_structure()

# 1. Load & Preprocess Images
function load_images(path::String)
    classes = ["Type_1", "Type_2", "Type_3"]
    images, labels = [], []

    total = 0
    for (i, label) in enumerate(classes)
        folder_path = joinpath(path, label)
        files = Glob.glob("*", folder_path)

        println("🔍 Found $(length(files)) files in $label")

        for file in files
            total += 1
            try
                img = load(file)
                img = imresize(img, (224, 224))
                img = channelview(img) |> x -> permutedims(x, (2, 3, 1))
                img = Float32.(img) ./ 255

                if size(img) == (224, 224, 3)
                    push!(images, img)
                    push!(labels, i)
                else
                    println("⚠️ Skipped image with wrong size: ", size(img), " — ", file)
                end
            catch e
                println("❌ Corrupted image: $file — $e")
            end
        end
    end

    println("✅ Successfully loaded $(length(images)) / $total images.")
    return images, labels
end

# 2. Model Definition
function build_model()
    Chain(
        Conv((3,3), 3=>16, relu), MaxPool((2,2)),
        Conv((3,3), 16=>32, relu), MaxPool((2,2)),
        flatten,
        Dense(32 * 54 * 54, 128, relu),
        Dense(128, 3), softmax
    )
end
# 3. Accuracy Evaluation
function evaluate_accuracy(data, model)
    correct = 0
    total = 0

    for (x, y) in data
        xbat = reshape(x, 224, 224, 3, 1)  # (H, W, C, B)
        pred = model(xbat)

        # Get predicted and true class indices (1, 2, or 3)
        pred_class = onecold(pred, 1:3)
        true_class = onecold(y, 1:3)  # Make sure y is a one-hot vector

        if !isempty(pred_class) && !isempty(true_class)
            correct += pred_class == true_class
            total += 1
        else
            println("⚠️ Skipped due to invalid prediction/label.")
        end
    end

    return total > 0 ? correct / total : 0.0
end

# 4. Utility to split input-label tuples
function unzip(pairs)
    xs = map(first, pairs)
    ys = map(last, pairs)
    return (xs, ys)
end
  

# 5. Load Data
X, y = load_images("images/train")

@show length(readdir("images/train/Type_1"))
@show length(readdir("images/train/Type_2"))
@show length(readdir("images/train/Type_3"))


labels = onehotbatch(y, 1:3)
folds = collect(kfolds((X, labels), k=10))

fold_accuracies = Float64[]
batch_size = 32

for fold in 1:10
    println("\n🔁 Fold $fold")
    model = build_model()
    opt = ADAM()
    opt_state = Flux.setup(opt, model)

    # UNPACK DATA CORRECTLY — FIXED
    (train_X, train_Y), (val_X, val_Y) = folds[fold]

    # Confirm if fold has training data
    if length(train_X) == 0
        println("⚠️ Skipping fold $fold — No training data.")
        continue
    end

    # Convert training data to (image, label vector) pairs
    train_data = [(train_X[i], train_Y[:, i]) for i in 1:length(train_X)]
    val_data   = [(val_X[i],   val_Y[:, i])   for i in 1:length(val_X)]

    # Batch labels and images correctly
    train_Y_matrix = hcat([y for (_, y) in train_data]...)
    train_X_batchable = [x for (x, _) in train_data]
    batched_train = batchview((train_X_batchable, train_Y_matrix), size = batch_size)

    for epoch in 1:10
        epoch_loss = 0.0
        for (xb, yb) in batched_train
            xb = cat([reshape(x, 224, 224, 3, 1) for x in xb]...; dims=4)
            gs = gradient(m -> crossentropy(m(xb), yb), model)
            Flux.Optimise.update!(opt_state, model, gs[1])
            epoch_loss += crossentropy(model(xb), yb)
        end
        acc = evaluate_accuracy(val_data, model)
        println("Fold $fold | Epoch $epoch — Loss: $(round(epoch_loss, digits=2)) — Val Acc: $(round(acc*100, digits=2))%")
        @save "model_fold$(fold)_epoch$(epoch).bson" model
        if epoch == 10
            push!(fold_accuracies, acc)
        end
    end
end

# 7. Final Accuracy Summary
println("\n📊 Fold-wise Validation Accuracies:")
for (i, acc) in enumerate(fold_accuracies)
    println("Fold $i: $(round(acc * 100, digits=2))%")
end

avg_acc = mean(fold_accuracies)
println("\n✅ Average Accuracy over 10 folds: $(round(avg_acc * 100, digits=2))%")

# 8. Compare with Original Paper
original_accuracy = 0.856  # AVE paper reported ~85.6% overall accuracy
println("\n📈 Comparison with AVE Paper:")
println("Original Paper Accuracy : $(round(original_accuracy * 100, digits=2))%")
println("Your Replication Accuracy: $(round(avg_acc * 100, digits=2))%")

diff = original_accuracy - avg_acc
println("Difference: $(round(diff * 100, digits=2))%")
