# AVE Paper Replication - Full Julia Script with 10-Fold Cross-Validation

using Images, FileIO, Glob, MLUtils
using MLUtils: kfolds
using Flux
using Flux: onehotbatch, onecold, crossentropy
using BSON: @save
using Random, StatsBase

# 1. Load & Preprocess Images
function load_images(path::String)
    classes = ["Type_1", "Type_2", "Type_3"]
    images, labels = [], []

    for (i, label) in enumerate(classes)
        for file in Glob.glob("*.jpg", joinpath(path, label))
            try
                img = load(file)
                img = imresize(img, (224, 224))
                img = channelview(img) |> x -> permutedims(x, (2, 3, 1))
                img = Float32.(img) ./ 255
                if size(img) == (224, 224, 3)
                    push!(images, img)
                    push!(labels, i)
                end
            catch e
                println("Skipping corrupted image $file: $e")
            end
        end
    end
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
    correct = 0; total = 0
    for (x, y) in data
        xbat = reshape(x, 224, 224, 3, 1)
        pred = model(xbat)
        if isa(y, AbstractArray) && isa(pred, AbstractArray)
            correct += onecold(collect(pred), 1:3) == onecold(collect(y), 1:3)
            total += 1
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
labels = onehotbatch(y, 1:3)
folds = collect(kfolds((X, labels), k=10))

fold_accuracies = Float64[]
batch_size = 32

# 6. Training Loop
for fold in 1:10
    println("\n🔁 Fold $fold")
    model = build_model()
    opt = ADAM()
    opt_state = Flux.setup(opt, model)

    train, val = folds[fold]
    train_data = [(x, y) for (x, y) in zip(train...) if isa(y, AbstractVector) && length(y) == 3]
    val_data = [(x, y) for (x, y) in zip(val...) if isa(y, AbstractVector) && length(y) == 3]

    if isempty(train_data)
        println("⚠️ Skipping fold $fold — No training data.")
        continue
    end

    train_X, train_Y = unzip(train_data)
    train_Y_matrix = hcat(train_Y...)
    batched_train = batchview((train_X, train_Y_matrix), size = batch_size)

    for epoch in 1:3
        epoch_loss = 0.0
        for (xb, yb) in batched_train
            xb = cat([reshape(x, 224, 224, 3, 1) for x in xb]...; dims=4)
            gs = gradient(m -> crossentropy(m(xb), yb), model)
            Flux.Optimise.update!(opt_state, model, gs)
            epoch_loss += crossentropy(model(xb), yb)
        end
        acc = evaluate_accuracy(val_data, model)
        println("Fold $fold | Epoch $epoch — Loss: $(round(epoch_loss, digits=2)) — Val Acc: $(round(acc*100, digits=2))%")
        @save "model_fold$(fold)_epoch$(epoch).bson" model
        if epoch == 3
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
