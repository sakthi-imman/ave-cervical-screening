### main.jl

using CSV, DataFrames, FilePathsBase
using Images, ImageIO, ImageTransformations
using Flux, Statistics, BSON, Random

println("🚀 Starting AVE-style cervical cancer classification pipeline...")

# ========== STEP 1: Dataset Setup ==========

# Paths
input_dir = "images/train"  # Kaggle's original folder structure: train/Type_1, etc.
output_img_dir = "images"
label_csv_path = "data/labels.csv"

label_map = Dict(
    "Type_1" => "normal",
    "Type_2" => "hpv",
    "Type_3" => "precancer"
)

# Create folders if not exist
isdir(output_img_dir) || mkpath(output_img_dir)
isdir(dirname(label_csv_path)) || mkpath(dirname(label_csv_path))

filenames, labels = String[], String[]

println("🗂️  Flattening image folders and generating labels.csv...")

for (folder, labelname) in label_map
    files = readdir(joinpath(input_dir, folder))
    for f in files
        src = joinpath(input_dir, folder, f)
        dest = joinpath(output_img_dir, f)
        cp(src, dest; force=true)
        push!(filenames, f)
        push!(labels, labelname)
    end
end

df = DataFrame(filename=filenames, label=labels)
CSV.write(label_csv_path, df)
println("✅ Dataset ready with $(length(filenames)) images and labels.csv saved.")

# ========== STEP 2: Load + Preprocess Images ==========

println("🧼 Preprocessing images...")

labels_df = CSV.read(label_csv_path, DataFrame)
classes = unique(labels_df.label)
class_to_int = Dict(c => i for (i, c) in enumerate(classes))
num_classes = length(classes)

function load_and_preprocess(img_path::String)
    img = load(img_path)
    img = imresize(img, (224, 224))
    img = float32.(channelview(img))  # CHW format
    return img
end

X = [load_and_preprocess(joinpath(output_img_dir, row.filename)) for row in eachrow(labels_df)]
X_tensor = cat(X..., dims=4)  # shape: 224×224×3×N
Y = Flux.onehotbatch([class_to_int[row.label] for row in eachrow(labels_df)], 1:num_classes)

# ========== STEP 3: Train/Test Split ==========
println("🔀 Splitting into training and test sets...")

n = size(X_tensor, 4)
shuffle = randperm(n)
X_tensor, Y = X_tensor[:, :, :, shuffle], Y[:, shuffle]

ntrain = Int(round(0.85 * n))
X_train, X_test = X_tensor[:, :, :, 1:ntrain], X_tensor[:, :, :, ntrain+1:end]
Y_train, Y_test = Y[:, 1:ntrain], Y[:, ntrain+1:end]

# ========== STEP 4: Define CNN Model ==========

model = Chain(
    Conv((3,3), 3=>16, relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, relu),
    MaxPool((2,2)),
    Dropout(0.3),
    flatten,
    Dense(32*54*54 => 128, relu),
    Dropout(0.3),
    Dense(128 => num_classes),
    softmax
)

loss(x, y) = Flux.crossentropy(model(x), y)
opt = ADAM()

function accuracy(x, y)
    ŷ = model(x)
    mean(Flux.onecold(ŷ) .== Flux.onecold(y)) * 100
end

# ========== STEP 5: Training Loop ==========

println("🏋️ Starting training...")
epochs = 10

for epoch in 1:epochs
    for (x, y) in zip(eachslice(X_train, dims=4), eachcol(Y_train))
        gs = gradient(Flux.params(model)) do
            loss(x, y)
        end
        Flux.Optimise.update!(opt, Flux.params(model), gs)
    end

    # Logging
    train_acc = accuracy(X_train, Y_train)
    test_acc = accuracy(X_test, Y_test)
    println("📈 Epoch $epoch → Train Acc: $(round(train_acc, digits=2))% | Test Acc: $(round(test_acc, digits=2))%")

    if epoch % 5 == 0
        BSON.@save "checkpoints/model_epoch_$epoch.bson" model
        println("💾 Model saved at epoch $epoch")
    end
end

println("✅ Training finished.")
