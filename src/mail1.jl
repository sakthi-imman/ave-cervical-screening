############## main.jl ################
using CSV, DataFrames, FileIO, Images, ImageTransformations
using Flux, BSON, Statistics, Random
using Base.Threads: @threads
using Printf

# --- CONFIGURATION ---
data_dir = "image/train"                      # path to downloaded Kaggle dataset
labels_csv_path = "data/labels.csv"         # label mapping file
image_size = (224, 224)                # resize all images for uniform input
n_epochs = 10
batch_size = 32
model_save_path = "ave_model.bson"
rng = Random.default_rng()

# --- LOAD LABELS ---
println("🔍 Loading label metadata...")
labels_df = CSV.read(labels_csv_path, DataFrame)
labels_map = Dict(row.filename => row.new_label for row in eachrow(labels_df))
class_names = sort!(unique(values(labels_map)))
label_to_index = Dict(name => i for (i, name) in enumerate(class_names))

# --- LOAD IMAGES ---
function load_image_paths(dir::String)
    image_paths = []
    for (root, _, files) in walkdir(dir)
        for f in files
            if endswith(f, ".jpg") || endswith(f, ".png")
                push!(image_paths, joinpath(root, f))
            end
        end
    end
    return image_paths
end

println("🖼️ Scanning image files...")
image_paths = load_image_paths(data_dir)

# --- IMAGE TRANSFORMATION ---
function preprocess_image(path)
    img = load(path)
    img = Images.channelview(img)
    img = imresize(img, image_size)
    img = Float32.(img) ./ 255.0
    return img
end

# --- CREATE DATASET ---
println("📦 Preprocessing dataset...")

function create_dataset(image_paths::Vector{Any})
    x_data, y_data = [], []
    for path in image_paths
        filename = split(path, "/") |> last
        if haskey(labels_map, filename)
            label = labels_map[filename]
            img_tensor = preprocess_image(path)
            push!(x_data, img_tensor)
            push!(y_data, Flux.onehot(label_to_index[label], 1:length(class_names)))
        end
    end
    return x_data, y_data
end

x_data, y_data = create_dataset(image_paths)

print(image_paths)

# Shuffle and split
n = length(x_data)
perm = randperm(rng, n)
ntrain = Int(round(0.8n)) 
x_train, y_train = x_data[perm[1:ntrain]], y_data[perm[1:ntrain]]
x_test, y_test = x_data[perm[ntrain+1:end]], y_data[perm[ntrain+1:end]]

# --- CREATE BATCHES ---
function get_batches(x, y, batch_size)
    batches = []
    for i in 1:batch_size:length(x)
        idx = i:min(i+batch_size-1, length(x))
        x_batch = cat(x[idx]..., dims=4)    # WHCN
        y_batch = hcat(y[idx]...)
        push!(batches, (x_batch, y_batch))
    end
    return batches
end

train_batches = get_batches(x_train, y_train, batch_size)
test_batches = get_batches(x_test, y_test, batch_size)

# --- MODEL (CNN like paper) ---
println("🔧 Building CNN model...")
model = Chain(
    Conv((3,3), 3=>16, relu), MaxPool((2,2)),
    Conv((3,3), 16=>32, relu), MaxPool((2,2)),
    Conv((3,3), 32=>64, relu), MaxPool((2,2)),
    flatten,
    Dense(64*27*27, 128, relu),
    Dense(128, length(class_names))
)

loss_fn(x, y) = Flux.logitcrossentropy(model(x), y)
accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))

# --- TRAINING LOOP ---
println("🚀 Training begins...")

opt = ADAM(1e-3)
for epoch in 1:n_epochs
    @printf "\nEpoch %d\n" epoch
    for (xb, yb) in train_batches
        grads = Flux.gradient(() -> loss_fn(xb, yb), Flux.params(model))
        Flux.Optimise.update!(opt, Flux.params(model), grads)
    end
    train_acc = mean([accuracy(xb, yb) for (xb, yb) in train_batches])
    test_acc = mean([accuracy(xb, yb) for (xb, yb) in test_batches])
    println("✅ Train Accuracy: $(round(train_acc*100, digits=2))% | Test Accuracy: $(round(test_acc*100, digits=2))%")
end

# --- SAVE MODEL ---
BSON.@save model_save_path model class_names label_to_index
println("💾 Model saved to $model_save_path")
