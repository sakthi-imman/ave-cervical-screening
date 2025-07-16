using Images, FileIO, Glob, MLUtils
using Flux
using Flux: onehotbatch, onecold, crossentropy
using Random

# --- 1. Load & preprocess images ---
function load_images(path::String)
    classes = ["Type_1", "Type_2", "Type_3"]
    images, labels = [], []
    for (i, label) in enumerate(classes)
        for file in glob("*", joinpath(path, label))
            try
                img = load(file)
                img = imresize(img, (224, 224))  # Resize
                img = channelview(img)                   # (3, H, W)
                img = permutedims(img, (2, 3, 1))         # (H, W, 3)
                img = Float32.(img) ./ 255                # Normalize
                push!(images, img)                        # (224, 224, 3)
                push!(labels, i)
            catch e
                println("Skipping broken image $file: $e")
            end
        end
    end
    return images, labels
end

X, y = load_images("images/train")

# --- 2. Define model ---
model = Chain(
    Conv((3,3), 3=>16, relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, relu),
    MaxPool((2,2)),
    Flux.flatten,
    Dense(32 * 54 * 54, 128, relu),
    Dense(128, 3),
    softmax
)

# --- 3. Preprocessing helper ---
function prepare_image(x)
    reshape(x, 224, 224, 3, 1)  # add batch dim
end

# --- 4. Prepare data ---
labels = onehotbatch(y, 1:3)
X, labels = shuffleobs((X, labels))
train_data = [(x, y) for (x, y) in zip(X, labels)]

# --- 5. Loss & optimizer ---
opt = ADAM()
loss(x, y) = crossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# --- 6. Sanity Check ---
println("Sample shape: ", size(train_data[1][1]))
@show size(prepare_image(train_data[1][1]))  # Should be (224, 224, 3, 1)

# --- 7. Training ---
for epoch in 1:10
    epoch_loss = 0.0
    for (x, y) in train_data
        try
            xbat = prepare_image(x)
            gs = gradient(model -> loss(model(xbat), y), model)
            Flux.Optimise.update!(opt, model, gs)
            epoch_loss += loss(xbat, y)
        catch e
            println("Skipping a sample due to: ", e)
        end
    end
    println("Epoch $epoch complete — Loss: $(epoch_loss / length(train_data))")
end
