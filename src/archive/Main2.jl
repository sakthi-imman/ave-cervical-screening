using Images, FileIO, Glob, MLUtils
using Flux
using Flux: onehotbatch, onecold, crossentropy
using Random

function load_images(path::String)
    classes = ["Type_1", "Type_2", "Type_3"]
    images, labels = [], []

    for (i, label) in enumerate(classes)
        for file in Glob.glob("*", joinpath(path, label))
            try
                img = load(file)
                img = imresize(img, (224, 224))           # Ensure size
                img = channelview(img)                    # (3, H, W)
                img = permutedims(img, (2, 3, 1))         # (H, W, 3)
                img = Float32.(img) ./ 255

                if size(img) != (224, 224, 3)
                    println("Skipping image with wrong shape: ", size(img), " — ", file)
                    continue
                end

                push!(images, img)
                push!(labels, i)
            catch e
                println("Skipping corrupted image: $file — $e")
            end
        end
    end
    return images, labels
end

X, y = load_images("images/train")

# Model

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


# Training 
labels = onehotbatch(y, 1:3)
X, labels = shuffleobs((X, labels))
train_data = [(x, y) for (x, y) in zip(X, labels)]

opt = ADAM()
loss(x, y) = crossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

function prepare_image(x)
    if size(x) == (224, 224, 3)
        return reshape(x, 224, 224, 3, 1)
    else
        error("Invalid image shape: $(size(x))")
    end
end



# Sanity Check
println("Validating all image shapes...")
for (i, x) in enumerate(X)
    if size(x) != (224, 224, 3)
        println("❌ Image $i has bad shape: ", size(x))
    end
end
println("✅ All shapes are OK")

println("Checking all image shapes in train_data...")

for (i, (x, y)) in enumerate(train_data)
    println("Sample $i: typeof(x) = ", typeof(x), ", size = ", size(x))
    if !(typeof(x) <: Array{Float32, 3}) || size(x) != (224, 224, 3)
        println("❌ Problematic sample at index $i: shape = $(size(x)), type = $(typeof(x))")
    end
end    

# --- 6. Batching ---
# Enable mini-batch training to significantly improve training speed
batch_size = 32
batched_train = batchview((train_X, train_y), size = batch_size)

# Training Loop
for epoch in 1:10
    epoch_loss = 0.0
    for (x, y) in train_data
        try
            xbat = reshape(x, 224, 224, 3, 1)  # safe now
            gs = gradient(m -> loss(m, xbat, y), model)
            Flux.Optimise.update!(opt, model, gs)
            epoch_loss += loss(model, xbat, y)
        catch e
            println("⚠️ Skipping a sample due to: ", e)
        end
    end
    println("✅ Epoch $epoch complete — Avg Loss: $(epoch_loss / length(train_data))")
end


