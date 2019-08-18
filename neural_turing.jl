using AutoGrad: Param, @diff, value, grad
using Knet: relu, sigm, tanh, softmax



in_size       = 12
hidden_size   = 20
out_size      = 12

memory_size   = 128
location_size = 32

hm_readers    = 1
hm_writers    = 1

Layer_Type    = :FeedForward



normalize(arr) = # why cant u normalize a 2d array LinearAlgebra
begin
    norm = sqrt(sum(arr .^2))
    if norm != 0
        arr ./ norm
    else
        arr
    end
end


mutable struct Recurrent
    state::Param
    wf1::Param
    wf2::Param
    bf::Param
    wk1::Param
    wk2::Param
    bk::Param
    wi::Param
    bi::Param

Recurrent(in_size,layer_size) = new(
    Param(zeros(1,layer_size)),
    Param(randn(in_size,layer_size)),
    Param(randn(layer_size,layer_size)),
    Param(zeros(1,layer_size)),
    Param(randn(in_size,layer_size)),
    Param(randn(layer_size,layer_size)),
    Param(zeros(1,layer_size)),
    Param(randn(in_size,layer_size)),
    Param(zeros(1,layer_size)),
)
end

(layer::Recurrent)(in) =
begin
    focus  = sigm.(in * layer.wf1 + layer.state * layer.wf2 + layer.bf)
    keep   = sigm.(in * layer.wk1 + layer.state * layer.wk2 + layer.bk)
    interm = tanh.(in * layer.wi  + layer.state .* focus    + layer.bi)
    layer.state = Param(keep .* interm + (1 .- keep) .* layer.state)

layer.state = Param(keep .* interm + (1 .- keep) .* layer.state)
end


mutable struct FeedForward
    w::Param
    b::Param

FeedForward(in_size, layer_size) = new(
    Param(randn(in_size, layer_size)),
    Param(zeros(1, layer_size)),
)
end

(layer::FeedForward)(in) =
begin

tanh.(in * layer.w + layer.b)
end



Layer = eval(Layer_Type)


mutable struct Processor
    layer_in::Layer
    layer_out::Layer
    key_creator::Layer
    importance_creator::Layer

Processor(in_size, hidden_size, out_size, location_size) = new(
    Layer(in_size+location_size, hidden_size),
    Layer(hidden_size          , out_size),
    Layer(out_size             , location_size),
    Layer(out_size             , 1),
)
end

(processor::Processor)(input, memory_attended) =
begin
    output = processor.layer_out(processor.layer_in(hcat(input, memory_attended)))
    key = processor.key_creator(output)
    importance = (processor.importance_creator(output).+1)./2

output, key, importance
end

mutable struct Reader
    read_focus::Layer

Reader(in_size, location_size) = new(
    Layer(in_size, memory_size)
)
end

(reader::Reader)(input, memory) =
begin
    attentions = softmax(reader.read_focus(input)) # (reader.read_focus(input).+1)./2
    memory_attended = sum([location .* attention for (attention, location) in zip(attentions, memory)])

memory_attended
end

mutable struct Writer
    memory_creator::Layer

Writer(out_size, location_size) = new(
    Layer(out_size, location_size),
)
end

(writer::Writer)(output, key, importance, memory) =
begin
    new_data = writer.memory_creator(output)
    memory_normalized = normalize.(memory)
    key_normalized = normalize(key)

    @show size(vcat([key_normalized .* location_normalized for location_normalized in memory_normalized]...))

    # attentions = sum(softmax(vcat([key_normalized .* location_normalized for location_normalized in memory_normalized]...),dims=1),dims=2)

    attentions = softmax([sum(key_normalized .* location_normalized) for location_normalized in memory_normalized])

    @show attentions

    new_memory = [location .* ((1 .- importance) .* (1 .- attention)) + new_data .* (importance .* attention) for (location, attention) in zip(memory, attentions)]

new_memory
end


mutable struct Model
    processor::Processor
    readers::Array{Reader}
    writers::Array{Writer}

Model(in_size, hidden_size, out_size, memory_size, location_size) = new(
    Processor(in_size, hidden_size, out_size, location_size),
    [Reader(in_size, location_size) for _ in 1:hm_readers],
    [Writer(out_size, location_size) for _ in 1:hm_writers],
)
end

(model::Model)(input, memory) =
begin
    memory_attended = sum([reader(input, memory) for reader in model.readers]) ./ hm_readers
    output, key, importance = model.processor(input, memory_attended)
    new_memory = sum([writer(output, key, importance, memory) for writer in model.writers]) ./ hm_writers

output, new_memory
end

propogate_timeseries(sequence, model; memory=nothing) =
begin
    memory == nothing ? memory = [zeros(1, location_size) for _ in 1:memory_size] : ()

    response = []
    for timestep in sequence
        output, memory = model(timestep, memory)
        push!(response, output)
    end

    if Layer_Type == "Recurrent"
        for component_name in fieldnames(typeof(model))
            component = getfield(model, component_name)
            for layer_name in fieldnames(typeof(component))
                layer = getfield(component, layer_name)
                layer.state = Param(zeros(size(layer.state)))
            end
        end
    end

response
end





##TESTS

model = Model(in_size, hidden_size, out_size, memory_size, location_size)
memory = [randn(1, location_size) for _ in 1:memory_size]

sample_input = randn(1, in_size)
sample_output = randn(1, out_size)

output, new_memory = model(sample_input, memory)

##TESTS2

sample_timeserie = [randn(1, in_size) for _ in 1:3]

response = propogate_timeseries(sample_timeserie, model, memory=memory)
