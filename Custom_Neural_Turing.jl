using AutoGrad: Param, @diff, value, grad
using Knet: relu, sigm, tanh, softmax



in_size       = 12
hidden_size   = 20
out_size      = 12

memory_size   = 128
location_size = 32

hm_readers    = 2
hm_writers    = 1

Layer_Type    = :FeedForward
Gate_Type     = :tanh # sigm only free/alloc, tanh reinforces too



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


Layer, gate = eval(Layer_Type), eval(Gate_Type)


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

gate.(in * layer.w + layer.b)
end



mutable struct Processor
    layer_in::Layer
    layer_out::Layer
    read_key::Layer
    write_key::Layer

Processor(in_size, hidden_size, out_size, location_size, hm_readers) = new(
    Layer(in_size+location_size*hm_readers, hidden_size),
    Layer(hidden_size                     , out_size),
    Layer(out_size                        , location_size),
    Layer(out_size                        , location_size),
)
end

(processor::Processor)(input, memory_attended) =
begin
    output = processor.layer_out(processor.layer_in(hcat(input, memory_attended)))
    read_key = processor.read_key(output)
    write_key = processor.write_key(output)

output, read_key, write_key
end

mutable struct Reader
    location_attend::Layer

Reader(location_size) = new(
    Layer(location_size*2, 1),
)
end

(reader::Reader)(read_key, memory) =
begin
    memory_attentions = softmax([reader.location_attend(hcat(read_key, location))[end] for location in memory])
    memory_attended = sum([location .* attention for (location, attention) in zip(memory, memory_attentions)])

memory_attended
end

mutable struct Writer
    memory_creator::Layer
    location_attend::Layer
    location_free::Layer
    location_alloc::Layer

Writer(out_size, location_size) = new(
    Layer(out_size, location_size),
    Layer(location_size*2, 1),
    Layer(location_size*2, location_size),
    Layer(location_size*2, location_size),
)
end

(writer::Writer)(output, memory, write_key) =
begin
    new_data = writer.memory_creator(output)
    memory_attentions = softmax([writer.location_attend(hcat(write_key, location))[end] for location in memory])
    free_attentions = [writer.location_free(hcat(write_key, location)) for location in memory]
    alloc_attentions = [writer.location_alloc(hcat(write_key, location)) for location in memory]
    new_memory = [(1 .- attention) .* location + (attention) .* (location .* free + new_data .* alloc) for (location, attention, free, alloc) in zip(memory, memory_attentions, free_attentions, alloc_attentions)]
    # [location .* (1 .- location_attention) .* (1 .- free_attention) .+ new_data .* location_attention .* alloc_attention for (location, attention, free, alloc) in zip(memory, memory_attentions, free_attentions, alloc_attentions)]

new_memory
end



mutable struct Model
    processor::Processor
    readers::Array{Reader}
    writers::Array{Writer}

Model(in_size, hidden_size, out_size, memory_size, location_size, hm_readers, hm_writers) = new(
    Processor(in_size, hidden_size, out_size, location_size, hm_readers),
    [Reader(location_size) for _ in 1:hm_readers],
    [Writer(out_size, location_size) for _ in 1:hm_writers],
)
end

(model::Model)(input, memory, read_key) =
begin
    memory_attended = hcat([reader(read_key, memory) for reader in model.readers]...)
    output, read_key, write_key = model.processor(input, memory_attended)
    new_memory = sum([writer(output, memory, write_key) for writer in model.writers]) ./ hm_writers

output, new_memory, read_key
end



propogate_timeseries(sequence, model; memory=nothing, read_key=nothing) =
begin
    memory == nothing ? memory = [zeros(1, location_size) for _ in 1:memory_size] : ()
    read_key == nothing ? read_key = randn(1, location_size) : ()

    response = []
    for timestep in sequence
        output, memory, read_key = model(timestep, memory, read_key)
        @show memory[end]
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

model = Model(in_size, hidden_size, out_size, memory_size, location_size, hm_readers, hm_writers)
memory = [zeros(1, location_size) for _ in 1:memory_size]

sample_input = randn(1, in_size)
sample_output = randn(1, out_size)

output, new_memory = model(sample_input, memory, randn(1,location_size))

##TESTS2

sample_timeserie = [randn(1, in_size) for _ in 1:5]

response = propogate_timeseries(sample_timeserie, model, memory=memory)
