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



normalize(arr) = # why cant u normalize a 2d array LinearAlgebra
begin
    norm  = sqrt(sum(arr .^2))
    norm != 0 ? arr./norm : arr
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
    read_key::Layer
    write_key::Layer
    memory_creator::Layer

Processor(in_size, hidden_size, out_size, location_size, hm_readers) = new(
    Layer(in_size+location_size*hm_readers, hidden_size),
    Layer(hidden_size, out_size),
    Layer(hidden_size, location_size),
    Layer(hidden_size, location_size),
    Layer(hidden_size, location_size),
)
end

(processor::Processor)(input, memory_attended) =
begin
    hidden = processor.layer_in(hcat(input, memory_attended))
    output = processor.layer_out(hidden)
    read_key = processor.read_key(hidden)
    write_key = processor.write_key(hidden)
    new_data = processor.memory_creator(hidden)

output, read_key, write_key, new_data
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
    data_interpret::Layer
    location_attend::Layer
    location_free::Layer
    location_alloc::Layer

Writer(location_size) = new(
    Layer(location_size*2, location_size),
    Layer(location_size*2, 1),
    Layer(location_size*2, location_size),
    Layer(location_size*2, location_size),
)
end

(writer::Writer)(memory, write_key, new_data) =
begin
    new_content = writer.data_interpret(hcat(write_key, new_data))
    memory_attentions = softmax([writer.location_attend(hcat(write_key, location))[end] for location in memory])
    free_attentions = [writer.location_free(hcat(write_key, location)) for location in memory]
    alloc_attentions = [writer.location_alloc(hcat(write_key, location)) for location in memory]
    new_memory = [(1 .- attention) .* location + (attention) .* (location .* free + new_content .* alloc) for (location, attention, free, alloc) in zip(memory, memory_attentions, free_attentions, alloc_attentions)]
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
    [Writer(location_size) for _ in 1:hm_writers],
)
end

(model::Model)(input, memory, read_key) =
begin
    memory_attended = hcat([reader(read_key, memory) for reader in model.readers]...)
    output, read_key, write_key, new_data = model.processor(input, memory_attended)
    new_memory = sum([writer(memory, write_key, new_data) for writer in model.writers]) ./ hm_writers

output, new_memory, read_key
end



propogate_timeseries(sequence, model; memory=nothing, read_key=nothing) =
begin
    memory == nothing ? memory = [randn(1, location_size) for _ in 1:memory_size] : ()
    read_key == nothing ? read_key = randn(1, location_size) : ()

    response = []
    for timestep in sequence
        output, memory, read_key = model(timestep, memory, read_key)
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
memory = [randn(1, location_size) for _ in 1:memory_size]

sample_input = randn(1, in_size)
sample_output = randn(1, out_size)

output, new_memory = model(sample_input, memory, randn(1,location_size))

##TESTS2

sample_timeserie = [randn(1, in_size) for _ in 1:5]

response = propogate_timeseries(sample_timeserie, model, memory=memory)
