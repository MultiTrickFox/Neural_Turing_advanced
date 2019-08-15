using AutoGrad: Param, @diff, value, grad
using Knet: relu, sigm, tanh, softmax

normalize(arr) =
begin
    norm = sqrt(sum(arr .^2))
    if norm != 0
        arr ./ norm
    else
        arr
    end
end



in_size       = 12
hidden_size   = 20
out_size      = 12

memory_size   = 128
location_size = 32


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


mutable struct Processor
    layer_1::Recurrent
    layer_2::Recurrent
    key_creator::Recurrent
    importance_creator::Recurrent

Processor(in_size, hidden_size, out_size, location_size) = new(
    Recurrent(in_size+location_size, hidden_size),
    Recurrent(hidden_size          , out_size),
    Recurrent(out_size             , location_size),
    Recurrent(out_size             , 1),
)
end

(processor::Processor)(input, memory_attended) =
begin
    output = processor.layer_2(processor.layer_1(hcat(input, memory_attended)))
    key = processor.key_creator(output)
    importance = (processor.importance_creator(output).+1)./2

output, key, importance
end

mutable struct Reader
    read_focus::Recurrent

Reader(in_size, location_size) = new(
    Recurrent(in_size, memory_size)
)
end

(reader::Reader)(input, memory) =
begin
    focused = softmax(reader.read_focus(input))
    memory_attended = sum([focus * location for (focus, location) in zip(focused, memory)])

memory_attended
end

mutable struct Writer
    memory_creator::Recurrent

Writer(out_size, location_size) = new(
    Recurrent(out_size, location_size),
)
end

(writer::Writer)(output, key, importance, memory) =
begin
    new_memory = importance .* writer.memory_creator(output)
    memory_normalized = normalize.(memory)
    key_normalized = normalize(key)
    distances = [sum(key .* location_normalized) for location_normalized in memory_normalized]
    attendeds = softmax(distances)
    memory = [(1 .- importance) .* location + new_memory .* attention for (location, attention) in zip(memory, attendeds)]

memory
end


mutable struct Model
    processor::Processor
    reader::Reader
    writer::Writer

Model(in_size, hidden_size, out_size, memory_size, location_size) = new(
    Processor(in_size, hidden_size, out_size, location_size),
    Reader(in_size, location_size),
    Writer(out_size, location_size),
)
end

(model::Model)(input, memory) =
begin
    focused_memory = model.reader(input, memory)
    output, key, importance = model.processor(input, focused_memory)
    new_memory = model.writer(output, key, importance, memory)

output, new_memory
end


propogate_time_series(sequence, model; memory=nothing) =
begin
    # memory == nothing ? [zeros(1, location_size) for _ in 1:memory_size] : ()
    memory = [zeros(1, location_size) for _ in 1:memory_size]

    response = []
    for timestep in sequence
        output, memory = model(timestep, memory)
        push!(response, output)
    end

    for component_name in fieldnames(typeof(model))
        component = getfield(model, component_name)
        for layer_name in fieldnames(typeof(component))
            layer = getfield(component, layer_name)
            layer.state = Param(zeros(size(layer.state)))
        end
    end

response
end



##TESTS

# model = Model(in_size, hidden_size, out_size, memory_size, location_size)
# memory = [zeros(1, location_size) for _ in 1:memory_size]
#
# sample_input = randn(1, in_size)
# sample_output = randn(1, out_size)
#
# output, new_memory = model(sample_input, memory)

##TESTS2

sample_timeserie = [randn(1, in_size) for _ in 1:10]
response = propogate_time_series(sample_timeserie, model)
