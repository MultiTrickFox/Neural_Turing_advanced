using AutoGrad: Param, @diff, value, grad
using Knet: relu, sigm, tanh, softmax


in_size       = 12
hidden_size   = 20
out_size      = 10

memory_size   = 20
location_size = 64



mutable struct Model


    read_key_data::Param
    read_key_func::Param
    read_data_similarity::Param
    read_func_similarity::Param

    read_intensity_data::Param
    read_intensity_func::Param
    read_as_data::Param
    read_as_func::Param

    func_finalize::Param
    hidden_layer::Param

    output::Param
    data_out::Param
    func_out::Param


    write_key_data::Param
    write_key_func::Param
    write_data_similarity::Param
    write_func_similarity::Param

    write_intensity_data::Param
    write_intensity_func::Param

    alloc_data::Param
    free_data::Param
    alloc_func::Param
    free_func::Param


Model(in_size, hidden_size, out_size, location_size) = new(


    Param(randn(in_size, location_size)),
    Param(randn(in_size, location_size)),
    Param(randn(location_size*2, 1)),
    Param(randn(location_size*2, 1)),

    Param(randn(in_size, location_size)),
    Param(randn(in_size, location_size)),
    Param(randn(in_size+location_size, 1)),
    Param(randn(in_size+location_size, 1)),


    Param(randn(in_size+location_size, in_size)),
    Param(randn(in_size*2+location_size, hidden_size)),

    Param(randn(hidden_size, out_size)),
    Param(randn(hidden_size, location_size)),
    Param(randn(hidden_size, location_size)),


    Param(randn(hidden_size, location_size)),
    Param(randn(hidden_size, location_size)),
    Param(randn(location_size*2, 1)),
    Param(randn(location_size*2, 1)),

    Param(randn(hidden_size, location_size)),
    Param(randn(hidden_size, location_size)),

    Param(randn(location_size*2, 1)),
    Param(randn(location_size*2, 1)),
    Param(randn(location_size*2, 1)),
    Param(randn(location_size*2, 1)),

)
end

(model::Model)(in, memory) =
begin


    data, funcs = memory


    read_key_data = in * model.read_key_data
    read_key_func = in * model.read_key_func
    attentions_data = softmax([(hcat(location, read_key_data) * model.read_data_similarity)[1] for location in data])
    attentions_funcs = softmax([(hcat(location, read_key_func) * model.read_func_similarity)[1] for location in funcs])
    attended_data = sum([location .* attention for (location, attention) in zip(data, attentions_data)])
    attended_func = sum([location .* attention for (location, attention) in zip(funcs, attentions_funcs)])

    read_intensity_data = in * model.read_intensity_data
    read_intensity_func = in * model.read_intensity_func
    read_as_data = hcat(in, attended_data) * model.read_as_data
    read_as_func = hcat(in, attended_func) * model.read_as_func


    data_for_hidden = attended_data .* read_intensity_data
    func_for_hidden = hcat(in, attended_func .* read_intensity_func) * model.func_finalize

    data_to_hidden = data_for_hidden .* read_as_data
    func_to_hidden = in .* func_for_hidden .* read_as_func

    hidden = hcat(in, data_to_hidden, func_to_hidden) * model.hidden_layer

    output = hidden * model.output
    data_out = hidden * model.data_out
    func_out = hidden * model.func_out


    write_key_data = hidden * model.write_key_data
    write_key_func = hidden * model.write_key_func
    attentions_data = softmax([(hcat(location, write_key_data) * model.write_data_similarity)[1] for location in data])
    attentions_funcs = softmax([(hcat(location, write_key_func) * model.write_func_similarity)[1] for location in funcs])
    attended_data = sum([location .* attention for (location, attention) in zip(data, attentions_data)])
    attended_func = sum([location .* attention for (location, attention) in zip(funcs, attentions_funcs)])

    write_intensity_data = hidden * model.write_intensity_data
    write_intensity_func = hidden * model.write_intensity_func

    alloc_data = hcat(attended_data, data_out) * model.alloc_data
    free_data = hcat(attended_data, data_out) * model.free_data
    alloc_func = hcat(attended_func, func_out) * model.alloc_func
    free_func = hcat(attended_func, func_out) * model.free_func


    data_to_memory = data_out .* write_intensity_data .* alloc_data
    func_to_memory = func_out .* write_intensity_func .* alloc_func


    new_data = [(1 .- attention) .* location + attention .* (location .* (1 .- free_data) + data_to_memory)
            for (location, attention) in zip(data, attentions_data)]
    new_funcs = [(1 .- attention) .* location + attention .* (location .* (1 .- free_func) + func_to_memory)
            for (location, attention) in zip(funcs, attentions_funcs)]



output, (new_data, new_funcs)

end



in = randn(1, in_size)

model = Model(in_size, hidden_size, out_size, location_size)

memory = [[randn(1, location_size) for _ in 1:memory_size] for _ in 1:2]



out, memory = model(in, memory)
