using AutoGrad: Param, @diff, value, grad
using Knet: relu, sigm, tanh, softmax


in_size       = 12
hidden_size   = 20
out_size      = 10

state_size    = 10
memory_size   = 20
location_size = 64



mutable struct Model

    intermediate_state::Param

    mem_attend_value::Param
    mem_attend_func::Param

    hidden_state::Param

    output::Param
    final_state::Param

    out_value::Param
    out_func::Param

    alloc_val::Param
    dealloc_val::Param
    alloc_fn::Param
    dealloc_fn::Param


Model(in_size, hidden_size, out_size, state_size, location_size) = new(

    Param(randn(in_size+state_size, state_size)),

    Param(randn(state_size+location_size, 1)),
    Param(randn(state_size+location_size, 1)),

    Param(randn(in_size+location_size, hidden_size)),

    Param(randn(hidden_size, out_size)),
    Param(randn(hidden_size, state_size)),

    Param(randn(hidden_size, location_size)),
    Param(randn(hidden_size, location_size)),

    Param(randn(location_size*2, 1)),
    Param(randn(location_size*2, 1)),
    Param(randn(location_size*2, 1)),
    Param(randn(location_size*2, 1)),

)
end

(model::Model)(in, state, memory) =
begin

    values, funcs = memory

    intermediate_state = tanh.(hcat(in, state) * model.intermediate_state)
    val_attentions = softmax([(hcat(intermediate_state, location) * model.mem_attend_value)[end] for location in values])
    fn_attentions  = softmax([(hcat(intermediate_state, location) * model.mem_attend_func)[end] for location in funcs])
    attended_val = sum([location .* attention for (location, attention) in zip(values, val_attentions)])
    attended_fn = sum([location .* attention for (location, attention) in zip(funcs, val_attentions)])

    hidden_input = hcat(in, attended_val .* attended_fn)
    hidden_state = tanh.(hidden_input * model.hidden_state)

    output = tanh.(hidden_state * model.output)
    final_state = tanh.(hidden_state * model.final_state)

    out_value = tanh.(hidden_state * model.out_value)
    out_func = tanh.(hidden_state * model.out_func)

    val_attentions = softmax([(hcat(final_state, location) * model.mem_attend_value)[end] for location in values])
    fn_attentions  = softmax([(hcat(final_state, location) * model.mem_attend_func)[end] for location in funcs])

    val_allocs = [sigm.(hcat(out_value, location) * model.alloc_val) for location in values]
    val_deallocs = [sigm.(hcat(out_value, location) * model.dealloc_val) for location in values]
    fn_allocs = [sigm.(hcat(out_value, location) * model.alloc_fn) for location in funcs]
    fn_deallocs = [sigm.(hcat(out_value, location) * model.dealloc_fn) for location in funcs]

    new_values = [(1 .- val_attention) * location + val_attention * (val_dealloc .* location + val_alloc .* out_value) for (location, val_attention, val_dealloc, val_alloc) in zip(values, val_attentions, val_deallocs, val_deallocs)]
    new_funcs = [(1 .- fn_attention) * location + fn_attention * (fn_dealloc .* location + fn_alloc .* out_func) for (location, fn_attention, fn_dealloc, fn_alloc) in zip(funcs, fn_attentions, fn_deallocs, fn_deallocs)]
    new_memory = (new_values, new_funcs)


output, final_state, new_memory
end



in = randn(1, in_size)
state = randn(1, state_size)
memory = [[randn(1, location_size) for _ in 1:memory_size] for _ in 1:2]


model = Model(in_size, hidden_size, out_size, state_size, location_size)


@show state

in, state, memory = model(in, state, memory)

@show state
