const TENSOR_1::ITensor = ITensor(1)

function evolve(A::ITensor, H_eff::ITensor, T::Complex, do_normalize::Bool)::ITensor
    new_A, info = exponentiate(H_eff, T, A, eager=true)
    @assert info.converged == 1
    if (do_normalize)
        normalize!(new_A)
    end
    return new_A
end

push_layer!(layers::Vector{ITensor}, site::ITensor, H_site::ITensor) =
    push!(layers, layers[end] * site * H_site * dag(prime(site)))

pop_layer!(layers::Vector{ITensor}) = pop!(layers)

function tdvp1(
    H::MPO,
    psi_0::MPS;
    step_size::Union{ComplexF64,Float64},
    num_steps::Int,
    sweeps_per_time_step::Int,
    max_bond_dim::Int,
    normalize::Bool=true
)::Vector{MPS}

    dt = -im * step_size / sweeps_per_time_step / 2

    num_cells = length(H)
    @assert num_cells == length(psi_0)

    layers_left::Vector{ITensor} = [TENSOR_1]
    layers_right::Vector{ITensor} = [TENSOR_1]

    results = [deepcopy(psi_0)]
    sizehint!(results, num_steps)

    psi = deepcopy(psi_0)

    # Fix the bond dimensions
    # https://itensor.discourse.group/t/how-do-i-set-an-mps-bond-dimension-that-is-higher-than-needed/1637
    psi = +(
        psi,
        0 * random_mps(siteinds(psi); linkdims=max_bond_dim - maximum(linkdims(psi)));
        alg="directsum"
    )
    @assert maximum(linkdims(psi)) == max_bond_dim


    orthogonalize!(psi, 1)
    for site_idx in num_cells:-1:2
        push_layer!(
            layers_right,
            psi[site_idx],
            H[site_idx]
        )
    end

    @showprogress desc = "Calculating Time Evolution" for _ in 2:num_steps
        for _ in 1:sweeps_per_time_step
            for site_idx in 1:num_cells
                psi[site_idx] = evolve(
                    psi[site_idx],
                    layers_left[end] * H[site_idx] * layers_right[end],
                    dt,
                    normalize
                )
                if site_idx != num_cells
                    inds_left = uniqueinds(psi[site_idx], psi[site_idx+1])
                    site_orthogonal, bond = qr(psi[site_idx], inds_left)
                    psi[site_idx] = site_orthogonal
                    push_layer!(
                        layers_left,
                        psi[site_idx],
                        H[site_idx]
                    )
                    new_bond = evolve(
                        bond,
                        layers_left[end] * layers_right[end],
                        -dt,
                        normalize
                    )
                    pop_layer!(layers_right)
                    psi[site_idx+1] *= new_bond
                end
            end
            for site_idx in num_cells:-1:1
                psi[site_idx] = evolve(
                    psi[site_idx],
                    layers_left[end] * H[site_idx] * layers_right[end],
                    dt,
                    normalize
                )
                if site_idx != 1
                    inds_right = uniqueinds(psi[site_idx], psi[site_idx-1])
                    site_orthogonal, bond = qr(psi[site_idx], inds_right)
                    psi[site_idx] = site_orthogonal
                    push_layer!(
                        layers_right,
                        psi[site_idx],
                        H[site_idx]
                    )
                    new_bond = evolve(
                        bond,
                        layers_left[end] * layers_right[end],
                        -dt,
                        normalize
                    )
                    pop_layer!(layers_left)
                    psi[site_idx-1] *= new_bond
                end
            end
        end
        push!(results, copy(psi))
    end
    return results
end

function tdvp2(
    H::MPO,
    psi_0::MPS;
    step_size::Union{ComplexF64,Float64},
    num_steps::Int,
    sweeps_per_time_step::Int,
    max_bond_dim::Int,
    svd_epsilon::Float64,
    normalize::Bool=true
)::Vector{MPS}

    dt = -im * step_size / sweeps_per_time_step / 2

    num_cells = length(H)
    @assert num_cells == length(psi_0)

    layers_left::Vector{ITensor} = [TENSOR_1]
    layers_right::Vector{ITensor} = [TENSOR_1]

    max_bond_dims::Vector{Int} = [
        min(2^i, 2^(num_cells - i), max_bond_dim)
        for i in 1:(num_cells-1)
    ]

    results = [deepcopy(psi_0)]
    sizehint!(results, num_steps)

    psi = deepcopy(psi_0)

    orthogonalize!(psi, 1)
    for site_idx in num_cells:-1:3
        push_layer!(
            layers_right,
            psi[site_idx],
            H[site_idx]
        )
    end

    @showprogress desc = "Calculating Time Evolution" for _ in 2:num_steps
        for _ in 1:sweeps_per_time_step
            for site_idx in 1:(num_cells-1)
                two_site_tensor = evolve(
                    psi[site_idx] * psi[site_idx+1],
                    layers_left[end] * H[site_idx] * H[site_idx+1] * layers_right[end],
                    dt,
                    normalize
                )
                inds_left = uniqueinds(psi[site_idx], psi[site_idx+1])
                left, S, right = svd(
                    two_site_tensor,
                    inds_left;
                    maxdim=max_bond_dims[site_idx],
                    cutoff=svd_epsilon
                )
                psi[site_idx] = left
                psi[site_idx+1] = S * right
                if site_idx != (num_cells - 1)
                    push_layer!(
                        layers_left,
                        psi[site_idx],
                        H[site_idx]
                    )
                    psi[site_idx+1] = evolve(
                        psi[site_idx+1],
                        layers_left[end] * H[site_idx+1] * layers_right[end],
                        -dt,
                        normalize
                    )
                    pop_layer!(layers_right)
                end
            end
            for site_idx in num_cells:-1:2
                two_site_tensor = evolve(
                    psi[site_idx-1] * psi[site_idx],
                    layers_left[end] * H[site_idx-1] * H[site_idx] * layers_right[end],
                    dt,
                    normalize
                )
                inds_right = uniqueinds(psi[site_idx], psi[site_idx-1])
                right, S, left = svd(
                    two_site_tensor,
                    inds_right;
                    maxdim=max_bond_dims[site_idx-1],
                    cutoff=svd_epsilon
                )
                psi[site_idx] = right
                psi[site_idx-1] = left * S
                if site_idx != 2
                    push_layer!(
                        layers_right,
                        psi[site_idx],
                        H[site_idx]
                    )
                    psi[site_idx-1] = evolve(
                        psi[site_idx-1],
                        layers_left[end] * H[site_idx-1] * layers_right[end],
                        -dt,
                        normalize
                    )
                    pop_layer!(layers_left)
                end
            end
        end
        push!(results, copy(psi))
    end
    return results
end