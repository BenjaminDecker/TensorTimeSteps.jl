const TENSOR_1::ITensor = ITensor(1)

function evolve(A::ITensor, H_eff::ITensor, T::Complex, do_normalize::Bool)::ITensor
    new_A, info = exponentiate(H_eff, T, A, eager=true)
    @assert info.converged == 1
    if (do_normalize)
        normalize!(new_A)
    end
    return new_A
end

function push_layer!(layers::Vector{ITensor}, site::ITensor, H_site::ITensor)
    new_layer = contract((layers[end], site, H_site, dag(prime(site))), sequence="automatic")
    push!(layers, new_layer)
end

pop_layer!(layers::Vector{ITensor}) = pop!(layers)

function fix_bond_dims(psi::MPS, H::MPO, max_bond_dims::Vector{Int})::MPS
    max_bond_dim = maximum(max_bond_dims)
    psi = truncate(psi; maxdim=max_bond_dim, cutoff=0.0)
    site_inds = siteinds(psi)
    if hasqns(site_inds[1])
        check_bond_dims(psi) = ((linkdims(psi) - max_bond_dims .< 0) |> sum) == 0

        last_linkdims = linkdims(psi)
        try_counter = 0
        while !check_bond_dims(psi)
            println(repeat('-', 80))
            try_counter += 1
            krylovdim = 2^try_counter
            println("Artificially increasing bond dimension of initial state. Attempt $(try_counter) with krylovdim $(krylovdim)")
            psi = expand(psi, H; alg="global_krylov", krylovdim=krylovdim)
            @show max_bond_dims
            @show linkdims(psi)
            if last_linkdims == linkdims(psi)
                println("Linkdims did not change. Trying with 'cutoff=0.0'.")
                psi = expand(psi, H; alg="global_krylov", krylovdim=krylovdim, cutoff=0.0)
            end
            if last_linkdims == linkdims(psi)
                println("Linkdims did not change. Initial state will have a lower bond dimension than requested. Cancelling expansion.")
                break
            end
            last_linkdims = linkdims(psi)
        end

        println(repeat('-', 80))
        @show max_bond_dims
        @show linkdims(psi)
        println("If linkdims are higher than the maximum, they will be truncated during the first sweep to fit max_bond_dims.")
        println(repeat('-', 80))
        psi
    else
        link_inds = [Index(max_bond_dims[n] - linkdims(psi)[n], "Link,l=$n") for n in eachindex(max_bond_dims)]
        num_cells = length(psi)

        bond_increaser = MPS(num_cells)
        for n in 1:num_cells
            if n == 1
                bond_increaser[n] = ITensor(site_inds[n], link_inds[n])
            elseif n == num_cells
                bond_increaser[n] = ITensor(link_inds[n-1], site_inds[n])
            else
                bond_increaser[n] = ITensor(link_inds[n-1], site_inds[n], link_inds[n])
            end
        end

        # Fix the bond dimensions
        # https://itensor.discourse.group/t/how-do-i-set-an-mps-bond-dimension-that-is-higher-than-needed/1637
        +(
            psi,
            0 * bond_increaser;
            alg="directsum"
        )
    end
end

function truncate_to_exact_bond_dims!(psi::MPS, link_dims::Vector{Int})::MPS
    orthogonalize!(psi, 1)
    for i in eachindex(linkdims(psi))
        inds_left = uniqueinds(psi[i], psi[i+1])
        left, S, right = svd(
            contract(psi[i], psi[i+1]),
            inds_left;
            maxdim=link_dims[i],
            cutoff=0.0
        )
        psi[i] = left
        psi[i+1] = contract(S, right)
    end
    psi
end

function tdvp1(
    H::MPO,
    psi_0::MPS;
    step_size::Union{ComplexF64,Float64},
    num_steps::Int,
    sweeps_per_time_step::Int,
    max_bond_dim::Int,
    normalize::Bool=true,
    expand_bond_dims=true,
)::Vector{MPS}

    num_cells = length(H)
    @assert num_cells == length(psi_0)

    local_site_dim = dim(siteind(first, H, 1))
    max_bond_dim = min(max_bond_dim, local_site_dim^(num_cells ÷ 2))
    max_bond_dims::Vector{Int} = [
        min(local_site_dim^i, local_site_dim^(num_cells - i), max_bond_dim)
        for i in 1:(num_cells-1)
    ]

    psi = psi_0
    if expand_bond_dims
        psi = fix_bond_dims(psi_0, H, max_bond_dims)
    end

    dt = -im * step_size / sweeps_per_time_step / 2

    results = [deepcopy(psi)]
    sizehint!(results, num_steps + 1)

    layers_left::Vector{ITensor} = [TENSOR_1]
    layers_right::Vector{ITensor} = [TENSOR_1]

    orthogonalize!(psi, 1)
    for site_idx in num_cells:-1:2
        push_layer!(
            layers_right,
            psi[site_idx],
            H[site_idx]
        )
    end

    p = Progress(num_steps; desc="Running 1-site TDVP", showspeed=true)
    for _ in 1:num_steps
        next!(p; showvalues=[("Linkdims", linkdims(psi))])
        for _ in 1:sweeps_per_time_step
            for site_idx in 1:num_cells
                psi[site_idx] = evolve(
                    psi[site_idx],
                    contract((layers_left[end], H[site_idx], layers_right[end]), sequence="automatic"),
                    dt,
                    normalize
                )
                if site_idx != num_cells
                    inds_left = uniqueinds(psi[site_idx], psi[site_idx+1])
                    site_orthogonal, bond = (
                        if linkdims(psi)[site_idx] > max_bond_dims[site_idx]
                            left, S, right = svd(
                                psi[site_idx],
                                inds_left;
                                maxdim=max_bond_dims[site_idx],
                                mindim=max_bond_dims[site_idx],
                                cutoff=0.0
                            )
                            left, contract(S, right)
                        else
                            qr(psi[site_idx], inds_left)
                        end
                    )
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
                    contract((layers_left[end], H[site_idx], layers_right[end]), sequence="automatic"),
                    dt,
                    normalize
                )
                if site_idx != 1
                    inds_right = uniqueinds(psi[site_idx], psi[site_idx-1])
                    site_orthogonal, bond = (
                        if linkdims(psi)[site_idx-1] > max_bond_dims[site_idx-1]
                            right, S, left = svd(
                                two_site_tensor,
                                inds_right;
                                maxdim=max_bond_dims[site_idx-1],
                                mindim=max_bond_dims[site_idx-1],
                                cutoff=0.0
                            )
                            right, contract(S, left)
                        else
                            qr(psi[site_idx], inds_right)
                        end
                    )
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
    finish!(p)
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
    normalize::Bool=true,
    switch_when_maxdim_reached::Bool=true
)::Vector{MPS}

    num_cells = length(H)
    @assert num_cells == length(psi_0)

    local_site_dim = dim(siteind(first, H, 1))

    max_bond_dims::Vector{Int} = [
        min(local_site_dim^i, local_site_dim^(num_cells - i), max_bond_dim)
        for i in 1:(num_cells-1)
    ]

    results = [deepcopy(psi_0)]
    sizehint!(results, num_steps + 1)

    psi = deepcopy(psi_0)

    layers_left::Vector{ITensor} = [TENSOR_1]
    layers_right::Vector{ITensor} = [TENSOR_1]

    dt = -im * step_size / sweeps_per_time_step / 2

    orthogonalize!(psi, 1)
    for site_idx in num_cells:-1:3
        push_layer!(
            layers_right,
            psi[site_idx],
            H[site_idx]
        )
    end

    p = Progress(num_steps; desc="Running 2-site TDVP", showspeed=true)
    for step_idx in 1:num_steps
        next!(p; showvalues=[("Linkdims", linkdims(psi))])
        if switch_when_maxdim_reached && maximum(linkdims(psi)) >= max_bond_dim
            println()
            println("Maximum bond dimension reached, switching to 1-site TDVP...")
            finish!(p)
            return [
                results[1:(end-1)];
                tdvp1(
                    H,
                    psi;
                    step_size,
                    num_steps=num_steps - step_idx + 1,
                    sweeps_per_time_step,
                    max_bond_dim,
                    normalize,
                    expand_bond_dims=false
                )
            ]
        end

        for _ in 1:sweeps_per_time_step
            for site_idx in 1:(num_cells-1)
                two_site_tensor = evolve(
                    psi[site_idx] * psi[site_idx+1],
                    contract((layers_left[end], H[site_idx], H[site_idx+1], layers_right[end]), sequence="automatic"),
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
                        contract((layers_left[end], H[site_idx+1], layers_right[end]), sequence="automatic"),
                        -dt,
                        normalize
                    )
                    pop_layer!(layers_right)
                end
            end
            for site_idx in num_cells:-1:2
                two_site_tensor = evolve(
                    psi[site_idx-1] * psi[site_idx],
                    contract((layers_left[end], H[site_idx-1], H[site_idx], layers_right[end]), sequence="automatic"),
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
                        contract((layers_left[end], H[site_idx-1], layers_right[end]), sequence="automatic"),
                        -dt,
                        normalize
                    )
                    pop_layer!(layers_left)
                end
            end
        end
        push!(results, copy(psi))
    end
    finish!(p)
    return results
end
