dagger(T) = swapprime(dag(T), 0, 1)

function sum_of_products(ops::Vector{Vector{MPO}}; maxdim, nsweeps=50, tol=1e-20)::MPO
    initial_guess = let
        initial_maxdim = maxdim ÷ 4
        apply_all(op_list) = apply(op_list...; maxdim=initial_maxdim)
        sum_all(sum_list) = +(sum_list...; maxdim=initial_maxdim)
        trunc(op) = truncate(op; maxdim=initial_maxdim)
        sum_all(apply_all.(map(ops -> trunc.(ops), ops)))
    end


    W = initial_guess
    N_sites = length(W)
    N_summands = length(ops)

    layers_left = [[ITensor(1)] for _ in 1:N_summands]
    layers_right = [[ITensor(1)] for _ in 1:N_summands]

    push_layer(summand_idx::Int, side) =
        let
            local layers::Vector{ITensor}
            local site_idx::Int
            if (side == :left)
                layers = layers_left[summand_idx]
                site_idx = length(layers)
                @assert orthocenter(W) > site_idx
            elseif (side == :right)
                layers = layers_right[summand_idx]
                site_idx = N_sites - length(layers_right[summand_idx]) + 1
                @assert orthocenter(W) < site_idx
            else
                error("Allowed sides are :left and :right")
            end

            layer = contract(swapprime(dag(W[site_idx]; tags="Site"), 0, 1; tags="Site"), layers[end])
            for T in ops[summand_idx]
                layer = replaceprime(contract(prime(layer; tags="Site"), T[site_idx]), 2 => 1)
            end

            site_inds = inds(layer; tags="Site")
            @assert length(site_inds) == 2
            layer = contract(layer, dag(ITensors.delta(site_inds)))

            @assert length(inds(layer)) == (1 + length(ops[summand_idx]))

            push!(layers, layer)
        end

    ops = [[truncate(A; maxdim) for A in As] for As in ops]

    orthogonalize!(W, 1)

    for _ in reverse(3:N_sites)
        for summand_idx in 1:N_summands
            push_layer(summand_idx, :right)
        end
    end

    create_new_Wi(site_idx; width=2) =
        let
            new_Wi = zero(contract(W[site_idx:(site_idx+width-1)]))
            for summand_idx in 1:N_summands
                left_side = layers_left[summand_idx][end]
                right_side = layers_right[summand_idx][end]

                new_Wi_term = right_side

                for idx in reverse(site_idx:(site_idx+width-1))
                    for T in ops[summand_idx]
                        site_tags = "Site,n=$idx"
                        new_Wi_term = replaceprime(
                            contract(prime(new_Wi_term; tags=site_tags), T[idx]),
                            2 => 1;
                            tags=site_tags,
                        )
                    end
                end
                new_Wi_term = contract(new_Wi_term, left_side)
                new_Wi += new_Wi_term
            end
            new_Wi
        end


    p = Progress(nsweeps; showspeed=true)
    prev_norm2 = norm(W[1])^2
    for sweep in 1:nsweeps
        for site_idx in 1:(N_sites-1) #forwards
            new_Wi = create_new_Wi(site_idx; width=2)
            left_inds = uniqueinds(W[site_idx], W[site_idx+1])
            left, S, right = svd(
                new_Wi,
                left_inds;
                maxdim=maxdim,
            )
            W[site_idx] = left
            W[site_idx+1] = contract(S, right)
            ITensorMPS.set_ortho_lims!(W, (site_idx+1):(site_idx+1))
            for summand_idx in 1:N_summands
                if site_idx < (N_sites-1)
                    pop!(layers_right[summand_idx])
                    push_layer(summand_idx, :left)
                end
            end
        end
        for site_idx in reverse(1:(N_sites-1)) #backwards
            new_Wi = create_new_Wi(site_idx; width=2)
            left_inds = uniqueinds(W[site_idx], W[site_idx+1])
            left, S, right = svd(
                new_Wi,
                left_inds;
                maxdim=maxdim,
            )
            W[site_idx+1] = right
            W[site_idx] = contract(S, left)
            ITensorMPS.set_ortho_lims!(W, (site_idx:site_idx))
            for summand_idx in 1:N_summands
                if site_idx > 1
                    pop!(layers_left[summand_idx])
                    push_layer(summand_idx, :right)
                end
            end
        end

        norm2 = norm(W[1])^2
        rel_change = abs(norm2 - prev_norm2) / norm2
        update!(p, sweep; showvalues=[
            ("linkdims", linkdims(W)),
            ("relative change", rel_change),
        ])
        if rel_change < tol
            break
        end
        prev_norm2 = norm2
    end
    finish!(p)

    W
end