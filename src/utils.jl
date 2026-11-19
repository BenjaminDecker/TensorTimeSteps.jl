dagger(T) = swapprime(dag(T), 0, 1)

function sum_of_products(ops::Vector{Vector{MPO}}; maxdim::Int, initial_guess::MPO, nsweeps::Int=20, tol::Float64=1e-10, verbose::Bool=false, width::Int=2, svd_cutoff::Float64=1e-12)::MPO
    if (width != 1 && width != 2)
        error("Only widths of 1 or 2 are supported")
    end

    W = copy(initial_guess)
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

    orthogonalize!(W, 1)

    for _ in reverse((1+width):N_sites)
        for summand_idx in 1:N_summands
            push_layer(summand_idx, :right)
        end
    end

    create_new_Wi(site_idx; width=2) =
        let
            # TODO create new_Wi without contracting something first
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
        for site_idx in 1:(N_sites-width+1) #forwards
            new_Wi = create_new_Wi(site_idx; width)
            if width==1
                W[site_idx] = new_Wi
                if site_idx < N_sites
                    orthogonalize!(W, site_idx+1)
                end
            else
                left_inds = uniqueinds(W[site_idx], W[site_idx+1])
                left, S, right = svd(
                    new_Wi,
                    left_inds;
                    maxdim=maxdim,
                    cutoff=svd_cutoff
                )
                W[site_idx] = left
                W[site_idx+1] = contract(S, right)
                ITensorMPS.set_ortho_lims!(W, (site_idx+1):(site_idx+1))
            end
            for summand_idx in 1:N_summands
                if site_idx < (N_sites-width+1)
                    pop!(layers_right[summand_idx])
                    push_layer(summand_idx, :left)
                end
            end
        end
        for site_idx in reverse(1:(N_sites-width+1)) #backwards
            new_Wi = create_new_Wi(site_idx; width)
            if width==1
                W[site_idx] = new_Wi
                if site_idx > 1
                    orthogonalize!(W, site_idx-1)
                end
            else
                left_inds = uniqueinds(W[site_idx], W[site_idx+1])
                left, S, right = svd(
                    new_Wi,
                    left_inds;
                    maxdim=maxdim,
                    cutoff=svd_cutoff
                )
                W[site_idx+1] = right
                W[site_idx] = contract(S, left)
                ITensorMPS.set_ortho_lims!(W, (site_idx:site_idx))
            end
            for summand_idx in 1:N_summands
                if site_idx > 1
                    pop!(layers_left[summand_idx])
                    push_layer(summand_idx, :right)
                end
            end
        end

        norm2 = norm(W[1])^2
        rel_change = abs(norm2 - prev_norm2) / norm2
        verbose && update!(p, sweep; showvalues=[
            ("linkdims", linkdims(W)),
            ("relative change", rel_change),
        ])
        if rel_change < tol
            break
        end
        prev_norm2 = norm2
    end
    verbose && finish!(p)

    W
end