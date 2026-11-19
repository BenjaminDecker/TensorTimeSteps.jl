using TensorTimeSteps, Test, ITensors, ITensorMPS, Random, Combinatorics, LinearAlgebra

function enable_qn_optimizations(verbose::Bool=false)
    verbose && println("Enabling qn optimizations")
    BLAS.set_num_threads(1)
    ITensors.Strided.disable_threads()

    ITensors.enable_threaded_blocksparse()

    if verbose
        @show Threads.nthreads()
        @show BLAS.get_num_threads()
        @show ITensors.using_threaded_blocksparse()
    end
end

function disable_qn_optimizations(verbose::Bool=false)
    verbose && println("Disabling qn optimizations")
    BLAS.set_num_threads(Threads.nthreads() ÷ 2)
    ITensors.Strided.enable_threads()

    ITensors.disable_threaded_blocksparse()

    if verbose
        @show Threads.nthreads()
        @show BLAS.get_num_threads()
        @show ITensors.using_threaded_blocksparse()
    end
end

function create_tfim(N; J::Float64=1.0, g::Float64=1.0)::MPO
    s = siteinds("Qubit", N)

    os = OpSum()
    for j in 1:(N-1)
        os .+= -J, "Z", j, "Z", j+1
    end
    for j in 1:N
        os .+= -g*J, "X", j
    end
    MPO(os, s)
end

function create_hubbard(N; t::Float64=1.0, U::Float64=1.0)
    s = siteinds("Electron", N; conserve_qns=true)

    os = OpSum()
    for j in 1:(N-1)
        os += -t, "Cdagup", j, "Cup", j+1
        os += -t, "Cdagup", j+1, "Cup", j
        os += -t, "Cdagdn", j, "Cdn", j+1
        os += -t, "Cdagdn", j+1, "Cdn", j
    end
    for j in 1:N
        os += U, "Nup", j, "Ndn", j
    end
    MPO(os, s)
end

function calculate_exact_results(psi_0::MPS, H::MPO, dt::Float64, num_steps::Int)::Vector{ITensor}
    U = exp(-im*dt*contract(H))
    exact_results = [contract(psi_0)]
    for _ in 1:num_steps
        push!(exact_results, noprime(contract(U, exact_results[end])))
    end
    @assert length(exact_results) == (num_steps + 1)
    exact_results
end

function test_against_exact(exact_results::Vector{ITensor}, results::Vector{MPS})
    @test length(results) == length(exact_results)
    @test all([
        isapprox(1, contract(dag(exact_results[i]), contract(results[i]))[1])
        for i in eachindex(exact_results)
    ])
end

function test_tdvp(
    H::MPO,
    psi_0::MPS,
    num_steps::Int,
    dt::Float64,
    maxdim::Int,
    sweeps_per_time_step::Int,
    cutoff::Float64
)
    exact_results = calculate_exact_results(psi_0, H, dt, num_steps)

    @testset "tdvp1" begin
        results1 = tdvp1(
            H,
            psi_0;
            step_size=dt,
            num_steps=num_steps,
            sweeps_per_time_step=sweeps_per_time_step,
            max_bond_dim=maxdim,
            verbose=false
        )
        test_against_exact(exact_results, results1)
    end

    @testset "tdvp2" begin
        results2 = tdvp2(
            H,
            psi_0;
            step_size=dt,
            num_steps=num_steps,
            sweeps_per_time_step=sweeps_per_time_step,
            max_bond_dim=maxdim,
            svd_epsilon=cutoff,
            switch_when_maxdim_reached=false,
            verbose=false
        )
        test_against_exact(exact_results, results2)
    end
end

rowcol_inds(s) = [prime.(s); dag.(s)]

function random_compressible_tensor(s, decay::Float64=1.0)
    C = ITensor(ComplexF64, 0.0, rowcol_inds(s))

    c1 = combiner(dag(s))
    c2 = prime(dag(c1))
    C = contract(c1, C, c2)

    i1 = combinedind(c1)
    i2 = combinedind(c2)
    vals = shuffle(exp.(-decay .* (1:(dim(s[1])^length(s)))))
    for (i, v) in enumerate(vals)
        C[i1=>i, i2=>i] = v
    end
    C = contract(dag(c1), C, dag(c2))

    T1 = random_itensor(ComplexF64, rowcol_inds(s))
    T2 = random_itensor(ComplexF64, rowcol_inds(s))
    T1C = swapprime(T1' * C, 2 => 1)
    T1CT2 = swapprime(T1C' * T2, 2 => 1)
    return T1CT2
end

function test_sum_of_products(s)
    n = length(s)

    random_tensors = [[random_itensor(ComplexF64, rowcol_inds(s)) for _ in 1:3] for _ in 1:10]
    random_compressible_tensors = [[random_compressible_tensor(s) for _ in 1:3] for _ in 1:10]

    multiply(A, B) = replaceprime(A' * B, 2 => 1)

    maxdim = Int((dim(s[1])^2)^(floor(n/2)))

    @testset verbose=true "$width-site" for width in 1:2
        @testset verbose=true "$label" for (label, tensors) in [("Not Compressible", random_tensors), ("Compressible", random_compressible_tensors)]
            ops_mpo = [[MPO(T, s) for T in Ts] for Ts in tensors]
            ITensors.set_warn_order(n*2 + 1)
            exact = sum([foldl(multiply, Ts) for Ts in tensors])
            ITensors.reset_warn_order()


            initial_guess = apply([
                truncate(op; maxdim) for op in ops_mpo[1]
            ]...; maxdim)
            should_pass = TensorTimeSteps.sum_of_products(ops_mpo; maxdim, initial_guess, width, verbose=false)
            @test isapprox(contract(should_pass), exact)

            truncate!(initial_guess; maxdim=maxdim-1)
            should_fail = TensorTimeSteps.sum_of_products(ops_mpo; maxdim=maxdim-1, initial_guess, width, verbose=false)
            @test !isapprox(contract(should_fail), exact)

            if (label=="Compressible")
                truncate!(initial_guess; maxdim=maxdim ÷ 2)
                should_improve = TensorTimeSteps.sum_of_products(ops_mpo; maxdim=maxdim ÷ 2, initial_guess, width, verbose=false)
                @test norm(exact - contract(should_improve)) < norm(exact - contract(initial_guess))
            end
        end
    end
end

@testset verbose=true "TFIM" begin
    disable_qn_optimizations()
    NUM_STEPS = 100
    DT = 1e-2
    N = 6
    MAXDIM = 8
    SWEEPS_PER_TIME_STEP = 1
    CUTOFF = 1e-12


    for g in (0.5, 1.0, 1.5)
        @testset verbose=true "g=$(g)" begin
            H = create_tfim(N; J=1.0, g=g)
            psi_0 = random_mps(dag(firstsiteinds(H)); linkdims=2)

            test_tdvp(H, psi_0, NUM_STEPS, DT, MAXDIM, SWEEPS_PER_TIME_STEP, CUTOFF)
        end
    end
end

@testset verbose=true "Hubbard" begin
    enable_qn_optimizations()
    NUM_STEPS = 100
    DT = 1e-2
    N = 4
    MAXDIM = 16
    SWEEPS_PER_TIME_STEP = 1
    CUTOFF = 1e-12


    for U in (0.5, 1.0, 1.5)
        @testset verbose=true "U=$(U)" begin
            H = create_hubbard(N; t=1.0, U=U)
            s = dag(firstsiteinds(H))
            psi_0 = random_mps(s, n -> isodd(n) ? "Up" : "Dn"; linkdims=2)

            test_tdvp(H, psi_0, NUM_STEPS, DT, MAXDIM, SWEEPS_PER_TIME_STEP, CUTOFF)
        end
    end
end

@testset verbose=true "sum of products Qubit" begin
    disable_qn_optimizations()
    s = siteinds("Qubit", 5)
    test_sum_of_products(s)
end

@testset verbose=true "sum of products Electron" begin
    enable_qn_optimizations()
    s = siteinds("Electron", 3; conserve_qns=true)
    test_sum_of_products(s)
end