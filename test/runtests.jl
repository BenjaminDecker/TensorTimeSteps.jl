using TensorTimeSteps, Test, ITensors, ITensorMPS, Random, Combinatorics


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
        )
        test_against_exact(exact_results, results2)
    end
end

@testset verbose=true "TFIM" begin
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

function test_sum_of_products(s, local_H_dim)
    n = length(s)
    ITensors.set_warn_order(n*2 + 1)
    rowcol_inds = [prime.(s); dag.(s)]

    random_tensors = [[random_itensor(ComplexF64, rowcol_inds) for _ in 1:3] for _ in 1:10]
    ops_mpo = [[MPO(T, s) for T in Ts] for Ts in random_tensors]

    multiply(A, B) = replaceprime(A' * B, 2 => 1)
    exact_result = sum([foldl(multiply, Ts) for Ts in random_tensors])
    ITensors.reset_warn_order()

    exact_linkdim = Int((local_H_dim^2)^(floor(n/2)))

    initial_guess = random_mpo(s)
    should_improve = TensorTimeSteps.sum_of_products(ops_mpo; maxdim=exact_linkdim ÷ 2, initial_guess)
    @test norm(exact_result - contract(should_improve)) < norm(exact_result - contract(initial_guess))

    should_pass = TensorTimeSteps.sum_of_products(ops_mpo; maxdim=exact_linkdim)
    @test isapprox(contract(should_pass), exact_result)

    should_fail = TensorTimeSteps.sum_of_products(ops_mpo; maxdim=exact_linkdim-1)
    @test !isapprox(contract(should_fail), exact_result)

    # maxdim = exact_linkdim ÷ 2
    # ops_mpo_truncated = [[truncate(mpo; maxdim) for mpo in mpo_list] for mpo_list in ops_mpo]
    # ITensors.set_warn_order(n*2 + 1)
    # tensors_truncated = [contract.(mpo_list) for mpo_list in ops_mpo_truncated]
    # exact_result_truncated = sum([foldl(multiply, Ts) for Ts in tensors_truncated])
    # ITensors.reset_warn_order()

    # should_pass = TensorTimeSteps.sum_of_products(ops_mpo_truncated; maxdim)
    # @test isapprox(contract(should_pass), exact_result_truncated)

end

@testset "sum of products Qubit" begin
    s = siteinds("Qubit", 5)
    test_sum_of_products(s, 2)
end

@testset "sum of products Electron" begin
    s = siteinds("Electron", 3; conserve_qns=true)
    test_sum_of_products(s, 4)
end