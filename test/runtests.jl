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

@testset verbose=true "HUBBARD" begin
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