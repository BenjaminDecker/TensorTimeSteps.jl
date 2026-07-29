using TensorTimeSteps, Test, ITensors, ITensorMPS, Random

function create_tfim(N; J=1, g=0.5)
    sites = siteinds("Qubit", N)

    # tfim Hamiltonian
    os = OpSum()
    for j=1:(N-1)
        os .+= -J, "Z", j, "Z", j+1
        os .+= -g*J, "X", j
    end
    os .+= -g*J, "X", N
    MPO(os, sites)
end

function exact_evolution(H, psi_0, T)
    noprime(
        contract(
            exp(-im*T*contract(H)),
            contract(psi_0)
        )
    )
end

@testset let
    N = 6
    T = 1
    num_steps = 10
    sweeps_per_time_step = 10
    maxdim = 32
    cutoff = 1e-10

    H = create_tfim(N)
    psi_0 = random_mps(dag(firstsiteinds(H)); linkdims=maxdim)
    exact=exact_evolution(H, psi_0, T)

    results1 = tdvp1(
        H,
        psi_0;
        step_size=T/num_steps,
        num_steps=num_steps,
        sweeps_per_time_step=sweeps_per_time_step,
        max_bond_dim=maxdim
    )

    @test isapprox(contract(contract(results1[end]), dag(exact))[1], 1)
    @test isapprox(contract(contract(results1[1]), dag(contract(psi_0)))[1], 1)
    @test length(results1) == (num_steps + 1)

    results2 = tdvp2(
        H,
        psi_0;
        step_size=T/num_steps,
        num_steps=num_steps,
        sweeps_per_time_step=sweeps_per_time_step,
        max_bond_dim=maxdim,
        svd_epsilon=cutoff,
        switch_when_maxdim_reached=false
    )

    @test isapprox(contract(contract(results2[end]), dag(exact))[1], 1)
    @test isapprox(contract(contract(results2[1]), dag(contract(psi_0)))[1], 1)
    @test length(results2) == (num_steps + 1)
end