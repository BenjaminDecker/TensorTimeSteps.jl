# TensorTimeSteps.jl

### A collection of time evolution algorithms for tensor network states

Given a Hamiltonian operator $H$ in [MPO](https://tensornetwork.org/mpo/) form, and an initial state $\psi_0$ in [MPS](https://tensornetwork.org/mps/) form, calculate

$$
\psi(t_k)=e^{-iHk\Delta t}\psi_0~,~~k=0,1,\dots,N 
$$

<br/>

For now, only the 1-site and 2-site [TDVP](https://tensornetwork.org/mps/algorithms/timeevo/tdvp.html) algorithms are supported, and $H$ and $\psi_0$ must be given as [ITensor](https://docs.itensor.org/ITensorMPS/stable/examples/MPSandMPO.html) MPS/MPO types.

## Usage

Provide some $H$ and $\psi_0$
```julia
using ITensors, ITensorMPS, Random

N = 10
J = 1
g = 0.5
sites = siteinds("Qubit", N)

# Random initial product state
psi_0 = MPS(
    sites,
    map(x -> x ? "1" : "0", bitrand(length(site_inds)))
) 

# tfim Hamiltonian
os = OpSum()
for j=1:N-1
    os .+= -J,"Z",j,"Z",j+1
    os .+= -g*J,"X", j
end
os .+= -g*J, "X", N
H = MPO(os, sites)
```
<br/>

Run the TDVP and collect intermediate states at different time steps
```julia
using TensorTimeSteps


T = 1
num_steps = 100
sweeps_per_time_step = 100
max_bond_dim = 32
svd_epsilon = 10e-10

results1 = tdvp1(
    H,
    psi_0;
    step_size=T/num_steps,
    num_steps=num_steps,
    sweeps_per_time_step=sweeps_per_time_step,
    max_bond_dim=max_bond_dim
)

results2 = tdvp2(
    H,
    psi_0;
    step_size=T/num_steps,
    num_steps=num_steps,
    sweeps_per_time_step=sweeps_per_time_step,
    max_bond_dim=max_bond_dim,
    svd_epsilon=svd_epsilon
)
```
