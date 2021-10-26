using Test
using FiniteTemperatureLanczos
using KrylovKit
using QuantumHamiltonian
using Random
using LinearAlgebra

@testset "observable" begin
    let obs = Observable([1 2; 3 4])
        @test eltype(obs) == Int
        @test size(obs) == (2,2)
        @test size(obs, 1) == 2
    end
    let obs = Observable([1.0 2.0; 3.0 4.0])
        @test eltype(obs) == Float64
        @test size(obs) == (2,2)
        @test size(obs, 1) == 2
    end
    # Hermitian
    let obs = Observable(Hermitian([1.0 0.1im; -0.1im 2.0]))
        @test eltype(obs) == ComplexF64
        @test size(obs) == (2,2)
        @test size(obs, 1) == 2
    end
    @test_throws DimensionMismatch Observable([1 2; 3 4; 5 6])
end

@testset "susceptibility" begin
    let susc = Susceptibility([1 2; 3 4], [5 6; 7 8])
        @test eltype(susc) == Int
        @test size(susc) == (2,2)
        @test size(susc, 1) == 2
    end
    let susc = Susceptibility([1 2; 3 4], [5.0 6.0; 7.0 8.0])
        @test eltype(susc) == Float64
        @test size(susc) == (2,2)
        @test size(susc, 1) == 2
    end
    @test_throws DimensionMismatch Susceptibility([1 2 3; 4 5 6], [7 8 9; 10 11 12])    
    @test_throws DimensionMismatch Susceptibility([1 2; 4 5], [7 8 9; 10 11 12; 13 14 15])
end

# Spin Half Problem
# H = -σz
# measure σx, σy, σz
@testset "measurement-single spin-half" begin
    σx = [0 1; 1 0]
    σy = [0 -im; im 0]
    σz = [1 0; 0 -1]
    Hσx = Hermitian(σx)
    Hσy = Hermitian(σy)
    Hσz = Hermitian(σz)
    
    H = -σz
    temperatures = [0.1, 0.5, 1.0]
    obs_z = zeros(Float64, length(temperatures))
    obs_zp = zeros(Float64, length(temperatures))
    obs_E = zeros(Float64, length(temperatures))
    obs_E2 = zeros(Float64, length(temperatures))
    obs_σz = zeros(ComplexF64, length(temperatures))
    obs_σx = zeros(ComplexF64, length(temperatures))
    obs_σy = zeros(ComplexF64, length(temperatures))
    obs_Hσz = zeros(ComplexF64, length(temperatures))
    obs_Hσx = zeros(ComplexF64, length(temperatures))
    obs_Hσy = zeros(ComplexF64, length(temperatures))

    R = 10000
    rng = Random.MersenneTwister(1)
    for r in 1:R
        iterator = LanczosIterator(H, randn(rng, ComplexF64, 2))
        factorization = initialize(iterator)
        expand!(iterator, factorization) # single expansion since D = 2
        pm = premeasure(factorization)
        pre_z = premeasure(pm)
        pre_zp = premeasure(pm, 0)
        pre_E = premeasure(pm, 1)
        pre_E2 = premeasure(pm, 2)
        pre_σx = premeasure(pm, Observable(σx))
        pre_σy = premeasure(pm, Observable(σy))
        pre_σz = premeasure(pm, Observable(σz))
        pre_Hσx = premeasure(pm, Observable(Hσx))
        pre_Hσy = premeasure(pm, Observable(Hσy))
        pre_Hσz = premeasure(pm, Observable(Hσz))
        for (iT, T) in enumerate(temperatures)
            # @show T
            obs_z[iT]  += measure(pre_z, pm.eigen.values, T)
            obs_zp[iT] += measure(pre_zp, pm.eigen.values, T)
            obs_E[iT]  += measure(pre_E, pm.eigen.values, T)
            obs_E2[iT] += measure(pre_E2, pm.eigen.values, T)
            obs_σx[iT] += measure(pre_σx, pm.eigen.values, T)
            obs_σy[iT] += measure(pre_σy, pm.eigen.values, T)
            obs_σz[iT] += measure(pre_σz, pm.eigen.values, T)
            obs_Hσx[iT] += measure(pre_Hσx, pm.eigen.values, T)
            obs_Hσy[iT] += measure(pre_Hσy, pm.eigen.values, T)
            obs_Hσz[iT] += measure(pre_Hσz, pm.eigen.values, T)
        end
    end
    @test obs_zp == obs_z

    meas_σx = obs_σx ./ obs_z
    meas_σy = obs_σy ./ obs_z
    meas_σz = obs_σz ./ obs_z
    meas_Hσx = obs_Hσx ./ obs_z
    meas_Hσy = obs_Hσy ./ obs_z
    meas_Hσz = obs_Hσz ./ obs_z
    meas_E  = obs_E ./ obs_z
    meas_Cv = ((obs_E2./obs_z) - (obs_E./obs_z).^2) ./ temperatures.^2

    # Z  = 2 cosh(1/T)
    # σz = tanh(1/T)
    # E  = -tanh(1/T)
    # Cv = sech(1/T)² / T²

    true_z = cosh.(1 ./ temperatures) .* 2
    true_σx = zeros(Float64, length(temperatures))
    true_σy = zeros(Float64, length(temperatures))
    true_σz = tanh.(1 ./ temperatures)
    true_E = -tanh.(1 ./ temperatures)
    true_Cv = sech.(1 ./ temperatures).^2 ./ temperatures.^2

    @test isapprox(obs_z * (2 / R), true_z; rtol=1E-2, atol=1E-2)
    @test isapprox(true_σx, meas_σx; atol=1E-2)
    @test isapprox(true_σy, meas_σy; atol=1E-2)
    @test isapprox(true_σz, meas_σz; rtol=1E-2)
    @test isapprox(true_σx, meas_Hσx; atol=1E-2)
    @test isapprox(true_σy, meas_Hσy; atol=1E-2)
    @test isapprox(true_σz, meas_Hσz; rtol=1E-2)
    @test isapprox(true_E, meas_E; rtol=1E-2)
    @test isapprox(true_Cv, meas_Cv; rtol=1E-2, atol=1E-2, nans=true)
end

@testset "measure-spin chain" begin
    nsites = 6
    hs, pauli = QuantumHamiltonian.Toolkit.spin_half_system(nsites)
    hsr = represent(hs)
    hamiltonian = - simplify(sum(pauli(i, μ) * pauli(mod(i, nsites)+1, μ) for i in 1:nsites for μ in [:x, :y, :z])) - simplify(sum(pauli(i, :z) for i in 1:nsites))
    hamiltonian_rep = represent(hsr, hamiltonian)
    hamiltonian_mat = Matrix(hamiltonian_rep)

    spin_corr = simplify(sum(pauli(1, μ) * pauli(2, μ) for μ in [:x, :y, :z]))
    spin_corr_rep = represent(hsr, spin_corr)
    spin_corr_mat = Matrix(spin_corr_rep)

    temperatures = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    obs_z1 = zeros(Float64, length(temperatures))
    obs_z2 = zeros(Float64, length(temperatures))
    obs_E1 = zeros(Float64, length(temperatures))
    obs_E2 = zeros(Float64, length(temperatures))
    obs_sc1 = zeros(ComplexF64, length(temperatures))
    obs_sc2 = zeros(ComplexF64, length(temperatures))
    rng = MersenneTwister(0)
    v0 = randn(rng, ComplexF64, dimension(hsr))
    normalize!(v0)
    pm1 = let
        iterator = LanczosIterator(hamiltonian_rep, v0)
        factorization = initialize(iterator)
        for i in 2:6
            expand!(iterator, factorization) # single expansion since D = 2
        end
        premeasure(factorization)
    end
    pm2 = let
        iterator = LanczosIterator(hamiltonian_mat, v0)
        factorization = initialize(iterator)
        for i in 2:6
            expand!(iterator, factorization) # single expansion since D = 2
        end
        premeasure(factorization)
    end
    @test all(isapprox.(pm1.basis, pm2.basis))
    @test isapprox(pm1.eigen.values, pm2.eigen.values)

    pre_z1 = premeasure(pm1)
    pre_z2 = premeasure(pm2)
    pre_E1 = premeasure(pm1, 1)
    pre_E2 = premeasure(pm2, 1)
    pre_sc1 = premeasure(pm1, Observable(spin_corr_rep))
    pre_sc2 = premeasure(pm2, Observable(spin_corr_mat))
    for (iT, T) in enumerate(temperatures)
        # @show T
        obs_z1[iT]  += measure(pre_z1, pm1.eigen.values, T)
        obs_z2[iT]  += measure(pre_z2, pm2.eigen.values, T)
        obs_E1[iT]  += measure(pre_E1, pm1.eigen.values, T)
        obs_E2[iT]  += measure(pre_E2, pm2.eigen.values, T)
        obs_sc1[iT] += measure(pre_sc1, pm1.eigen.values, T)
        obs_sc2[iT] += measure(pre_sc2, pm2.eigen.values, T)
    end
    @test isapprox(obs_z1, obs_z2)
    @test isapprox(obs_E1, obs_E2)
    @test isapprox(obs_sc1, obs_sc2)
end
