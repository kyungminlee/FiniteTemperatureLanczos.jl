using Test
using FiniteTemperatureLanczos
using KrylovKit

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

@testset "measurement" begin
    # Spin Half Problem
    # H = -σz
    # measure σx, σy, σz
    @testset "single spin-half" begin
        σx = [0 1; 1 0]
        σy = [0 -im; im 0]
        σz = [1 0; 0 -1]
        H = -σz
        temperatures = 0:0.1:10
        obs_z = zeros(Float64, length(temperatures))
        obs_zp = zeros(Float64, length(temperatures))
        obs_E = zeros(Float64, length(temperatures))
        obs_E2 = zeros(Float64, length(temperatures))
        obs_σz = zeros(ComplexF64, length(temperatures))
        obs_σx = zeros(ComplexF64, length(temperatures))
        obs_σy = zeros(ComplexF64, length(temperatures))
        R = 5000
        for r in 1:R
            iterator = LanczosIterator(H, rand(Float64, 2) .- 0.5)
            factorization = initialize(iterator)
            expand!(iterator, factorization)
            pm = premeasure(factorization)
            pre_z = premeasure(pm)
            pre_zp = premeasure(pm, 0)
            pre_E = premeasure(pm, 1)
            pre_E2 = premeasure(pm, 2)
            pre_σx = premeasure(pm, Observable(σx))
            pre_σy = premeasure(pm, Observable(σy))
            pre_σz = premeasure(pm, Observable(σz))
            for (iT, T) in enumerate(temperatures)
                obs_z[iT]  += measure(pre_z, pm.eigen.values, T)
                obs_zp[iT] += measure(pre_zp, pm.eigen.values, T)
                obs_E[iT]  += measure(pre_E, pm.eigen.values, T)
                obs_E2[iT] += measure(pre_E2, pm.eigen.values, T)
                obs_σx[iT] += measure(pre_σx, pm.eigen.values, T)
                obs_σy[iT] += measure(pre_σy, pm.eigen.values, T)
                obs_σz[iT] += measure(pre_σz, pm.eigen.values, T)
            end
        end
        @test obs_zp == obs_z

        meas_σx = obs_σx ./ obs_z
        meas_σy = obs_σy ./ obs_z
        meas_σz = obs_σz ./ obs_z
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

        @test isapprox(obs_z[2:end] * (2 / R), true_z[2:end]; rtol=1E-1, atol=1E-1)
        @test isapprox(true_σx, meas_σx; atol=1E-1)
        @test isapprox(true_σy, meas_σy; atol=1E-1)
        @test isapprox(true_σz, meas_σz; rtol=1E-1)
        @test isapprox(true_E, meas_E; rtol=1E-1)
        @test isapprox(true_Cv, meas_Cv; rtol=1E-1, atol=1E-1, nans=true)
    end
end