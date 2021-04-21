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
        obs_σz = zeros(ComplexF64, length(temperatures))
        obs_σx = zeros(ComplexF64, length(temperatures))
        obs_σy = zeros(ComplexF64, length(temperatures))
        R = 1000
        for r in 1:R
            iterator = LanczosIterator(H, rand(Float64, 2) .- 0.5)
            factorization = initialize(iterator)
            expand!(iterator, factorization)
            pm = premeasure(factorization)
            pre_z = premeasure(pm)
            pre_σx = premeasure(pm, Observable(σx))
            pre_σy = premeasure(pm, Observable(σy))
            pre_σz = premeasure(pm, Observable(σz))
            for (iT, T) in enumerate(temperatures)
                obs_z[iT] += measure(pre_z, pm.eigen.values, T)
                obs_σx[iT] += measure(pre_σx, pm.eigen.values, T)
                obs_σy[iT] += measure(pre_σy, pm.eigen.values, T)
                obs_σz[iT] += measure(pre_σz, pm.eigen.values, T)
            end
        end
        true_z = cosh.(1 ./ temperatures) .* 2
        true_σx = 0.0
        true_σy = 0.0
        true_σz = tanh.(1 ./ temperatures)
        @test all(<(0.1), abs.(obs_z[2:end] * (2 / R) - true_z[2:end]) ./ true_z[2:end])
        @test all(x -> (abs2(x) < 1E-2), true_σx .- obs_σx ./ obs_z)
        @test all(x -> (abs2(x) < 1E-2), true_σy .- obs_σy ./ obs_z)
        @test all(x -> (abs2(x) < 1E-2), true_σz .- obs_σz ./ obs_z)
        @show true_σz .- obs_σz ./ obs_z
        # @show true_σz
    end
end