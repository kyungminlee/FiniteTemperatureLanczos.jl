using Test
using FiniteTemperatureLanczos

@testset "observable" begin
    let obs = StaticObservable([1 2; 3 4])
        @test eltype(obs) == Int
        @test size(obs) == (2,2)
        @test size(obs, 1) == 2
    end
    let obs = StaticObservable([1.0 2.0; 3.0 4.0])
        @test eltype(obs) == Float64
        @test size(obs) == (2,2)
        @test size(obs, 1) == 2
    end
    @test_throws DimensionMismatch StaticObservable([1 2; 3 4; 5 6])
end

@testset "susceptibility" begin
    let susc = StaticSusceptibility([1 2; 3 4], [5 6; 7 8])
        @test eltype(susc) == Int
        @test size(susc) == (2,2)
        @test size(susc, 1) == 2
    end
    let susc = StaticSusceptibility([1 2; 3 4], [5.0 6.0; 7.0 8.0])
        @test eltype(susc) == Float64
        @test size(susc) == (2,2)
        @test size(susc, 1) == 2
    end
    @test_throws DimensionMismatch StaticSusceptibility([1 2 3; 4 5 6], [7 8 9; 10 11 12])    
    @test_throws DimensionMismatch StaticSusceptibility([1 2; 4 5], [7 8 9; 10 11 12; 13 14 15])
end