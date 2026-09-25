using Test
using KerrzBB
import ForwardDiff
import SpectralFitting

const EDGES = [0.3, 1.0, 3.0, 8.0]
const FAST = KerrzBBOptions(n_theta = 48, n_rho = 32, n_outer = 16)
const BASE = (eta = 0.1, a = 0.8, incl = 50.0, mass = 8.0, mdot = 1.2, distance = 7.0, fcol = 1.6)

@testset "ccall wrapper" begin
    flux = kerrzbb_flux(EDGES; BASE..., options = FAST)
    @test length(flux) == 3 && all(>(0), flux)
    flux2, jac = kerrzbb_flux(EDGES; BASE..., options = FAST, free = (:a, :fcol))
    @test flux2 == flux
    @test size(jac) == (3, 2)
    h = 3e-4
    fd = (kerrzbb_flux(EDGES; BASE..., a = BASE.a + h, options = FAST) .-
          kerrzbb_flux(EDGES; BASE..., a = BASE.a - h, options = FAST)) ./ (2h)
    @test jac[:, 1] ≈ fd rtol = 1e-4
    @test_throws KerrzBB.KerrzBBError kerrzbb_flux(EDGES; BASE..., a = 0.0, options = FAST)
    cache = KerrzBBCache(FAST)
    @test kerrzbb_flux(EDGES; BASE..., cache = cache) == flux
end

@testset "SpectralFitting model and ForwardDiff" begin
    model = KerrzBBModel(options = FAST)
    out = SpectralFitting.invokemodel(EDGES, model)
    @test length(out) == 3 && all(>(0), out)

    f(a) = begin
        m = SpectralFitting.remake_with_parameters(
            model, (1.0, BASE.eta, a, BASE.incl, BASE.mass, BASE.mdot, BASE.distance, BASE.fcol),
        )
        y = zeros(typeof(a), 3, 1)
        SpectralFitting.invoke!(view(y, :, 1), EDGES, m)
        y[:, 1]
    end
    d = ForwardDiff.derivative(f, BASE.a)
    _, jac = kerrzbb_flux(EDGES; BASE..., options = FAST, free = (:a,))
    @test d ≈ jac[:, 1] rtol = 1e-12
end
