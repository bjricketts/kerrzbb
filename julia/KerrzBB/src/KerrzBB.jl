"""
    KerrzBB

Julia bindings for kerrzbb (a `ccall` wrapper over `include/kerrzbb.h`) and a
SpectralFitting.jl model whose ForwardDiff derivatives come from kerrzbb's
own Jacobian rather than from differentiating through the library.

The shared library is found from `ENV["KERRZBB_LIBRARY"]`, falling back to
`zig-out/lib` in the repository. Build it with
`zig build lib -Doptimize=ReleaseSafe`.
"""
module KerrzBB

using Libdl
import ForwardDiff
import SpectralFitting
using SpectralFitting: AbstractSpectralModel, Additive, FitParam

export kerrzbb_flux, KerrzBBOptions, KerrzBBCache, KerrzBBModel, PARAMETERS

"Parameter names in Jacobian-column order (bit k of the free mask)."
const PARAMETERS = (:eta, :a, :incl, :mass, :mdot, :distance, :fcol, :norm, :r_in)

struct CParams
    eta::Cdouble
    a::Cdouble
    incl::Cdouble
    mass::Cdouble
    mdot::Cdouble
    distance::Cdouble
    fcol::Cdouble
    norm::Cdouble
    r_in::Cdouble
    use_r_in::Cint
    limb_darkening::Cint
    returning_radiation::Cint
end

"Numerical options; `KerrzBBOptions(; n_theta = 64, n_threads = 0, ...)` overrides the defaults."
struct KerrzBBOptions
    n_theta::Csize_t
    n_rho::Csize_t
    n_outer::Csize_t
    n_energy::Csize_t
    r_break::Cdouble
    r_out::Cdouble
    observer_distance::Cdouble
    n_radii::Csize_t
    n_psi::Csize_t
    n_chi::Csize_t
    r_max::Cdouble
    n_threads::Csize_t
    energy_grid::Cint
    grid_step::Cdouble
end

const _handle = Ref{Ptr{Cvoid}}(C_NULL)

function library_path()
    haskey(ENV, "KERRZBB_LIBRARY") && return ENV["KERRZBB_LIBRARY"]
    joinpath(@__DIR__, "..", "..", "..", "zig-out", "lib", "libkerrzbb." * Libdl.dlext)
end

function _sym(name::Symbol)
    if _handle[] == C_NULL
        _handle[] = Libdl.dlopen(library_path())
    end
    Libdl.dlsym(_handle[], name)
end

default_options() = ccall(_sym(:kzbb_default_options), KerrzBBOptions, ())

function KerrzBBOptions(; kwargs...)
    d = default_options()
    vals = map(fieldnames(KerrzBBOptions), fieldtypes(KerrzBBOptions)) do f, T
        convert(T, get(kwargs, f, getfield(d, f)))
    end
    KerrzBBOptions(vals...)
end

version() = unsafe_string(ccall(_sym(:kzbb_version), Cstring, ()))

struct KerrzBBError <: Exception
    code::Cint
    message::String
end
Base.showerror(io::IO, e::KerrzBBError) = print(io, "kerrzbb error ", e.code, ": ", e.message)

"""
    KerrzBBCache(options)

Handle to a kerrzbb model with a cache of the ray tracing. Pass it as
`cache` to `kerrzbb_flux` so repeated calls that change only mass, mdot,
distance, fcol, eta or norm skip the ray tracing. Not thread-safe.
"""
mutable struct KerrzBBCache
    ptr::Ptr{Cvoid}
    options::KerrzBBOptions
    function KerrzBBCache(options::KerrzBBOptions = default_options())
        ptr = ccall(_sym(:kzbb_model_create), Ptr{Cvoid}, (Ref{KerrzBBOptions},), options)
        ptr == C_NULL && throw(ArgumentError("invalid kerrzbb options"))
        c = new(ptr, options)
        finalizer(c) do x
            x.ptr != C_NULL && ccall(_sym(:kzbb_model_destroy), Cvoid, (Ptr{Cvoid},), x.ptr)
            x.ptr = C_NULL
        end
        c
    end
end

"""
    kerrzbb_flux(edges; a, incl, mass, mdot, distance, eta = 0, fcol = 1.7,
                 norm = 1, r_in = nothing, limb_darkening = false,
                 returning_radiation = false, free = (), options = default_options(),
                 cache = nothing)

Photon flux per bin (photons cm⁻² s⁻¹) for bin `edges` in keV. With a
non-empty `free` (a collection of names from `PARAMETERS`) returns
`(flux, jacobian)` where `jacobian[b, k]` is the derivative of bin `b` with
respect to the `k`-th free parameter in `PARAMETERS` order. The inclination and
its derivative are in degrees. With `cache::KerrzBBCache` the cache's
options are used instead of `options`.
"""
function kerrzbb_flux(
    edges::AbstractVector{<:Real};
    a, incl, mass, mdot, distance,
    eta = 0.0, fcol = 1.7, norm = 1.0, r_in = nothing,
    limb_darkening::Bool = false, returning_radiation::Bool = false,
    free = (), options::KerrzBBOptions = default_options(),
    cache::Union{Nothing,KerrzBBCache} = nothing,
)
    e = collect(Float64, edges)
    n_bins = length(e) - 1
    n_bins >= 1 || throw(ArgumentError("need at least two bin edges"))
    mask = UInt32(0)
    for name in free
        k = findfirst(==(Symbol(name)), PARAMETERS)
        isnothing(k) && throw(ArgumentError("unknown parameter $(name)"))
        mask |= UInt32(1) << (k - 1)
    end
    n_free = count_ones(mask)
    p = CParams(eta, a, incl, mass, mdot, distance, fcol, norm,
                isnothing(r_in) ? 0.0 : r_in, !isnothing(r_in), limb_darkening,
                returning_radiation)
    flux = Vector{Float64}(undef, n_bins)
    # C writes row-major (bin, parameter): allocate transposed and permute.
    jac_t = Matrix{Float64}(undef, n_free, n_bins)
    jac_ptr = n_free > 0 ? pointer(jac_t) : Ptr{Cdouble}(C_NULL)
    status = if isnothing(cache)
        ccall(
            _sym(:kzbb_evaluate), Cint,
            (Ref{CParams}, UInt32, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}, Ref{KerrzBBOptions}),
            p, mask, e, n_bins, flux, jac_ptr, options,
        )
    else
        GC.@preserve cache ccall(
            _sym(:kzbb_model_evaluate), Cint,
            (Ptr{Cvoid}, Ref{CParams}, UInt32, Ptr{Cdouble}, Csize_t, Ptr{Cdouble}, Ptr{Cdouble}),
            cache.ptr, p, mask, e, n_bins, flux, jac_ptr,
        )
    end
    if status != 0
        msg = unsafe_string(ccall(_sym(:kzbb_status_string), Cstring, (Cint,), status))
        throw(KerrzBBError(status, msg))
    end
    n_free == 0 ? flux : (flux, permutedims(jac_t))
end

# ---------------------------------------------------------------------------
# SpectralFitting.jl model

"Non-fitted configuration carried by `KerrzBBModel`, including the ray-tracing cache."
struct KerrzBBConfig
    limb_darkening::Bool
    returning_radiation::Bool
    cache::KerrzBBCache
end

"""
    KerrzBBModel(; K, eta, a, incl, mass, mdot, distance, fcol,
                 limb_darkening = false, returning_radiation = false,
                 options = KerrzBBOptions())

Additive SpectralFitting.jl model. `K` is SpectralFitting's normalisation
(kerrbb's `norm`). Units follow XSPEC kerrbb: `incl` in degrees, `mdot` in
1e18 g/s, `distance` in kpc, `mass` in M_sun. Spins with |a| < 1e-3 are not
supported, so fits should start away from a = 0.

Derivatives: when SpectralFitting differentiates with ForwardDiff, the model
calls kerrzbb once with the needed Jacobian columns and assembles the dual
output from it. Nested duals (Hessians) are not supported. The model keeps a
ray-tracing cache, so it is not safe to invoke one instance from several
threads at once.
"""
struct KerrzBBModel{T,C} <: AbstractSpectralModel{T,Additive}
    config::C
    "Normalisation."
    K::T
    "Torque parameter."
    eta::T
    "Spin."
    a::T
    "Inclination (degrees)."
    incl::T
    "Black hole mass (M_sun)."
    mass::T
    "Effective accretion rate (1e18 g/s)."
    mdot::T
    "Distance (kpc)."
    distance::T
    "Spectral hardening factor."
    fcol::T
end

function KerrzBBModel(;
    K = FitParam(1.0; frozen = true),
    eta = FitParam(0.0; frozen = true, lower_limit = 0.0, upper_limit = 10.0),
    a = FitParam(0.5; lower_limit = -0.9999, upper_limit = 0.9999),
    incl = FitParam(30.0; frozen = true, lower_limit = 0.1, upper_limit = 89.0),
    mass = FitParam(10.0; frozen = true, lower_limit = 0.1, upper_limit = 1e3),
    mdot = FitParam(1.0; lower_limit = 1e-6, upper_limit = 1e6),
    distance = FitParam(10.0; frozen = true, lower_limit = 1e-3, upper_limit = 1e5),
    fcol = FitParam(1.7; frozen = true, lower_limit = 1.0, upper_limit = 3.0),
    limb_darkening::Bool = false,
    returning_radiation::Bool = false,
    options::KerrzBBOptions = KerrzBBOptions(),
)
    config = KerrzBBConfig(limb_darkening, returning_radiation, KerrzBBCache(options))
    KerrzBBModel(config, K, eta, a, incl, mass, mdot, distance, fcol)
end

const _MODEL_PARAMS = (:eta, :a, :incl, :mass, :mdot, :distance, :fcol)

_params(m::KerrzBBModel) = (m.eta, m.a, m.incl, m.mass, m.mdot, m.distance, m.fcol)

function _flux(domain, config::KerrzBBConfig, vals; free = ())
    kerrzbb_flux(domain; eta = vals[1], a = vals[2], incl = vals[3], mass = vals[4],
                 mdot = vals[5], distance = vals[6], fcol = vals[7], norm = 1.0,
                 limb_darkening = config.limb_darkening,
                 returning_radiation = config.returning_radiation,
                 cache = config.cache, free = free)
end

function SpectralFitting.invoke!(output, domain, model::KerrzBBModel)
    _invoke!(output, domain, model.config, _params(model))
end

function _invoke!(output, domain, config, ps::NTuple{7,<:Real})
    output .= _flux(domain, config, Float64.(ps))
end

function _invoke!(output, domain, config, ps::NTuple{7,D}) where {D<:ForwardDiff.Dual}
    vals = map(ForwardDiff.value, ps)
    active = [k for k = 1:7 if !iszero(ForwardDiff.partials(ps[k]))]
    if isempty(active)
        output .= _flux(domain, config, vals)
        return output
    end
    flux, jac = _flux(domain, config, vals; free = _MODEL_PARAMS[active])
    for b in eachindex(flux)
        partials = sum(jac[b, j] * ForwardDiff.partials(ps[k]) for (j, k) in enumerate(active))
        output[b] = D(flux[b], partials)
    end
    output
end

end # module
