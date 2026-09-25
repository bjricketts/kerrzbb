// XSPEC local-model wrapper for kerrzbb.
//
// The shared library is loaded at run time with dlopen, so the XSPEC build
// needs no extra link flags. Point KERRZBB_LIBRARY at libkerrzbb (built with
// `zig build lib -Doptimize=ReleaseSafe`) before starting XSPEC:
//
//     export KERRZBB_LIBRARY=/path/to/kerrzbb/zig-out/lib/libkerrzbb.dylib
//     cd xspec && initpackage kerrzbb lmodel_kerrzbb.dat . && hmake
//     xspec> lmod kerrzbb /path/to/kerrzbb/xspec
//
// Parameters (as kerrbb): eta, a, i [deg], Mbh [M_sun], Mdd [1e18 g/s],
// Dbh [kpc], hd (fcol), rflag, lflag. XSPEC supplies the additive norm.

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" {
#include "../include/kerrzbb.h"
}

namespace {

typedef int (*evaluate_fn)(const kzbb_Params *, uint32_t, const double *, size_t,
                           double *, double *, const kzbb_Options *);
typedef const char *(*status_fn)(int);

struct Library {
    evaluate_fn evaluate = nullptr;
    status_fn status_string = nullptr;
    bool tried = false;
};

Library &library() {
    static Library lib;
    if (!lib.tried) {
        lib.tried = true;
        const char *path = std::getenv("KERRZBB_LIBRARY");
        if (!path) {
            std::fprintf(stderr, "kerrzbb: set KERRZBB_LIBRARY to the path of libkerrzbb\n");
            return lib;
        }
        void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            std::fprintf(stderr, "kerrzbb: dlopen failed: %s\n", dlerror());
            return lib;
        }
        lib.evaluate = reinterpret_cast<evaluate_fn>(dlsym(handle, "kzbb_evaluate"));
        lib.status_string = reinterpret_cast<status_fn>(dlsym(handle, "kzbb_status_string"));
    }
    return lib;
}

} // namespace

extern "C" void kerrzbbxs(const double *energy, int nFlux, const double *parameter,
                          int spectrum, double *flux, double *fluxError,
                          const char *init) {
    (void)spectrum;
    (void)fluxError;
    (void)init;
    for (int i = 0; i < nFlux; ++i) flux[i] = 0.0;

    Library &lib = library();
    if (!lib.evaluate) return;

    kzbb_Params p;
    p.eta = parameter[0];
    p.a = parameter[1];
    p.incl = parameter[2];
    p.mass = parameter[3];
    p.mdot = parameter[4];
    p.distance = parameter[5];
    p.fcol = parameter[6];
    p.norm = 1.0;
    p.r_in = 0.0;
    p.use_r_in = 0;
    p.returning_radiation = std::lround(parameter[7]) > 0 ? 1 : 0;
    p.limb_darkening = std::lround(parameter[8]) > 0 ? 1 : 0;

    const int status = lib.evaluate(&p, 0u, energy, static_cast<size_t>(nFlux), flux, nullptr, nullptr);
    if (status != KZBB_SUCCESS) {
        std::fprintf(stderr, "kerrzbb: %s (a = %g, i = %g)\n",
                     lib.status_string ? lib.status_string(status) : "error", p.a, p.incl);
        for (int i = 0; i < nFlux; ++i) flux[i] = 0.0;
    }
}
