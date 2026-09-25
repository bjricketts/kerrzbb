/*
 * kerrzbb: multi-temperature blackbody spectrum of a thin accretion disc
 * around a Kerr black hole (Li et al. 2005, KERRBB), ray traced on the fly
 * with kerrz, with forward-mode derivatives.
 *
 * All functions are thread-safe and hold no global state.
 */
#ifndef KERRZBB_H
#define KERRZBB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model parameters (XSPEC kerrbb conventions). */
typedef struct {
    double eta;      /* torque parameter, >= 0 */
    double a;        /* spin, 1e-3 <= |a| <= 0.9999 */
    double incl;     /* inclination in degrees, 0 < i <= 89 */
    double mass;     /* black hole mass in M_sun */
    double mdot;     /* effective accretion rate in 1e18 g/s */
    double distance; /* distance in kpc */
    double fcol;     /* spectral hardening factor */
    double norm;     /* normalisation */
    double r_in;     /* inner radius in r_g, used only if use_r_in != 0 */
    int use_r_in;            /* 0: r_in = marginally stable orbit */
    int limb_darkening;      /* kerrbb lflag */
    int returning_radiation; /* kerrbb rflag: self-irradiation of the disc */
} kzbb_Params;

/* Numerical options; obtain defaults with kzbb_default_options(). */
typedef struct {
    size_t n_theta;           /* image-plane angles */
    size_t n_rho;             /* image-plane radii between r_in and r_break contours */
    size_t n_outer;           /* radii in the weak-field region */
    size_t n_energy;          /* Gauss-Legendre nodes per energy bin (<= 32) */
    double r_break;           /* start of the weak-field region, r_g */
    double r_out;             /* outer disc radius, r_g */
    double observer_distance; /* launch radius of the rays, r_g */
    size_t n_radii;           /* returning radiation: radial nodes */
    size_t n_psi;             /* returning radiation: polar nodes per azimuth */
    size_t n_chi;             /* returning radiation: azimuths */
    double r_max;             /* returning radiation: outermost node, r_g */
    size_t n_threads;         /* threads per evaluation (0: one per CPU) */
    int energy_grid;          /* use an interpolated ln E grid for many narrow bins */
    double grid_step;         /* its step in ln E */
} kzbb_Options;

/* A model with a cache of the ray tracing: repeated calls that change only
 * mass, mdot, distance, fcol, eta or norm skip the ray tracing. A model is not
 * thread-safe; use one per thread. */
typedef struct kzbb_Model kzbb_Model;

/* Bits of the free-parameter mask, also the Jacobian column order. */
enum {
    KZBB_ETA = 1u << 0,
    KZBB_A = 1u << 1,
    KZBB_INCL = 1u << 2,
    KZBB_MASS = 1u << 3,
    KZBB_MDOT = 1u << 4,
    KZBB_DISTANCE = 1u << 5,
    KZBB_FCOL = 1u << 6,
    KZBB_NORM = 1u << 7,
    KZBB_R_IN = 1u << 8
};

/* Status codes returned by kzbb_evaluate. */
enum {
    KZBB_SUCCESS = 0,
    KZBB_OUT_OF_MEMORY = 1,
    KZBB_SPIN_OUT_OF_RANGE = 2,
    KZBB_INCLINATION_OUT_OF_RANGE = 3,
    KZBB_INNER_RADIUS_BELOW_ISCO = 4,
    KZBB_INVALID_PARAMETER = 5,
    KZBB_FREE_INNER_RADIUS_WITHOUT_VALUE = 6,
    /* 7 is unused */
    KZBB_CONTOUR_FAILED = 8,
    KZBB_INVALID_ARGUMENT = 9,
    KZBB_SINGULAR_SYSTEM = 10
};

kzbb_Options kzbb_default_options(void);

/* Number of set bits in a free-parameter mask. */
int kzbb_free_count(uint32_t free_mask);

/*
 * Photon flux per energy bin, in photons cm^-2 s^-1.
 *
 * edges:    n_bins + 1 bin edges in keV (ascending)
 * flux:     output, n_bins values
 * jacobian: output, n_bins * kzbb_free_count(free_mask) values, row-major
 *           (bin, then free parameter in mask-bit order); may be NULL when
 *           free_mask == 0. The inclination derivative is per degree.
 * options:  may be NULL for defaults
 *
 * Returns KZBB_SUCCESS or one of the status codes above.
 */
int kzbb_evaluate(const kzbb_Params *params, uint32_t free_mask,
                  const double *edges, size_t n_bins, double *flux,
                  double *jacobian, const kzbb_Options *options);

/* Create a model (NULL options: defaults). Returns NULL on invalid options
 * or allocation failure. */
kzbb_Model *kzbb_model_create(const kzbb_Options *options);
void kzbb_model_destroy(kzbb_Model *model);

/* As kzbb_evaluate, with the model's options and cache. */
int kzbb_model_evaluate(kzbb_Model *model, const kzbb_Params *params,
                        uint32_t free_mask, const double *edges, size_t n_bins,
                        double *flux, double *jacobian);

const char *kzbb_status_string(int code);
const char *kzbb_version(void);

#ifdef __cplusplus
}
#endif

#endif /* KERRZBB_H */
