/* Minimal use of the kerrzbb C API.
 *
 *   zig build lib
 *   cc examples/example.c -Izig-out/include -Lzig-out/lib -lkerrzbb -o example
 */
#include <stdio.h>
#include <kerrzbb.h>

int main(void) {
    kzbb_Params p = {
        .eta = 0.0, .a = 0.9, .incl = 60.0, .mass = 10.0, .mdot = 1.0,
        .distance = 10.0, .fcol = 1.7, .norm = 1.0,
        .use_r_in = 0, .limb_darkening = 0, .returning_radiation = 0,
    };
    double edges[] = {0.5, 1.0, 2.0, 5.0, 10.0};
    enum { NB = 4 };
    double flux[NB];
    uint32_t mask = KZBB_A | KZBB_INCL;
    double jac[NB * 2];

    int status = kzbb_evaluate(&p, mask, edges, NB, flux, jac, NULL);
    if (status != KZBB_SUCCESS) {
        fprintf(stderr, "kerrzbb: %s\n", kzbb_status_string(status));
        return 1;
    }
    printf("kerrzbb %s\n", kzbb_version());
    for (int b = 0; b < NB; b++)
        printf("[%5.2f, %5.2f] keV: %.6e ph/cm^2/s  dN/da %+.4e  dN/di %+.4e\n",
               edges[b], edges[b + 1], flux[b], jac[2 * b], jac[2 * b + 1]);
    return 0;
}
