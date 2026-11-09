// edgemodelr: the x86 quantized dot-product kernels built a second time for
// AVX2, so the portable default build can use them on CPUs that have AVX2.
// ggml_cpu_init() (ggml-cpu.c) switches to these after a CPU check; without
// AVX2 the generic kernels stay in use. No compiler flag enables AVX2: the
// target pragma below applies to this file only, so the package still builds
// and runs on any x86-64 CPU.
//
// This file is compiled without GGML_CPU_GENERIC, so arch/x86/quants.c takes
// its AVX2 paths. Every kernel it defines is renamed with an _avx2 suffix,
// and the *_generic fallbacks it calls for some types are provided at the
// end of this file by calling the generic kernels, which the generic build
// names without the suffix.

#include "edgemodelr-dispatch.h"

#if EDGEMODELR_AVX2_DISPATCH

#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c,bmi2")

#define EDGE_AVX2_RENAME
#include "edgemodelr-dispatch.h"

#include "arch/x86/quants.c"

#undef EDGE_AVX2_RENAME
#include "edgemodelr-dispatch.h"

#pragma GCC pop_options

// quants.h was read with the _avx2 names above, so the generic kernels are
// declared here.
#define EDGE_GENERIC_VEC_DOT(name)                                              \
    void name(int n, float * GGML_RESTRICT s, size_t bs,                        \
              const void * GGML_RESTRICT vx, size_t bx,                         \
              const void * GGML_RESTRICT vy, size_t by, int nrc);               \
    void name##_generic(int n, float * GGML_RESTRICT s, size_t bs,              \
                        const void * GGML_RESTRICT vx, size_t bx,               \
                        const void * GGML_RESTRICT vy, size_t by, int nrc) {    \
        name(n, s, bs, vx, bx, vy, by, nrc);                                    \
    }

EDGE_GENERIC_VEC_DOT(ggml_vec_dot_q1_0_q8_0)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_q2_0_q8_0)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_q4_1_q8_1)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_q5_0_q8_0)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_q5_1_q8_1)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_tq1_0_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_tq2_0_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_q2_K_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_q3_K_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_q4_K_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_q5_K_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_q6_K_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_iq2_xxs_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_iq2_xs_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_iq2_s_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_iq3_xxs_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_iq3_s_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_iq1_s_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_iq1_m_q8_K)
EDGE_GENERIC_VEC_DOT(ggml_vec_dot_iq4_xs_q8_K)

#else

// Nothing to build on this platform or compiler. ISO C needs a declaration.
typedef int edgemodelr_avx2_unused;

#endif
