// edgemodelr: runtime choice of the AVX2 kernels in the portable build.
// See edgemodelr-avx2.c. This header is included several times on purpose:
// with EDGE_AVX2_RENAME defined it renames the x86 kernels to *_avx2, and
// without it it removes those names again.

#ifndef EDGEMODELR_DISPATCH_H
#define EDGEMODELR_DISPATCH_H

// Only GCC on x86-64: its target pragma also defines __AVX2__ and friends,
// which arch/x86/quants.c tests. Elsewhere the generic kernels stay in use.
#if defined(EDGEMODELR_CPU_DISPATCH) && defined(__GNUC__) && !defined(__clang__) && \
    (defined(__x86_64__) || defined(_M_X64))
#define EDGEMODELR_AVX2_DISPATCH 1
#else
#define EDGEMODELR_AVX2_DISPATCH 0
#endif

// Every vec_dot kernel of arch/x86/quants.c.
#define EDGE_AVX2_VEC_DOTS(X)                                                   \
    X(ggml_vec_dot_q1_0_q8_0) X(ggml_vec_dot_q2_0_q8_0)                         \
    X(ggml_vec_dot_q4_0_q8_0) X(ggml_vec_dot_q4_1_q8_1)                         \
    X(ggml_vec_dot_q5_0_q8_0) X(ggml_vec_dot_q5_1_q8_1)                         \
    X(ggml_vec_dot_q8_0_q8_0) X(ggml_vec_dot_mxfp4_q8_0)                        \
    X(ggml_vec_dot_nvfp4_q8_0) X(ggml_vec_dot_tq1_0_q8_K)                       \
    X(ggml_vec_dot_tq2_0_q8_K) X(ggml_vec_dot_q2_K_q8_K)                        \
    X(ggml_vec_dot_q3_K_q8_K) X(ggml_vec_dot_q4_K_q8_K)                         \
    X(ggml_vec_dot_q5_K_q8_K) X(ggml_vec_dot_q6_K_q8_K)                         \
    X(ggml_vec_dot_iq2_xxs_q8_K) X(ggml_vec_dot_iq2_xs_q8_K)                    \
    X(ggml_vec_dot_iq2_s_q8_K) X(ggml_vec_dot_iq3_xxs_q8_K)                     \
    X(ggml_vec_dot_iq3_s_q8_K) X(ggml_vec_dot_iq1_s_q8_K)                       \
    X(ggml_vec_dot_iq1_m_q8_K) X(ggml_vec_dot_iq4_nl_q8_0)                      \
    X(ggml_vec_dot_iq4_xs_q8_K)

// Every quantize_row kernel of arch/x86/quants.c.
#define EDGE_AVX2_QUANTIZERS(X) \
    X(quantize_row_q8_0) X(quantize_row_q8_1) X(quantize_row_q8_K)

#endif // EDGEMODELR_DISPATCH_H

#if EDGEMODELR_AVX2_DISPATCH
#ifdef EDGE_AVX2_RENAME
#define ggml_vec_dot_q1_0_q8_0    ggml_vec_dot_q1_0_q8_0_avx2
#define ggml_vec_dot_q2_0_q8_0    ggml_vec_dot_q2_0_q8_0_avx2
#define ggml_vec_dot_q4_0_q8_0    ggml_vec_dot_q4_0_q8_0_avx2
#define ggml_vec_dot_q4_1_q8_1    ggml_vec_dot_q4_1_q8_1_avx2
#define ggml_vec_dot_q5_0_q8_0    ggml_vec_dot_q5_0_q8_0_avx2
#define ggml_vec_dot_q5_1_q8_1    ggml_vec_dot_q5_1_q8_1_avx2
#define ggml_vec_dot_q8_0_q8_0    ggml_vec_dot_q8_0_q8_0_avx2
#define ggml_vec_dot_mxfp4_q8_0   ggml_vec_dot_mxfp4_q8_0_avx2
#define ggml_vec_dot_nvfp4_q8_0   ggml_vec_dot_nvfp4_q8_0_avx2
#define ggml_vec_dot_tq1_0_q8_K   ggml_vec_dot_tq1_0_q8_K_avx2
#define ggml_vec_dot_tq2_0_q8_K   ggml_vec_dot_tq2_0_q8_K_avx2
#define ggml_vec_dot_q2_K_q8_K    ggml_vec_dot_q2_K_q8_K_avx2
#define ggml_vec_dot_q3_K_q8_K    ggml_vec_dot_q3_K_q8_K_avx2
#define ggml_vec_dot_q4_K_q8_K    ggml_vec_dot_q4_K_q8_K_avx2
#define ggml_vec_dot_q5_K_q8_K    ggml_vec_dot_q5_K_q8_K_avx2
#define ggml_vec_dot_q6_K_q8_K    ggml_vec_dot_q6_K_q8_K_avx2
#define ggml_vec_dot_iq2_xxs_q8_K ggml_vec_dot_iq2_xxs_q8_K_avx2
#define ggml_vec_dot_iq2_xs_q8_K  ggml_vec_dot_iq2_xs_q8_K_avx2
#define ggml_vec_dot_iq2_s_q8_K   ggml_vec_dot_iq2_s_q8_K_avx2
#define ggml_vec_dot_iq3_xxs_q8_K ggml_vec_dot_iq3_xxs_q8_K_avx2
#define ggml_vec_dot_iq3_s_q8_K   ggml_vec_dot_iq3_s_q8_K_avx2
#define ggml_vec_dot_iq1_s_q8_K   ggml_vec_dot_iq1_s_q8_K_avx2
#define ggml_vec_dot_iq1_m_q8_K   ggml_vec_dot_iq1_m_q8_K_avx2
#define ggml_vec_dot_iq4_nl_q8_0  ggml_vec_dot_iq4_nl_q8_0_avx2
#define ggml_vec_dot_iq4_xs_q8_K  ggml_vec_dot_iq4_xs_q8_K_avx2
#define quantize_row_q8_0         quantize_row_q8_0_avx2
#define quantize_row_q8_1         quantize_row_q8_1_avx2
#define quantize_row_q8_K         quantize_row_q8_K_avx2
#else
#undef ggml_vec_dot_q1_0_q8_0
#undef ggml_vec_dot_q2_0_q8_0
#undef ggml_vec_dot_q4_0_q8_0
#undef ggml_vec_dot_q4_1_q8_1
#undef ggml_vec_dot_q5_0_q8_0
#undef ggml_vec_dot_q5_1_q8_1
#undef ggml_vec_dot_q8_0_q8_0
#undef ggml_vec_dot_mxfp4_q8_0
#undef ggml_vec_dot_nvfp4_q8_0
#undef ggml_vec_dot_tq1_0_q8_K
#undef ggml_vec_dot_tq2_0_q8_K
#undef ggml_vec_dot_q2_K_q8_K
#undef ggml_vec_dot_q3_K_q8_K
#undef ggml_vec_dot_q4_K_q8_K
#undef ggml_vec_dot_q5_K_q8_K
#undef ggml_vec_dot_q6_K_q8_K
#undef ggml_vec_dot_iq2_xxs_q8_K
#undef ggml_vec_dot_iq2_xs_q8_K
#undef ggml_vec_dot_iq2_s_q8_K
#undef ggml_vec_dot_iq3_xxs_q8_K
#undef ggml_vec_dot_iq3_s_q8_K
#undef ggml_vec_dot_iq1_s_q8_K
#undef ggml_vec_dot_iq1_m_q8_K
#undef ggml_vec_dot_iq4_nl_q8_0
#undef ggml_vec_dot_iq4_xs_q8_K
#undef quantize_row_q8_0
#undef quantize_row_q8_1
#undef quantize_row_q8_K
#endif
#endif
