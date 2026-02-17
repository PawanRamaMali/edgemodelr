#include <Rcpp.h>
#include <vector>
#include <string>

// This file MUST be compiled with GGML_CXXFLAGS (not standard R CXXFLAGS)
// to correctly detect SIMD features enabled for the GGML engine.

// [[Rcpp::export]]
Rcpp::List edge_simd_info_internal() {
  std::vector<std::string> features;

#ifdef __SSE2__
  features.push_back("SSE2");
#endif
#ifdef __SSE3__
  features.push_back("SSE3");
#endif
#ifdef __SSSE3__
  features.push_back("SSSE3");
#endif
#ifdef __SSE4_1__
  features.push_back("SSE4.1");
#endif
#ifdef __SSE4_2__
  features.push_back("SSE4.2");
#endif
#ifdef __AVX__
  features.push_back("AVX");
#endif
#ifdef __AVX2__
  features.push_back("AVX2");
#endif
#ifdef __FMA__
  features.push_back("FMA");
#endif
#ifdef __F16C__
  features.push_back("F16C");
#endif
#ifdef __AVX512F__
  features.push_back("AVX512F");
#endif
#ifdef __AVX512BW__
  features.push_back("AVX512BW");
#endif
#ifdef __AVX512DQ__
  features.push_back("AVX512DQ");
#endif
#ifdef __AVX512VL__
  features.push_back("AVX512VL");
#endif
#ifdef __ARM_NEON
  features.push_back("NEON");
#endif
#ifdef __ARM_FEATURE_SVE
  features.push_back("SVE");
#endif

  std::string arch = "unknown";
#if defined(__x86_64__) || defined(_M_X64)
  arch = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  arch = "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
  arch = "arm";
#elif defined(__i386__) || defined(_M_IX86)
  arch = "x86";
#elif defined(__powerpc64__)
  arch = "ppc64";
#elif defined(__s390x__)
  arch = "s390x";
#elif defined(__riscv)
  arch = "riscv";
#endif

  bool is_generic = false;
#ifdef GGML_CPU_GENERIC
  is_generic = true;
#endif

  std::vector<std::string> ggml_features;
#ifdef GGML_SSE42
  ggml_features.push_back("GGML_SSE42");
#endif
#ifdef GGML_AVX
  ggml_features.push_back("GGML_AVX");
#endif
#ifdef GGML_AVX2
  ggml_features.push_back("GGML_AVX2");
#endif
#ifdef GGML_FMA
  ggml_features.push_back("GGML_FMA");
#endif
#ifdef GGML_F16C
  ggml_features.push_back("GGML_F16C");
#endif
#ifdef GGML_AVX512
  ggml_features.push_back("GGML_AVX512");
#endif
#ifdef GGML_CPU_GENERIC
  ggml_features.push_back("GGML_CPU_GENERIC");
#endif

  return Rcpp::List::create(
    Rcpp::Named("architecture") = arch,
    Rcpp::Named("compiler_features") = features,
    Rcpp::Named("ggml_features") = ggml_features,
    Rcpp::Named("is_generic") = is_generic
  );
}
