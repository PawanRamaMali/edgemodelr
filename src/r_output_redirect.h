#ifndef R_OUTPUT_REDIRECT_H
#define R_OUTPUT_REDIRECT_H

// Feature test macros must be defined before any system headers
#if defined(__linux__) || defined(__gnu_linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

// R-compatible output redirection for CRAN compliance
// This header redirects problematic C output functions to R-appropriate equivalents

#ifdef USING_R

// Define this before including R headers to prevent Boolean conflicts on macOS
#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif

// Include R headers first
#include <R.h>
#include <Rinternals.h>

// Custom Boolean.h replacement for macOS to avoid enum conflicts
#ifdef __APPLE__
// Manually define Rboolean to avoid enum conflicts with system headers
#ifndef Rboolean
typedef int Rboolean;
#endif
#ifndef FALSE
#define FALSE ((Rboolean) 0)
#endif
#ifndef TRUE
#define TRUE ((Rboolean) 1)
#endif
#else
// On non-macOS, use standard R Boolean header
#include <R_ext/Boolean.h>
#endif
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

// Use function approach instead of macro redefinition to avoid conflicts

// Define format attribute for GCC/Clang compatibility
// Check if R_PRINTF_FORMAT is already defined by R headers
#ifndef R_PRINTF_FORMAT
#    if !defined(__GNUC__) || defined(__APPLE__)
#        define R_PRINTF_FORMAT(...)
#    else
#        define R_PRINTF_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#endif

// Protect against R macros interfering with C++ standard library
#ifdef __cplusplus
#ifdef length
#undef length
#endif
// Fix for isNull macro conflict with Rcpp
#ifdef isNull
#undef isNull
#endif
#endif

// Global variable to control console output suppression
extern bool g_suppress_console_output;

// Create R-compatible output functions that can be suppressed
static inline void r_fputs(const char* text, FILE* stream) {
    // Always use Rprintf for CRAN compliance - ignore stream parameter
    if (!g_suppress_console_output) {
        Rprintf("%s", text);
    }
}

R_PRINTF_FORMAT(2, 3)
static inline int r_fprintf(FILE* stream, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int result = 0;
    
    // Always use Rprintf for CRAN compliance - ignore stream parameter  
    if (!g_suppress_console_output) {
        char buffer[4096];
        result = vsnprintf(buffer, sizeof(buffer), format, args);
        if (result > 0) {
            Rprintf("%s", buffer);
        }
    } else {
        // Still calculate result for compatibility but don't output
        result = vsnprintf(NULL, 0, format, args);
    }
    
    va_end(args);
    return result;
}

R_PRINTF_FORMAT(1, 2)
static inline int r_printf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    int result = 0;
    
    if (!g_suppress_console_output) {
        char buffer[4096];
        result = vsnprintf(buffer, sizeof(buffer), format, args);
        if (result > 0) {
            Rprintf("%s", buffer);
        }
    } else {
        // Still calculate result for compatibility but don't output
        result = vsnprintf(NULL, 0, format, args);
    }
    
    va_end(args);
    return result;
}

static inline int r_putchar(int c) {
    if (!g_suppress_console_output) {
        char temp[2] = {(char)c, '\0'};
        Rprintf("%s", temp);
    }
    return c;
}

static inline int r_puts(const char* str) {
    if (!g_suppress_console_output) {
        Rprintf("%s\n", str);
    }
    return strlen(str) + 1; /* Return positive value for success */
}

static inline int r_fflush(FILE* stream) {
    // For CRAN compliance, ignore fflush calls to stdout/stderr
    // R handles buffering automatically
    return 0;
}

// Redirect macros (only define if not already defined)
#ifndef fputs
#define fputs r_fputs
#endif

#ifndef fprintf
#define fprintf r_fprintf
#endif

#ifndef printf
#define printf r_printf
#endif

#ifndef putchar
#define putchar r_putchar
#endif

#ifndef puts
#define puts r_puts
#endif

#ifndef fflush
#define fflush r_fflush
#endif

// Completely nullify stderr and stdout access for CRAN compliance
// This prevents any direct access to these streams
#undef stderr
#undef stdout
#define stderr ((FILE*)0)
#define stdout ((FILE*)0)

// Also nullify the file handle numbers to prevent direct access
#ifndef STDERR_FILENO
#define STDERR_FILENO (-1)
#endif
#ifndef STDOUT_FILENO  
#define STDOUT_FILENO (-1)
#endif

#else
/* Not using R - use standard functions */
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

#endif /* USING_R */

#endif /* R_OUTPUT_REDIRECT_H */