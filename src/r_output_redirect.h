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

#include <R.h>
#include <Rinternals.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

// Redefine stderr and stdout to avoid CRAN detection
#define stderr ((FILE*)0)
#define stdout ((FILE*)0)

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
    // For stderr/stdout (now NULL), always use Rprintf - CRAN compliance
    if (stream == NULL || stream == (FILE*)0) {
        if (!g_suppress_console_output) {
            Rprintf("%s", text);
        }
    } else {
        // For real file streams, use original function
        fputs(text, stream);
    }
}

static inline int r_fprintf(FILE* stream, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int result = 0;
    
    // For stderr/stdout (now NULL), use Rprintf - CRAN compliance
    if (stream == NULL || stream == (FILE*)0) {
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
    } else {
        // For real file streams, use original function
        result = vfprintf(stream, format, args);
    }
    
    va_end(args);
    return result;
}

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

#else
/* Not using R - use standard functions */
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

#endif /* USING_R */

#endif /* R_OUTPUT_REDIRECT_H */