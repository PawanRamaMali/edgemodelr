#ifndef R_OUTPUT_REDIRECT_H
#define R_OUTPUT_REDIRECT_H

// R-compatible output redirection for CRAN compliance
// This header redirects problematic C output functions to R-appropriate equivalents

#ifdef USING_R

#include <R.h>
#include <Rinternals.h>

// Protect against R macros interfering with C++ standard library
#ifdef __cplusplus
#ifdef length
#undef length
#endif
#endif

// Create R-compatible output functions
static inline void r_fputs(const char* text, FILE* stream) {
    if (stream == stderr || stream == stdout) {
        Rprintf("%s", text);
    } else {
        /* Use original fputs for file streams */
        fputs(text, stream);
    }
}

    int result = 0;
    
    if (stream == stderr || stream == stdout) {
        /* Use Rprintf for console output */
        char buffer[4096];
        result = vsnprintf(buffer, sizeof(buffer), format, args);
        if (result > 0) {
            Rprintf("%s", buffer);
        }
    } else {
        /* Use original fprintf for file streams */
        result = vfprintf(stream, format, args);
    }
    
    va_end(args);
    return result;
}

static inline int r_printf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    char buffer[4096];
    int result = vsnprintf(buffer, sizeof(buffer), format, args);
    if (result > 0) {
        Rprintf("%s", buffer);
    }
    
    va_end(args);
    return result;
}

static inline int r_putchar(int c) {
    char temp[2] = {(char)c, '\0'};
    Rprintf("%s", temp);
    return c;
}

static inline int r_puts(const char* str) {
    Rprintf("%s\n", str);
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

#endif /* USING_R */

#endif /* R_OUTPUT_REDIRECT_H */