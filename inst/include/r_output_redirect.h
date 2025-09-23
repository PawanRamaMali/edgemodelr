#ifndef R_OUTPUT_REDIRECT_H
#define R_OUTPUT_REDIRECT_H

#include <iostream>
#include <streambuf>
#include <string>

// Simple output suppression for CRAN compliance
class NullBuffer : public std::streambuf {
public:
    int overflow(int c) { return c; }
};

class SuppressOutput {
private:
    std::streambuf* orig_cout;
    std::streambuf* orig_cerr;
    NullBuffer null_buffer;

public:
    SuppressOutput() {
        orig_cout = std::cout.rdbuf();
        orig_cerr = std::cerr.rdbuf();
        std::cout.rdbuf(&null_buffer);
        std::cerr.rdbuf(&null_buffer);
    }

    ~SuppressOutput() {
        std::cout.rdbuf(orig_cout);
        std::cerr.rdbuf(orig_cerr);
    }
};

// Global variable to control output suppression
extern bool g_suppress_console_output;

#endif // R_OUTPUT_REDIRECT_H