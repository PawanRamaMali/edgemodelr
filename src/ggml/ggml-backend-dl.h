#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
// Note: <winevt.h> removed — not available in MinGW and not used here
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Conditional filesystem support (macOS < 10.15 lacks full C++17 filesystem)
#if defined(__APPLE__) && defined(__MACH__)
#    if __cplusplus >= 201703L && defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500
#        include <filesystem>
#        define GGML_DL_HAS_FILESYSTEM 1
#    else
#        define GGML_DL_HAS_FILESYSTEM 0
#    endif
#else
#    if __cplusplus >= 201703L
#        include <filesystem>
#        define GGML_DL_HAS_FILESYSTEM 1
#    else
#        define GGML_DL_HAS_FILESYSTEM 0
#    endif
#endif

#if GGML_DL_HAS_FILESYSTEM
namespace fs = std::filesystem;
#else
#    include <string>
// Minimal path shim for older macOS
namespace fs {
    struct path {
        std::string p;
        path() = default;
        path(const std::string & s) : p(s) {}
        path(const char * s) : p(s) {}
        std::string string() const { return p; }
        std::wstring wstring() const { return std::wstring(p.begin(), p.end()); }
        bool empty() const { return p.empty(); }
    };
}
#endif

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(const fs::path & path);
void * dl_get_sym(dl_handle * handle, const char * name);
const char * dl_error();
