// Call a GRiM XLA FFI implementation (``static ffi::Error XImpl(cudaStream_t, ...)``) directly from
// C, with no XLA: each parameter is built from flat arrays in the handler's Bind order. Buffers
// (Arg and Ret alike) come from ``b``, raw device pointers wrapped as XLA_FFI_Buffer; attributes
// come from ``a`` as doubles, cast to the parameter's type (int64_t / float / bool).
#pragma once

#include <cstdio>
#include <tuple>

#include "xla/ffi/api/ffi.h"

namespace gb {
namespace ffi = xla::ffi;

struct Cursor {
    XLA_FFI_Buffer* b;
    const double* a;
    int bi = 0, ai = 0;
};

template <class T>
struct Make {
    static T get(Cursor& c) { return static_cast<T>(c.a[c.ai++]); }
};
template <ffi::DataType D, size_t R>
struct Make<ffi::Buffer<D, R>> {
    static ffi::Buffer<D, R> get(Cursor& c) { return ffi::Buffer<D, R>(&c.b[c.bi++]); }
};
template <class T>
struct Make<ffi::Result<T>> {
    static ffi::Result<T> get(Cursor& c) { return ffi::Result<T>(Make<T>::get(c)); }
};

template <class... A>
int invoke(ffi::Error (*f)(cudaStream_t, A...), cudaStream_t s, XLA_FFI_Buffer* b, const double* a)
{
    Cursor c{b, a};
    std::tuple<A...> args{Make<A>::get(c)...};   // braced init: evaluated left to right
    const ffi::Error e = std::apply([&](A&... x) { return f(s, x...); }, args);
    if (e.failure()) std::fprintf(stderr, "grim bridge: %s\n", e.message().c_str());
    return e.failure() ? 1 : 0;
}

}  // namespace gb
