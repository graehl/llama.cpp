#include "write-numpy.h"
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <vector>
#include <cassert>

namespace {
unsigned constexpr NPY_MAGIC_VER_LEN = 8;
char npy_header_0_magic_cstr[NPY_MAGIC_VER_LEN] = "\x93NUMPY\x01";
std::string const npy_header_0_magic(npy_header_0_magic_cstr, 8);
std::string const npy_header_1_then_type = "{'descr':'<";  // e.g. f4 for little endian float32
std::string const npy_header_2_then_shape = "','fortran_order':False,'shape':(";  // #r, or #r, #c ...
std::string const npy_header_3_then_padding_nl = ")}";
std::string const npy_shape_sep = ",";

// append little-endian bytes of x
template <class Buf, class T>
void operator+=(Buf& buf, const T x) {
  auto bytes = reinterpret_cast<char const*>(&x);
  for (unsigned i = 0; i < sizeof(T); ++i) buf.push_back(bytes[i]);
}

template <class Buf>
void operator+=(Buf& buf, char const* x) {
  while (x) buf.push_back(*x++);
}

template <class Buf>
void operator+=(Buf& buf, std::string const& x) {
  buf.insert(buf.end(), x.begin(), x.end());
}

template <class Buf>
void pad(Buf& buf, unsigned n, char pad = ' ') {
  buf.insert(buf.end(), n, pad);
}

/// \pre buf is empty
template <class Buf>
void npy_append_buf(Buf& buf, char dtype, unsigned dtype_bytes, int64_t const* shape, unsigned ndims, unsigned align = 64) {
  assert(dtype_bytes < 100);
  buf.reserve(align);
  buf += npy_header_0_magic;
  unsigned constexpr isz = NPY_MAGIC_VER_LEN; // would need to use dynamic buf.size() if buf weren't empty at start
  assert(buf.size() == 8);
  buf.resize(isz + 2);
  buf += npy_header_1_then_type;
  buf.push_back(dtype);
  if (dtype_bytes > 10) {
    buf.push_back('0' + dtype_bytes / 10);
    dtype_bytes %= 10;
  }
  buf.push_back('0' + dtype_bytes);
  buf += npy_header_2_then_shape;
  if (ndims == 1) {
    buf += std::to_string(shape[0]);
    buf.push_back(',');
  } else {
    for (unsigned i = 0; i < ndims; ++i) {
      if (i) buf += npy_shape_sep;
      buf += std::to_string(shape[i]);
    }
  }
  buf += npy_header_3_then_padding_nl;
  unsigned sz = buf.size();
  unsigned headsz = sz - isz - 1;
  unsigned npad = align - (sz + 1) % align;
  if (npad == align) npad = 0;
  pad(buf, npad);
  buf.push_back('\n');
  headsz += npad;
  buf[isz] = (char)(headsz & 0xff);
  buf[isz + 1] = (char)((headsz & 0xff00) >> 8);
}

inline std::size_t prod(int64_t const* shape, unsigned ndims) {
  if (!ndims) return 0;
  std::size_t nels = shape[0];
  for (unsigned dim = 1; dim < ndims; ++dim) nels *= shape[dim];
  return nels;
}

inline bool write_strided(FILE* fp, void const* data, size_t dtype_bytes, int64_t const* shape,
                          unsigned ndims, size_t const* strides) {
  if (!data) return false;
  if (ndims == 0) return false;
  char const* bytes = (char const*)data;
  if (ndims == 1) {
    size_t nels = shape[0];
    size_t stride = strides[0];
    if (stride == dtype_bytes) return fwrite(bytes, dtype_bytes, nels, fp) == nels;
    for (size_t i = 0; i != nels; bytes += stride, ++i)
      if (fwrite(bytes, dtype_bytes, 1, fp) != 1) return false;
    return true;
  } else {
    unsigned d = ndims - 1;
    for (size_t i = 0, n = shape[d], stride = strides[d]; i != n; ++i, bytes += stride) {
      if (!write_strided(fp, bytes, dtype_bytes, shape, ndims - 1, strides)) return false;
    }

    return true;
  }
}

}  // namespace


extern "C" {

bool write_strided_npy(FILE* fp, bool closefp, int64_t const* shape, unsigned ndims, size_t const* strides,
                       void const* data, unsigned dtype_bytes, char dtype) {
  if (!fp) return false;
  bool written = false;
  if (dtype_bytes <= 8 && ndims && fp) {
    std::vector<char> buf;
    npy_append_buf(buf, dtype, dtype_bytes, shape, ndims);
    size_t nbuf = buf.size();
    if ((written = (fwrite(buf.data(), nbuf, 1, fp) == 1))) {
      written = write_strided(fp, data, dtype_bytes, shape, ndims, strides);
    }
  }
  if (closefp) fclose(fp);
  return written;
}

bool write_strided_npy_create(char const* filename, int64_t const* shape, unsigned ndims,
                              size_t const* strides, void const* data, unsigned dtype_bytes, char dtype) {
  return write_strided_npy(fopen(filename, "wb"), true, shape, ndims, strides, data, dtype_bytes, dtype);
}


bool write_npy_create(char const* filename, int64_t const* shape, unsigned ndims, void const* data,
                      unsigned dtype_bytes, char dtype) {
  return write_npy(fopen(filename, "wb"), true, shape, ndims, data, dtype_bytes, dtype);
}

bool write_npy(FILE* fp, bool closefp, int64_t const* shape, unsigned ndims, void const* data_row_major,
               unsigned dtype_bytes, char dtype) {
  if (!fp) return false;
  bool written = false;
  if (dtype_bytes <= 8 && ndims && fp) {
    std::vector<char> buf;
    npy_append_buf(buf, dtype, dtype_bytes, shape, ndims);
    size_t nbuf = buf.size();
    size_t nels = prod(shape, ndims);
    written = fwrite(buf.data(), 1, nbuf, fp) == nbuf && fwrite(data_row_major, dtype_bytes, nels, fp) == nels;
  }
  if (closefp) fclose(fp);
  return written;
}

bool save_npy(FILE* fp, bool closefp, struct ggml_tensor const* tensor) {
  char dtype;
  unsigned dtype_bytes = 4;
  switch (tensor->type) {
    case GGML_TYPE_F64:
      dtype = 'f';
      dtype_bytes = 8;
      break;
    case GGML_TYPE_F32: dtype = 'f'; break;
    case GGML_TYPE_F16:
      dtype = 'f';
      dtype_bytes = 2;
      break;
    case GGML_TYPE_I32: dtype = 'i'; break;
    case GGML_TYPE_I16:
      dtype = 'i';
      dtype_bytes = 2;
      break;
    case GGML_TYPE_I8:
      dtype = 'i';
      dtype_bytes = 1;
      break;
    default: return false;
  }
  void *data = ggml_get_data(tensor);
  if (data)
    return write_strided_npy(fp, closefp, tensor->ne, ggml_n_dims(tensor), tensor->nb, data,
                             dtype_bytes, dtype);
  else {
    size_t nbytes = ggml_nbytes(tensor);
    std::unique_ptr<char[]> bytes(new char[nbytes]);
    char *data = bytes.get();
    ggml_backend_tensor_get(tensor, data, 0, nbytes);
    return write_npy(fp, closefp, tensor->ne, ggml_n_dims(tensor), data, dtype_bytes, dtype);
  }
}

bool save_npy_create(char const* filename, struct ggml_tensor const* tensor) {
  return save_npy(fopen(filename, "wb"), true, tensor);
}

}
