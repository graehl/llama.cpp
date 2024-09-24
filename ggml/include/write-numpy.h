// write .npy file

#pragma once

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

/// \return true iff success. close fp if \param closefp.
/// \param dtype_bytes <= 8.
/// \param dtype is e.g. 'i' or 'f' per numpy format
bool write_npy(FILE* fp, bool closefp, int64_t const* shape, unsigned ndims, void const* data_row_major,
               unsigned dtype_bytes, char dtype);

/// as above but create new filename and close after writing
bool write_npy_create(char const* filename, int64_t const* shape, unsigned ndims, void const* data_row_major,
                      unsigned dtype_bytes, char dtype);

/// \param strides as per ggml_tensor.nb
/// \param shape as per ggml_tensor.ne
/// \param ndims as per ggml_n_dims(ggml_tensor *)
bool write_strided_npy(FILE* fp, bool closefp, int64_t const* shape, unsigned ndims, size_t const* strides,
                       void const* data, unsigned dtype_bytes, char dtype);

bool write_strided_npy_create(char const* filename, int64_t const* shape, unsigned ndims,
                              size_t const* strides, void const* data, unsigned dtype_bytes, char dtype);

/// as above but \param tensor is e.g. llama_get_model_tensor(model, tensor_name);
/// limitation: only works for unquantized int, float types.
/// temporary limitation: possible this only works correctly for cpu backends i.e. tensor->data is live.
bool save_npy(FILE* fp, bool closefp, struct ggml_tensor const* tensor);

bool save_npy_create(char const* filename, struct ggml_tensor const* tensor);

#ifdef __cplusplus
}  // extern
#endif
