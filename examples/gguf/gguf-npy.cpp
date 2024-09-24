#include "ggml-cpu.h"
#include "write-numpy.h"
#include "gguf-ctx.hpp"

#include <cstdio>
#include <string>
#include <sstream>
#include <vector>
#include <memory>

template <typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}


static bool gguf_ex_write_npy(const std::string & fname, const std::string & tensorname) {
    ggml_backend_ptr backend_cpu{ggml_backend_cpu_init()};
    if (!backend_cpu) {
      printf("%s: ERROR ggml_backend_cpu_init()\n", __func__);
      return false;
    }

    ggml_context_ptr ggml_ctx;

    gguf_context_ptr gguf_ctx(gguf_ex_ctx(ggml_ctx));
    struct ggml_context* ctx = ggml_ctx.get();

    char const* name = tensorname.c_str();
    struct ggml_tensor* tensor = ggml_get_tensor(ctx, name);
    bool ok = true;
    if (!tensor) {
      printf("%s: ERROR ggml_get_tensor('%s')\n", __func__, name);
      return false;
    }

    {
      bool alreadyHost = ggml_get_data(tensor) || (tensor->buffer && ggml_backend_buft_is_host(ggml_backend_buffer_get_type(tensor->buffer)));
      printf("%s: got tensor '%s' is_host=%d buffer=%p data=%p\n", __func__, name, (int)alreadyHost, (void*)tensor->buffer, (void*)ggml_get_data(tensor));
      struct ggml_tensor * cpu_tensor;
      if (alreadyHost)
        cpu_tensor = tensor;
      else if (!alreadyHost) {
        cpu_tensor = ggml_new_tensor(ctx, tensor->type, ggml_n_dims(tensor), tensor->ne);
        if (!cpu_tensor) {
          printf("%s: ERROR ggml_new_tensor('%s')\n", __func__, name);
          return false;
        }
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx); // no API to free?
        struct ggml_tensor* cpy = ggml_cpy(ctx, tensor, cpu_tensor);  // no API to free?
        ggml_build_forward_expand(cgraph, cpy); // no API to free?
        ggml_graph_compute_with_ctx(ctx, cgraph, 1); // no API to free?
      }

      if (!save_npy_create(fname.c_str(), tensor)) {
        printf("%s: ERROR save_npy_create('%s', '%s')\n", __func__, fname.c_str(), name);
        return false;
      }
      printf("%s: wrote file '%s'\n", __func__, fname.c_str());

      if (!alreadyHost) {
        // everything is freed in ggml_free(ctx)?
      }
    }


    return ok;
}


int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s data.npy [tensorname]\n", argv[0]);
        return -1;
    }

    srand(123456);

    const std::string fname(argv[1]);
    // tensor_0 ... tensor_9
    const std::string tensorname(argc == 3 ? argv[2] : "tensor_0");

    GGML_ASSERT(gguf_ex_write_npy(fname, tensorname) && "failed to write numpy file");

    return 0;
}
