#include "gguf.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include <vector>
#include <string>
#include <cstdlib>

typedef struct gguf_context * GgContext;

static GgContext gguf_ex_ctx(ggml_context_ptr &ctx_data_out) {
    GgContext ctx = gguf_init_empty();

    gguf_set_val_u8  (ctx, "some.parameter.uint8",    0x12);
    gguf_set_val_i8  (ctx, "some.parameter.int8",    -0x13);
    gguf_set_val_u16 (ctx, "some.parameter.uint16",   0x1234);
    gguf_set_val_i16 (ctx, "some.parameter.int16",   -0x1235);
    gguf_set_val_u32 (ctx, "some.parameter.uint32",   0x12345678);
    gguf_set_val_i32 (ctx, "some.parameter.int32",   -0x12345679);
    gguf_set_val_f32 (ctx, "some.parameter.float32",  0.123456789f);
    gguf_set_val_u64 (ctx, "some.parameter.uint64",   0x123456789abcdef0ull);
    gguf_set_val_i64 (ctx, "some.parameter.int64",   -0x123456789abcdef1ll);
    gguf_set_val_f64 (ctx, "some.parameter.float64",  0.1234567890123456789);
    gguf_set_val_bool(ctx, "some.parameter.bool",     true);
    gguf_set_val_str (ctx, "some.parameter.string",   "hello world");

    gguf_set_arr_data(ctx, "some.parameter.arr.i16", GGUF_TYPE_INT16,   std::vector<int16_t>{ 1, 2, 3, 4, }.data(), 4);
    gguf_set_arr_data(ctx, "some.parameter.arr.f32", GGUF_TYPE_FLOAT32, std::vector<float>{ 3.145f, 2.718f, 1.414f, }.data(), 3);
    gguf_set_arr_str (ctx, "some.parameter.arr.str",                    std::vector<const char *>{ "hello", "world", "!" }.data(), 3);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 128ull*1024ull*1024ull,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context *ctx_data = ggml_init(params);
    ctx_data_out.reset(ctx_data);

    const int n_tensors = 10;

    // tensor infos
    std::string name = "tensor_0";
    char *dig = &name.back(); // [name.size() - 1];
    for (int i = 0; i < n_tensors; ++i) {
        *dig = '0' + i;
        int64_t ne[GGML_MAX_DIMS] = { 1 };
        int32_t n_dims = std::rand() % GGML_MAX_DIMS + 1;

        for (int j = 0; j < n_dims; ++j) {
            ne[j] = rand() % 10 + 1;
        }

        struct ggml_tensor * cur = ggml_new_tensor(ctx_data, GGML_TYPE_F32, n_dims, ne);
        ggml_set_name(cur, name.c_str());

        {
            float * data = (float *) cur->data;
            for (int j = 0; j < ggml_nelements(cur); ++j) {
                data[j] = 100 + i + j;
            }
        }

        gguf_add_tensor(ctx, cur);
    }
    return ctx;
}
