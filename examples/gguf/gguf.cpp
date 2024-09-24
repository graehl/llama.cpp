#include "gguf-ctx.hpp"

#include <cstdio>
#include <string>
#include <sstream>
#include <vector>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

template <typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

static bool gguf_ex_write(const std::string & fname) {
    ggml_context_ptr ctx_data;

    gguf_context_ptr ctx{gguf_ex_ctx(ctx_data)};

    gguf_write_to_file(ctx.get(), fname.c_str(), false);

    printf("%s: wrote file '%s;\n", __func__, fname.c_str());

    return true;
}

// just read tensor info
static bool gguf_ex_read_0(const std::string & fname) {
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ NULL,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    if (!ctx) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    printf("%s: version:      %d\n", __func__, gguf_get_version(ctx));
    printf("%s: alignment:   %zu\n", __func__, gguf_get_alignment(ctx));
    printf("%s: data offset: %zu\n", __func__, gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = gguf_get_n_kv(ctx);

        printf("%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);

            printf("%s: kv[%d]: key = %s\n", __func__, i, key);
        }
    }

    // find kv string
    {
        const char * findkey = "some.parameter.string";

        const int keyidx = gguf_find_key(ctx, findkey);
        if (keyidx == -1) {
            printf("%s: find key: %s not found.\n", __func__, findkey);
        } else {
            const char * key_value = gguf_get_val_str(ctx, keyidx);
            printf("%s: find key: %s found, kv[%d] value = %s\n", __func__, findkey, keyidx, key_value);
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        printf("%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ctx, i);
            const size_t size   = gguf_get_tensor_size  (ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);

            printf("%s: tensor[%d]: name = %s, size = %zu, offset = %zu\n", __func__, i, name, size, offset);
        }
    }

    gguf_free(ctx);

    return true;
}

// read and create ggml_context containing the tensors and their data
static bool gguf_ex_read_1(const std::string & fname, bool check_data) {
    struct ggml_context * ctx_data = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    printf("%s: version:      %d\n", __func__, gguf_get_version(ctx));
    printf("%s: alignment:   %zu\n", __func__, gguf_get_alignment(ctx));
    printf("%s: data offset: %zu\n", __func__, gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = gguf_get_n_kv(ctx);

        printf("%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);

            printf("%s: kv[%d]: key = %s\n", __func__, i, key);
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        printf("%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ctx, i);
            const size_t size   = gguf_get_tensor_size  (ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);

            printf("%s: tensor[%d]: name = %s, size = %zu, offset = %zu\n", __func__, i, name, size, offset);
        }
    }

    // data
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        for (int i = 0; i < n_tensors; ++i) {
            printf("%s: reading tensor %d data\n", __func__, i);

            const char * name = gguf_get_tensor_name(ctx, i);

            struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);

            printf("%s: tensor[%d]: n_dims = %d, ne = (%d, %d, %d, %d), name = %s, data = %p\n",
                __func__, i, ggml_n_dims(cur), int(cur->ne[0]), int(cur->ne[1]), int(cur->ne[2]), int(cur->ne[3]), cur->name, cur->data);

            // print first 10 elements
            const float * data = (const float *) cur->data;

            printf("%s data[:10] : ", name);
            for (int j = 0; j < MIN(10, ggml_nelements(cur)); ++j) {
                printf("%f ", data[j]);
            }
            printf("\n\n");

            // check data
            if (check_data) {
                const float * data = (const float *) cur->data;
                for (int j = 0; j < ggml_nelements(cur); ++j) {
                    if (data[j] != 100 + i + j) {
                        fprintf(stderr, "%s: tensor[%d], data[%d]: found %f, expected %f\n", __func__, i, j, data[j], float(100 + i + j));
                        gguf_free(ctx);
                        return false;
                    }
                }
            }
        }
    }

    printf("%s: ctx_data size: %zu\n", __func__, ggml_get_mem_size(ctx_data));

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("usage: %s data.gguf r|w [n]\n", argv[0]);
        printf("r: read data.gguf file\n");
        printf("w: write data.gguf file\n");
        printf("n: no check of tensor data\n");
        return -1;
    }
    bool check_data = true;
    if (argc == 4) {
        check_data = false;
    }

    srand(123456);

    const std::string fname(argv[1]);
    const std::string mode (argv[2]);

    GGML_ASSERT((mode == "r" || mode == "w") && "mode must be r or w");

    if (mode == "w") {
        GGML_ASSERT(gguf_ex_write(fname) && "failed to write gguf file");
    } else if (mode == "r") {
        GGML_ASSERT(gguf_ex_read_0(fname) && "failed to read gguf file");
        GGML_ASSERT(gguf_ex_read_1(fname, check_data) && "failed to read gguf file");
    }

    return 0;
}
