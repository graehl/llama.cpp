#define LLAMA_API_INTERNAL
#include "llama-beam.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_RPC
#  include "ggml-rpc.h"
#endif

#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#  include "ggml-opencl.h"
#elif defined(GGML_USE_VULKAN)
#  include "ggml-vulkan.h"
#elif defined(GGML_USE_SYCL)
#  include "ggml-sycl.h"
#elif defined(GGML_USE_KOMPUTE)
#   include "ggml-kompute.h"
#endif

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif

// TODO: replace with ggml API call
#define QK_K 256

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
            #include <fcntl.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <forward_list>
#include <fstream>
#include <functional>
#include <future>
#include <initializer_list>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>


#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

#define LLAMA_MAX_NODES   8192
#define LLAMA_MAX_EXPERTS 160

//
// logging
//

LLAMA_ATTRIBUTE_FORMAT(2, 3)
static void llama_log_internal        (ggml_log_level level, const char* format, ...);
static void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLAMA_LOG_INFO(...)  llama_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLAMA_LOG_WARN(...)  llama_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

//
// Beam search
//

struct llama_beam {
    std::vector<llama_token> tokens;
    float p;  // Cumulative beam probability (renormalized relative to all beams)
    bool eob; // Initialize end-of-beam to false. Callback sets this to true.
    // Sort beams by probability. In case of ties, prefer beams at eob.
    bool operator<(const llama_beam & rhs) const {
        return std::make_pair(p, eob) < std::make_pair(rhs.p, rhs.eob);
    }
    // Shift off first n tokens and discard them.
    void shift_tokens(const size_t n) {
        if (n) {
            std::copy(tokens.begin() + n, tokens.end(), tokens.begin());
            tokens.resize(tokens.size() - n);
        }
    }
    llama_beam_view view() const { return {tokens.data(), tokens.size(), p, eob}; }
};

// A struct for calculating logit-related info.
struct llama_logit_info {
    const float * const logits;
    const int n_vocab;
    const float max_l;
    const float normalizer;
    struct sum_exp {
        float max_l;
        float operator()(float sum, float l) const { return sum + std::exp(l - max_l); }
    };
    llama_logit_info(llama_context * ctx)
      : logits(llama_get_logits(ctx))
      , n_vocab(llama_n_vocab(llama_get_model(ctx)))
      , max_l(*std::max_element(logits, logits + n_vocab))
      , normalizer(1.0f / std::accumulate(logits, logits + n_vocab, 0.0f, sum_exp{max_l}))
      { }
    llama_token_data get_token_data(const llama_token token_id) const {
        constexpr auto p = std::numeric_limits<float>::quiet_NaN();  // never used
        return {token_id, logits[token_id], p};
    }
    // Return top k token_data by logit.
    std::vector<llama_token_data> top_k(size_t k) {
        std::vector<llama_token_data> min_heap;  // min-heap by logit
        const llama_token k_min = std::min(static_cast<llama_token>(k), n_vocab);
        min_heap.reserve(k_min);
        for (llama_token token_id = 0 ; token_id < k_min ; ++token_id) {
            min_heap.push_back(get_token_data(token_id));
        }
        auto comp = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; };
        std::make_heap(min_heap.begin(), min_heap.end(), comp);
        for (llama_token token_id = k_min ; token_id < n_vocab ; ++token_id) {
            if (min_heap.front().logit < logits[token_id]) {
                std::pop_heap(min_heap.begin(), min_heap.end(), comp);
                min_heap.back().id = token_id;
                min_heap.back().logit = logits[token_id];
                std::push_heap(min_heap.begin(), min_heap.end(), comp);
            }
        }
        return min_heap;
    }
    float probability_from_logit(float logit) const {
        return normalizer * std::exp(logit - max_l);
    }
};

int constexpr REUSE_SINGLE_SEQUENCE = 0;

struct llama_beam_search_data {
    llama_context * ctx;
    size_t n_beams;
    int n_past;
    int n_predict;
    std::vector<llama_beam> beams;
    std::vector<llama_beam> next_beams;

    // Re-calculated on each loop iteration
    size_t common_prefix_length;

    // Used to communicate to/from callback on beams state.
    std::vector<llama_beam_view> beam_views;

    llama_beam_search_data(llama_context * ctx, size_t n_beams, int n_past, int n_predict)
      : ctx(ctx)
      , n_beams(n_beams)
      , n_past(n_past)
      , n_predict(n_predict)
      , beam_views(n_beams) {
        beams.reserve(n_beams);
        next_beams.reserve(n_beams);
    }

    // Collapse beams to a single beam given by index.
    void collapse_beams(const size_t beam_idx) {
        if (0u < beam_idx) {
            std::swap(beams[0], beams[beam_idx]);
        }
        beams.resize(1);
    }

    // Min-heaps are used to efficiently collect the top-k elements (k=n_beams).
    // The repetitive patterns below reflect the 2 stages of heaps:
    //  * Gather elements until the vector is full, then call std::make_heap() on it.
    //  * If the heap is full and a new element is found that should be included, pop the
    //    least element to the back(), replace it with the new, then push it into the heap.
    void fill_next_beams_by_top_probabilities(llama_beam & beam) {
        // Min-heaps use a greater-than comparator.
        const auto comp = [](const llama_beam & a, const llama_beam & b) { return a.p > b.p; };
        if (beam.eob) {
            // beam is at end-of-sentence, so just copy it to next_beams if its probability is high enough.
            if (next_beams.size() < n_beams) {
                next_beams.push_back(std::move(beam));
                if (next_beams.size() == n_beams) {
                    std::make_heap(next_beams.begin(), next_beams.end(), comp);
                }
            } else if (next_beams.front().p < beam.p) {
                std::pop_heap(next_beams.begin(), next_beams.end(), comp);
                next_beams.back() = std::move(beam);
                std::push_heap(next_beams.begin(), next_beams.end(), comp);
            }
        } else {
            // beam is not at end-of-sentence, so branch with next top_k tokens.
            if (!beam.tokens.empty()) {
                llama_decode(ctx, llama_batch_get_one(beam.tokens.data(), beam.tokens.size(), n_past, REUSE_SINGLE_SEQUENCE));
                //TODO: llama_batch_get_one 'this is a helper function to facilitate transition to the new batch API - avoid using it'
            }
            llama_logit_info logit_info(ctx);
            std::vector<llama_token_data> next_tokens = logit_info.top_k(n_beams);
            // TODO: top-k on gpu

            // Clear the kv slot so that other beams may try different tokens at this position. The llama_decode()
            // call in loop() will conclusively fill in the kv slot once the beams converge at this position.
            llama_kv_cache_seq_rm(ctx, REUSE_SINGLE_SEQUENCE, n_past, -1);

            size_t i=0;
            if (next_beams.size() < n_beams) {
                for (; next_beams.size() < n_beams ; ++i) {
                    llama_beam next_beam = beam;
                    next_beam.tokens.push_back(next_tokens[i].id);
                    next_beam.p *= logit_info.probability_from_logit(next_tokens[i].logit);
                    next_beams.push_back(std::move(next_beam));
                }
                std::make_heap(next_beams.begin(), next_beams.end(), comp);
            } else {
                for (; next_beams.front().p == 0.0f ; ++i) {
                    std::pop_heap(next_beams.begin(), next_beams.end(), comp);
                    llama_beam &next_beam = next_beams.back();
                    next_beam = beam;
                    next_beam.tokens.push_back(next_tokens[i].id);
                    next_beam.p *= logit_info.probability_from_logit(next_tokens[i].logit);
                    std::push_heap(next_beams.begin(), next_beams.end(), comp);
                }
            }
            for (; i < n_beams ; ++i) {
                const float next_p = beam.p * logit_info.probability_from_logit(next_tokens[i].logit);
                if (next_beams.front().p < next_p) {
                    std::pop_heap(next_beams.begin(), next_beams.end(), comp);
                    llama_beam &next_beam = next_beams.back();
                    next_beam = beam;
                    next_beam.tokens.push_back(next_tokens[i].id);
                    next_beam.p = next_p;
                    std::push_heap(next_beams.begin(), next_beams.end(), comp);
                }
            }
        }
    }

    // Find common_prefix_length based on beams.
    // Requires beams is not empty.
    size_t find_common_prefix_length() {
        size_t common_prefix_length = beams[0].tokens.size();
        for (size_t i = 1 ; i < beams.size() ; ++i) {
            common_prefix_length = std::min(common_prefix_length, beams[i].tokens.size());
            for (size_t j = 0 ; j < common_prefix_length ; ++j) {
                if (beams[0].tokens[j] != beams[i].tokens[j]) {
                    common_prefix_length = j;
                    break;
                }
            }
        }
        return common_prefix_length;
    }

    // Construct beams_state to send back to caller via the callback function.
    // Side effect: set common_prefix_length = find_common_prefix_length();
    llama_beams_state get_beams_state(const bool last_call) {
        for (size_t i = 0 ; i < beams.size() ; ++i) {
            beam_views[i] = beams[i].view();
        }
        common_prefix_length = find_common_prefix_length();
        return {beam_views.data(), beams.size(), common_prefix_length, last_call};
    }

    // Loop:
    //  * while i < n_predict, AND
    //  * any of the beams have not yet reached end-of-beam (eob), AND
    //  * the highest probability beam(s) (plural in case of ties) are not at end-of-sentence
    //    (since all other beam probabilities can only decrease)
    void loop(const llama_beam_search_callback_fn_t callback, void * const callback_data) {
        beams.push_back({{}, 1.0f, false});  // Start with one empty beam w/ probability = 1.0 and !eob.
        const auto not_eob = [](const llama_beam & beam) { return !beam.eob; };
        for (int i = 0 ; i < n_predict && std::any_of(beams.begin(),beams.end(),not_eob) &&
                       !beams[top_beam_index()].eob ; ++i) {
            callback(callback_data, get_beams_state(false));  // Sets common_prefix_length
            update_beams_from_beam_views();   // Update values (p,eob) that callback may have changed.
            if (common_prefix_length) {
                llama_decode(ctx, llama_batch_get_one(beams[0].tokens.data(), common_prefix_length, n_past, REUSE_SINGLE_SEQUENCE));
                n_past += common_prefix_length;
            }
            // Zero-out next_beam probabilities to place them last in following min-heap.
            std::for_each(next_beams.begin(), next_beams.end(), [](llama_beam & beam) { beam.p = 0.0f; });
            for (llama_beam & beam : beams) {
                beam.shift_tokens(common_prefix_length);
                fill_next_beams_by_top_probabilities(beam);
            }
            // next_beams become the beams of next/final iteration. Swap them to re-use memory.
            beams.swap(next_beams);
            renormalize_beam_probabilities(beams);
        }
        collapse_beams(top_beam_index());
        callback(callback_data, get_beams_state(true));
    }

    // As beams grow, the cumulative probabilities decrease.
    // Renormalize them to avoid floating point underflow.
    static void renormalize_beam_probabilities(std::vector<llama_beam> & beams) {
        const auto sum_p = [](float sum, llama_beam & beam) { return sum + beam.p; };
        const float inv_sum = 1.0f / std::accumulate(beams.begin(), beams.end(), 0.0f, sum_p);
        std::for_each(beams.begin(), beams.end(), [=](llama_beam & beam) { beam.p *= inv_sum; });
    }

    // Assumes beams is non-empty.  Uses llama_beam::operator<() for ordering.
    size_t top_beam_index() {
        return std::max_element(beams.begin(), beams.end()) - beams.begin();
    }

    // Copy (p,eob) for each beam which may have been changed by the callback.
    void update_beams_from_beam_views() {
        for (size_t i = 0 ; i < beams.size() ; ++i) {
            beams[i].p = beam_views[i].p;
            beams[i].eob = beam_views[i].eob;
        }
    }
};

void llama_beam_search(llama_context * ctx,
                       llama_beam_search_callback_fn_t callback, void * callback_data,
                       size_t n_beams, int n_past, int n_predict) {
    assert(ctx);
    const int64_t t_start_sample_us = ggml_time_us();

    llama_beam_search_data beam_search_data(ctx, n_beams, n_past, n_predict);

    beam_search_data.loop(callback, callback_data);

    //    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    //    ctx->n_sample++;
}
