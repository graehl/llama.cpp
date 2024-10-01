#include "arg.h"
#include "common.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <signal.h>
#include <windows.h>
#endif

#include "llama-beam.cpp"

// Used for debugging to print out beam tokens.
struct ostream_beam_view {
  llama_context* ctx;
  llama_beam_view beam_view;
};

static std::ostream& operator<<(std::ostream& os, const ostream_beam_view& obv) {
  os << "p(" << obv.beam_view.p << ") eob(" << std::boolalpha << obv.beam_view.eob << ") tokens(";
  for (size_t i = 0; i < obv.beam_view.n_tokens; ++i) {
    os << llama_token_to_piece(obv.ctx, obv.beam_view.tokens[i]);
  }
  return os << ')';
}

inline bool has_nl(char const* txt) {
  char const* s = txt;
  for (; *s; ++s)
    if (*s == '\n') {
      LLAMA_LOG_INFO("nl: %s\n", txt);
      return true;
    }
  LLAMA_LOG_INFO("no newline: %s\n", txt);
  return false;
}

inline bool has_nl(std::string const& txt) {
  return txt.find('\n') != std::string::npos;
}

static void beam_print_usage(int argc, char** argv) {
  printf(
      "Usage: %s [BEAM_WIDTH=2] [llama-cli args CARE: most are ineffective in beam search, try -f "
      "promptfile -m model.gguf --override-kv tokenizer.ggml.pre=str:llama3 ]\n",
      argv[0]);
}

// Put here anything you want back in beam_search_callback().
struct beam_search_callback_data {
  llama_context* ctx;
  /// nl char contained in token => end generation
  bool eob_nl = true;
  std::vector<llama_token> response;
  llama_token nl_tok = (llama_token)-2;
  llama_model const* model = nullptr;

#if 0
  /// bizarre - doesn't work at all (but llama_token_to_piece does)
  char const* text(llama_token tok) const {
    return llama_token_get_text(model, tok);
  }
#else
  std::string text(llama_token tok) const { return llama_token_to_piece(ctx, tok); }
#endif

  void init() {
    model = llama_get_model(ctx);
    LLAMA_LOG_INFO("nl ends generation: %d", eob_nl);
    if (eob_nl && model) {
      nl_tok = llama_token_nl(model);
      LLAMA_LOG_INFO("nl token id: %d text: '%s'\n", nl_tok, text(nl_tok));
    }
  }

  bool eob(llama_token tok) const {
    return llama_token_is_eog(model, tok) || eob_nl && (tok == nl_tok || has_nl(text(tok)));
  }
};

// In this case, end-of-beam (eob) is equivalent to end-of-sentence (eos) but this need not always be the
// same. For example, eob can be flagged due to maximum token length, stop words, etc.
static bool is_at_eob(const beam_search_callback_data& callback_data, const llama_token* tokens, size_t n_tokens) {
  return n_tokens && callback_data.eob(tokens[n_tokens - 1]);
}

// Function matching type llama_beam_search_callback_fn_t.
// Custom callback example is called each time the beams lengths increase:
//  * Show progress by printing ',' following by number of convergent beam tokens if any.
//  * When all beams converge to a common prefix, they are made available in beams_state.beams[0].
//    This is also called when the stop condition is met.
//    Collect tokens into std::vector<llama_token> response which is pointed to by callback_data.
static void beam_search_callback(void* callback_data_ptr, llama_beams_state beams_state) {
  auto& callback_data = *static_cast<beam_search_callback_data*>(callback_data_ptr);
  // Mark beams as EOS as needed.
  for (size_t i = 0; i < beams_state.n_beams; ++i) {
    llama_beam_view& beam_view = beams_state.beam_views[i];
    if (!beam_view.eob && is_at_eob(callback_data, beam_view.tokens, beam_view.n_tokens)) {
      beam_view.eob = true;
    }
  }
  printf(",");  // Show progress
  if (const size_t n = beams_state.common_prefix_length) {
    callback_data.response.resize(callback_data.response.size() + n);
    assert(0u < beams_state.n_beams);
    const llama_token* tokens = beams_state.beam_views[0].tokens;
    std::copy(tokens, tokens + n, callback_data.response.end() - n);
    printf("%zu", n);
  }
  fflush(stdout);
#if 1  // DEBUG: print current beams for this iteration
  std::cout << "\n\nCurrent beams (last_call=" << beams_state.last_call << "):\n";
  for (size_t i = 0; i < beams_state.n_beams; ++i) {
    std::cout << "beams[" << i << "]: " << ostream_beam_view{callback_data.ctx, beams_state.beam_views[i]}
              << std::endl;
  }
#endif
}

int main(int argc, char** argv) {

  // params.n_gpu_layers = 200;

  //---------------------------------
  // Print help :
  //---------------------------------

  bool have_arg = argc >= 2;
  char const* beam_arg = argv[1];
  bool have_beam = have_arg && beam_arg && beam_arg[0] != '-';

  gpt_params params;
  if (!have_beam || !gpt_params_parse(argc - 1, argv + 1, params, LLAMA_EXAMPLE_COMMON, beam_print_usage))
    return 1;

  //---------------------------------
  // Load parameters :
  //---------------------------------

  int const n_beams = std::stoi(beam_arg);

  //---------------------------------
  // Init LLM :
  //---------------------------------

  llama_backend_init();
  llama_numa_init(params.numa);


  llama_init_result r = llama_init_from_gpt_params(params);

  llama_model* model = r.model;
  llama_context* ctx = r.context;

  if (model == NULL) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return 1;
  }

  //---------------------------------
  // Tokenize the prompt :
  //---------------------------------

  const bool add_bos = llama_add_bos_token(model);
  if (!llama_model_has_encoder(model)) {
    GGML_ASSERT(llama_add_eos_token(model) != 1);
  }
  LLAMA_LOG_INFO("add_bos: %d\n", add_bos);

  std::vector<llama_token> embd_inp = llama_tokenize(ctx, params.prompt, true, true);

  // Should not run without any tokens
  if (embd_inp.empty()) {
    if (add_bos) {
      embd_inp.push_back(llama_token_bos(model));
      LLAMA_LOG_INFO("embd_inp was considered empty and bos was added\n");
    } else {
      LLAMA_LOG_ERROR("error: input is empty\n");
      return -1;
    }
  }

  const size_t max_context_size = llama_n_ctx(ctx);
  const size_t max_embd_inp_size = max_context_size - 4;

  if (embd_inp.size() > max_embd_inp_size) {
    fprintf(stderr, "%s: error: prompt too long (%zu tokens, max %zu)\n", __func__, embd_inp.size(),
            max_embd_inp_size);
    return 1;
  }

  fprintf(stderr, "\n\n");

  // Print the tokens from the prompt :

  for (auto id : embd_inp) {
    std::cout << llama_token_to_piece(ctx, id);
  }
  std::cout << std::flush;

  int n_past = 0;

  if (llama_decode(ctx, llama_batch_get_one(embd_inp.data(), embd_inp.size(), n_past, 0))) {
    fprintf(stderr, "%s : failed to eval prompt.\n", __func__);
    return 1;
  }
  n_past += embd_inp.size();

  beam_search_callback_data callback_data{ctx, params.sparams.stop_nl};
  callback_data.init();
  size_t const beam_width = n_beams;
  int const n_predict = params.n_predict > 0 ? params.n_predict : INT_MAX;
  llama_beam_search(ctx, beam_search_callback, &callback_data, beam_width, n_past, n_predict);

  std::cout << "\n\n";
  for (llama_token const token_id : callback_data.response) {
    std::cout << llama_token_to_piece(ctx, token_id);
  }
  std::cout << std::endl;

  llama_free(ctx);
  llama_free_model(model);

  llama_backend_free();

  return 0;
}
