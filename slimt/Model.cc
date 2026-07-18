#include "slimt/Model.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "slimt/Aligned.hh"
#include "slimt/Arena.hh"
#include "slimt/Input.hh"
#include "slimt/Io.hh"
#include "slimt/QMM.hh"
#include "slimt/Shortlist.hh"
#include "slimt/Tensor.hh"
#include "slimt/TensorOps.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

namespace {

size_t model_id = 0;

Package<io::MmapFile> mmap_from(const Package<std::string> &package) {
  auto maybe_mmap = [](const std::string &path) {
    return path.empty() ? io::MmapFile() : io::MmapFile(path);
  };

  return {
      .model = maybe_mmap(package.model),                          //
      .vocabulary = maybe_mmap(package.vocabulary),                //
      .target_vocabulary = maybe_mmap(package.target_vocabulary),  //
      .shortlist = maybe_mmap(package.shortlist),                  //
  };
}

Package<View> view_from(const Package<io::MmapFile> &mmap) {
  return {
      .model = {mmap.model.data(), mmap.model.size()},                 //
      .vocabulary = {mmap.vocabulary.data(), mmap.vocabulary.size()},  //
      .target_vocabulary = {mmap.target_vocabulary.data(),             //
                            mmap.target_vocabulary.size()},            //
      .shortlist = {mmap.shortlist.data(), mmap.shortlist.size()},     //
  };
}

std::unique_ptr<Vocabulary> maybe_target_vocabulary(View target_view) {
  if (target_view.data == nullptr || target_view.size == 0) {
    return nullptr;
  }
  return std::make_unique<Vocabulary>(target_view);
}

}  // namespace

Model::Model(const Config &config, const Package<View> &package)
    : id_(model_id++),
      config_(config),
      view_(package),
      src_vocabulary_(package.vocabulary),
      tgt_vocabulary_(maybe_target_vocabulary(package.target_vocabulary)),
      processor_(src_vocabulary_),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.num_heads, config.feed_forward_depth, package.model),
      shortlist_generator_(make_shortlist_generator(
          package.shortlist, src_vocabulary_, target_vocabulary())) {}

Model::Model(const Config &config, const Package<std::string> &package)
    : id_(model_id++),
      config_(config),
      mmap_(mmap_from(package)),
      view_(view_from(*mmap_)),
      src_vocabulary_(view_.vocabulary),
      tgt_vocabulary_(maybe_target_vocabulary(view_.target_vocabulary)),
      processor_(src_vocabulary_),
      transformer_(config.encoder_layers, config.decoder_layers,
                   config.num_heads, config.feed_forward_depth, view_.model),
      shortlist_generator_(make_shortlist_generator(
          view_.shortlist, src_vocabulary_, target_vocabulary())) {}

std::optional<ShortlistGenerator> Model::make_shortlist_generator(
    View view, const Vocabulary &source, const Vocabulary &target) {
  if (view.data == nullptr || view.size == 0) {
    return std::nullopt;
  }
  return ShortlistGenerator(view, source, target);
}

namespace {
Tensor select_batch(const Tensor &tensor, const std::vector<size_t> &indices,
                    const std::string &name) {
  Shape shape = tensor.shape();
  shape.set_dim(0, indices.size());
  Tensor selected(tensor.type(), shape, name);

  if (indices.empty()) {
    return selected;
  }

  size_t batch_size = tensor.dim(0);
  size_t bytes_per_entry =
      size_in_bytes(tensor.type()) * tensor.size() / batch_size;
  const char *source = tensor.data<char>();
  char *target = selected.data<char>();
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t index = indices[i];
    std::memcpy(target + i * bytes_per_entry, source + index * bytes_per_entry,
                bytes_per_entry);
  }

  return selected;
}

// In-place variant of select_batch for tensors the decode loop owns: `keep`
// is ascending, so every kept slab moves to an offset at or before its
// source within the same buffer. No fresh allocation means the pages stay
// resident across compactions (a fresh heap tensor per finished sentence
// showed up as ~4% of runtime in page-fault handling).
void compact_batch(Tensor &tensor, const std::vector<size_t> &keep) {
  size_t batch_size = tensor.dim(0);
  size_t bytes_per_entry =
      size_in_bytes(tensor.type()) * tensor.size() / batch_size;
  char *data = tensor.data<char>();
  for (size_t i = 0; i < keep.size(); ++i) {
    if (keep[i] == i) {
      continue;
    }
    std::memmove(data + i * bytes_per_entry, data + keep[i] * bytes_per_entry,
                 bytes_per_entry);
  }
  tensor.shape().set_dim(0, keep.size());
}

void update_alignment(const std::vector<size_t> &active_to_original,
                      const std::vector<size_t> &lengths,
                      const std::vector<bool> &finished, const Tensor &attn,
                      Alignments &alignments) {
  const auto *data = attn.data<float>();
  size_t batch_size = attn.dim(-4);
  size_t num_heads = attn.dim(-3);
  size_t slice = attn.dim(-2);
  size_t source_length = attn.dim(-1);

  for (size_t id = 0; id < batch_size; id++) {
    size_t original_id = active_to_original[id];
    if (!finished[original_id]) {
      size_t batch_stride = (num_heads * slice * source_length);
      size_t head_stride = (slice * source_length);
      const float *alignment = data + id * batch_stride + head_stride * 0;
      size_t length = lengths[original_id];
      Distribution distribution(length);
      std::copy(alignment, alignment + length, distribution.data());
      alignments[original_id].push_back(std::move(distribution));
    }
  }
}
}  // namespace

Histories Model::decode(const Tensor &encoder_out, const Tensor &mask,
                        const RowShortlists &row_shortlists,
                        const std::vector<size_t> &lengths, float limit_factor,
                        Arena &arena, std::vector<double> *out_deficits,
                        const std::optional<AlternativesConfig> &alt_cfg,
                        const std::vector<Words> &forced_prefix) const {
  size_t batch_size = encoder_out.dim(-3);
  size_t source_sequence_length = encoder_out.dim(-2);

  const Decoder &decoder = transformer_.decoder();

  // One shortlisted output projection per distinct request in the batch
  // (rows of one request share their shortlist's shared_ptr, so grouping is
  // pointer equality). The column-select on `W` and the bias gather run once
  // per decode; each step then multiplies a request's rows only against its
  // own ~hundreds of candidates instead of a batch-wide union. A null entry
  // is the full-vocabulary group, for rows whose request has no shortlist.
  // Built before any ArenaScope so the tensors heap-allocate and survive
  // across step iterations.
  std::vector<std::shared_ptr<const Words>> group_words;
  std::vector<SelectedAffine> group_selected;
  std::vector<size_t> row_group(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    std::shared_ptr<const Words> words =
        row_shortlists.empty() ? nullptr : row_shortlists[i];
    size_t group = 0;
    while (group < group_words.size() && group_words[group] != words) {
      group++;
    }
    if (group == group_words.size()) {
      group_words.push_back(words);
      group_selected.push_back(words ? decoder.prepare_shortlisted_output(*words)
                                     : SelectedAffine{});
    }
    row_group[i] = group;
  }

  std::vector<bool> complete(batch_size, false);
  const Vocabulary &target_vocab = target_vocabulary();
  uint32_t eos = target_vocab.eos_id();
  // Cap each sentence's target length from its own (unpadded) source length.
  // A cap derived from the batch's padded length would let a sentence run
  // longer the longer its co-batched neighbours are, making the output
  // depend on batch composition. The additive slack exists because a purely
  // multiplicative cap starves short sources — "CHAPTER 102. A Bower in the
  // Arsacides." is 16 subwords, and 1.5×16 = 24 truncated the Spanish
  // mid-word ("...los Arsaci") — while barely moving the runaway bound for
  // long ones.
  constexpr size_t kTargetLengthSlack = 8;
  std::vector<size_t> target_caps(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    target_caps[i] =
        limit_factor * lengths[i] + kTargetLengthSlack;
  }
  // Per-token alternatives, accumulated in lock-step with `sentences` under the
  // same completeness guard so `alt_acc[s]` stays parallel to the sentence's
  // target tokens. `step_alternatives` holds the current step's per-active-row
  // candidates, filled by `sample_step` just before each `record`.
  std::vector<StepAlternatives> step_alternatives;
  std::vector<TokenAlternatives> alt_acc(alt_cfg ? batch_size : 0);

  auto record = [eos, &complete, &target_caps, &step_alternatives, &alt_acc](
                    const std::vector<size_t> &active_to_original, Words &step,
                    Sentences &sentences) {
    size_t finished = 0;
    for (size_t i = 0; i < step.size(); i++) {
      size_t original_id = active_to_original[i];
      if (not complete[original_id]) {
        sentences[original_id].push_back(step[i]);
        if (not alt_acc.empty()) {
          alt_acc[original_id].push_back(std::move(step_alternatives[i]));
        }
        complete[original_id] =
            (step[i] == eos) ||
            sentences[original_id].size() >= target_caps[original_id];
      }
    }
    for (bool done : complete) {
      finished += static_cast<int>(done);
    }
    return sentences.size() - finished;
  };

  // Initialize a first step.
  Sentences sentences(batch_size);
  Alignments alignments(sentences.size());

  std::vector<size_t> active_to_original(batch_size);
  std::iota(active_to_original.begin(), active_to_original.end(), 0);

  Words previous_slice = {};
  std::vector<Tensor> states = decoder.start_states(batch_size);
  std::vector<AttentionContext> contexts = decoder.prepare_contexts(encoder_out);

  const Tensor *active_encoder_out = &encoder_out;
  const Tensor *active_mask = &mask;
  Tensor selected_encoder_out;
  Tensor selected_mask;

  size_t decoder_rows = active_to_original.size();

  // Per-step transient tensors (Q/K/V projections, attention scores, FFN
  // intermediates, returned logits/attn) come from the shared scratch arena;
  // allocations that must outlive the step (states, contexts, encoder_out,
  // select_batch outputs from compact) happen outside arena scopes. The arena
  // arrives holding the now-dead encoder transients (encoder_out was cloned to
  // the heap), so reset before the first step reclaims that space.
  arena.reset();
  // Loop bound only; must dominate every per-row cap (lengths[i] ≤ the
  // padded source_sequence_length). Rows stop at their own cap via record().
  size_t max_seq_length =
      limit_factor * source_sequence_length + kTargetLengthSlack;

  auto compact = [&]() {
    std::vector<size_t> keep;
    std::vector<size_t> next_active_to_original;
    Words next_previous_slice;
    keep.reserve(active_to_original.size());
    next_active_to_original.reserve(active_to_original.size());
    next_previous_slice.reserve(active_to_original.size());

    for (size_t i = 0; i < active_to_original.size(); ++i) {
      size_t original_id = active_to_original[i];
      if (!complete[original_id]) {
        keep.push_back(i);
        next_active_to_original.push_back(original_id);
        next_previous_slice.push_back(previous_slice[i]);
      }
    }

    previous_slice = std::move(next_previous_slice);
    if (keep.empty()) {
      active_to_original.clear();
      return;
    }
    if (keep.size() == active_to_original.size()) {
      active_to_original = std::move(next_active_to_original);
      return;
    }

    if (active_encoder_out != &selected_encoder_out) {
      // First shrink: encoder_out and mask are caller-owned consts, so the
      // initial compaction copies them once; every later one is in place.
      selected_encoder_out =
          select_batch(*active_encoder_out, keep, active_encoder_out->name());
      selected_mask = select_batch(*active_mask, keep, active_mask->name());
      active_encoder_out = &selected_encoder_out;
      active_mask = &selected_mask;
    } else {
      compact_batch(selected_encoder_out, keep);
      compact_batch(selected_mask, keep);
    }
    for (Tensor &state : states) {
      compact_batch(state, keep);
    }
    for (AttentionContext &context : contexts) {
      compact_batch(context.keys, keep);
      compact_batch(context.values, keep);
    }
    active_to_original = std::move(next_active_to_original);
  };

  // Deficit-router state (only maintained when out_deficits is requested, i.e.
  // the greedy pass). Tokens above kDeficitBaseline credit the running sum,
  // below it accrue deficit; def_best holds each sentence's worst contiguous
  // run (Kadane).
  constexpr double kDeficitBaseline = 0.6;
  std::vector<double> def_cur(out_deficits ? batch_size : 0, 0.0);
  std::vector<double> def_best(out_deficits ? batch_size : 0, 0.0);

  // Expand an alternative first-subword into a complete word: fork the active
  // row's decoder state and greedily decode continuation subwords until the
  // next word boundary (a word-start piece or eos). Runs only on the
  // alternatives path, for a handful of candidates per gated position.
  constexpr size_t kMaxExpand = 8;
  auto expand_word = [&](size_t active_row, Word first, size_t next_step,
                         size_t group) -> Words {
    Words word{first};
    if (target_vocab.is_word_start(first) == false) {
      return word;
    }
    std::vector<size_t> pick{active_row};
    std::vector<Tensor> spec_states;
    spec_states.reserve(states.size());
    for (const Tensor &st : states) {
      spec_states.push_back(select_batch(st, pick, st.name()));
    }
    std::vector<AttentionContext> spec_contexts;
    spec_contexts.reserve(contexts.size());
    for (const AttentionContext &ctx : contexts) {
      spec_contexts.push_back(
          AttentionContext{select_batch(ctx.keys, pick, ctx.keys.name()),
                           select_batch(ctx.values, pick, ctx.values.name())});
    }
    Tensor spec_enc = select_batch(*active_encoder_out, pick, "spec_enc");
    Tensor spec_mask = select_batch(*active_mask, pick, "spec_mask");

    Words prev{first};
    for (size_t e = 0; e < kMaxExpand; ++e) {
      auto [spec_hidden, spec_attn] = decoder.step(
          spec_enc, spec_mask, spec_states, spec_contexts, prev, next_step + e);
      Tensor logits =
          group_words[group]
              ? affine_with_selected(group_selected[group], spec_hidden,
                                     "spec_logits")
              : decoder.project(spec_hidden);
      size_t stride =
          group_words[group] ? group_words[group]->size() : target_vocab.size();
      size_t local = argmax(logits.data<float>(), stride);
      Word next = group_words[group] ? (*group_words[group])[local]
                                     : static_cast<Word>(local);
      if (next == eos || target_vocab.ends_word(next)) {
        break;
      }
      word.push_back(next);
      prev = Words{next};
    }
    return word;
  };

  auto sample_step = [&](const Tensor &hidden, size_t step_index) {
    size_t active_rows = active_to_original.size();
    size_t hidden_dim = hidden.dim(-1);
    Words sampled(active_rows);
    if (alt_cfg) {
      step_alternatives.assign(active_rows, {});
    }

    std::vector<std::vector<size_t>> members(group_words.size());
    for (size_t i = 0; i < active_rows; ++i) {
      members[row_group[active_to_original[i]]].push_back(i);
    }

    const float *hidden_data = hidden.data<float>();
    for (size_t g = 0; g < members.size(); ++g) {
      if (members[g].empty()) {
        continue;
      }
      Tensor gathered;
      const Tensor *x = &hidden;
      if (members[g].size() != active_rows) {
        gathered = Tensor(Type::f32, Shape({members[g].size(), 1, hidden_dim}),
                          "grouped_hidden");
        float *target = gathered.data<float>();
        for (size_t j = 0; j < members[g].size(); ++j) {
          const float *source = hidden_data + members[g][j] * hidden_dim;
          std::copy(source, source + hidden_dim, target + j * hidden_dim);
        }
        x = &gathered;
      }

      Tensor logits = group_words[g]
                          ? affine_with_selected(group_selected[g], *x, "logits")
                          : decoder.project(*x);
      size_t stride =
          group_words[g] ? group_words[g]->size() : target_vocab.size();
      Words group_sample =
          group_words[g]
              ? greedy_sample_from_words(logits, target_vocab, *group_words[g],
                                         members[g].size())
              : greedy_sample(logits, target_vocab, members[g].size());

      // Online max-contiguous log-prob deficit (Kadane) per sentence, from the
      // chosen token's softmax probability — the two-pass router's flag signal.
      if (out_deficits != nullptr) {
        const float *L = logits.data<float>();
        for (size_t j = 0; j < members[g].size(); ++j) {
          const float *r = L + j * stride;
          float mx = r[0];
          for (size_t k = 1; k < stride; ++k) mx = std::max(mx, r[k]);
          double denom = 0.0;
          for (size_t k = 0; k < stride; ++k) {
            denom += std::exp(static_cast<double>(r[k] - mx));
          }
          double d = std::log(kDeficitBaseline) - std::log(1.0 / denom);
          size_t orig = active_to_original[members[g][j]];
          def_cur[orig] = std::max(d, def_cur[orig] + d);
          def_best[orig] = std::max(def_best[orig], def_cur[orig]);
        }
      }

      // Harvest each row's alternatives, with the word currently at this
      // position first as the confidence anchor: the forced token on a
      // user-confirmed prefix (so every steered word stays swappable and
      // revertible), else the greedy argmax. The offered alternatives are the
      // other top-k words — for a forced word that naturally includes the
      // original the model preferred. Only word-boundary positions carry
      // alternatives; each alternative first-subword is expanded to a whole word.
      if (alt_cfg) {
        const float *L = logits.data<float>();
        size_t kk = std::min(alt_cfg->top_k + 1, stride);
        for (size_t j = 0; j < members[g].size(); ++j) {
          size_t orig = active_to_original[members[g][j]];
          bool forced = orig < forced_prefix.size() &&
                        step_index < forced_prefix[orig].size();
          Word chosen = forced ? forced_prefix[orig][step_index] : group_sample[j];
          if (!target_vocab.is_word_start(chosen)) {
            continue;
          }
          const float *r = L + j * stride;
          float mx = r[0];
          for (size_t k = 1; k < stride; ++k) mx = std::max(mx, r[k]);
          double denom = 0.0;
          for (size_t k = 0; k < stride; ++k) {
            denom += std::exp(static_cast<double>(r[k] - mx));
          }
          std::vector<size_t> top;
          top.reserve(kk + 1);
          for (size_t k = 0; k < stride; ++k) {
            if (top.size() < kk || r[k] > r[top.back()]) {
              auto pos = std::lower_bound(
                  top.begin(), top.end(), k,
                  [&](size_t a, size_t b) { return r[a] > r[b]; });
              top.insert(pos, k);
              if (top.size() > kk) top.pop_back();
            }
          }
          auto prob_of = [&](size_t local) {
            return static_cast<float>(
                std::exp(static_cast<double>(r[local] - mx)) / denom);
          };
          // Probability of the word currently at this position. For a forced
          // token that isn't the argmax, look up its column in the row.
          float chosen_prob;
          if (!forced) {
            chosen_prob = prob_of(top[0]);
          } else {
            size_t chosen_local = stride;
            if (group_words[g]) {
              for (size_t k = 0; k < stride; ++k) {
                if ((*group_words[g])[k] == chosen) {
                  chosen_local = k;
                  break;
                }
              }
            } else if (chosen < stride) {
              chosen_local = chosen;
            }
            if (chosen_local == stride) {
              continue;  // forced token outside this row's vocab slice
            }
            chosen_prob = prob_of(chosen_local);
          }
          StepAlternatives candidates;
          candidates.push_back(TokenAlternative{Words{chosen}, chosen_prob});
          for (size_t rank = 0; rank < top.size(); ++rank) {
            Word w = group_words[g] ? (*group_words[g])[top[rank]]
                                    : static_cast<Word>(top[rank]);
            if (w == chosen) {
              continue;  // never offer the current word as its own alternative
            }
            float p = prob_of(top[rank]);
            if (p < alt_cfg->min_prob) {
              break;
            }
            if (!target_vocab.is_word_start(w)) {
              continue;
            }
            candidates.push_back(TokenAlternative{
                expand_word(members[g][j], w, step_index + 1, g), p});
            if (candidates.size() > alt_cfg->top_k) {
              break;
            }
          }
          step_alternatives[members[g][j]] = std::move(candidates);
        }
      }

      // Force the user-confirmed prefix token for early positions; free-run
      // (argmax) thereafter.
      for (size_t j = 0; j < members[g].size(); ++j) {
        size_t orig = active_to_original[members[g][j]];
        bool forced = orig < forced_prefix.size() &&
                      step_index < forced_prefix[orig].size();
        sampled[members[g][j]] =
            forced ? forced_prefix[orig][step_index] : group_sample[j];
      }
    }
    return sampled;
  };

  size_t remaining;
  {
    ArenaScope arena_scope(arena);
    auto [hidden, attn] = decoder.step(*active_encoder_out, *active_mask,
                                       states, contexts, previous_slice,
                                       /*step_index=*/0);
    previous_slice = sample_step(hidden, /*step_index=*/0);
    update_alignment(active_to_original, lengths, complete, attn,
                     alignments);
    remaining = record(active_to_original, previous_slice, sentences);
  }
  compact();

  size_t steps = 1;
  for (size_t i = 1; i < max_seq_length && remaining > 0; i++) {
    arena.reset();
    decoder_rows += active_to_original.size();
    {
      ArenaScope arena_scope(arena);
      auto [hidden, attn] = decoder.step(*active_encoder_out, *active_mask,
                                         states, contexts, previous_slice,
                                         /*step_index=*/i);
      steps++;
      previous_slice = sample_step(hidden, /*step_index=*/i);
      update_alignment(active_to_original, lengths, complete, attn,
                       alignments);
      remaining = record(active_to_original, previous_slice, sentences);
    }
    compact();
  }

  // The per-step output projections cached ruy packs of the per-request
  // shortlisted Ws (heap-owned, pointer-keyed cache); drop them before
  // `group_selected` is freed so a later allocation reusing an address
  // can't hit a stale pack.
  qmm::clear_standalone_pack_cache();

  if (std::getenv("SLIMT_DECODE_STATS") != nullptr) {
    size_t target_tokens = 0;
    for (const auto &sentence : sentences) {
      target_tokens += sentence.size();
    }
    size_t wasted_rows =
        decoder_rows > target_tokens ? decoder_rows - target_tokens : 0;
    size_t shortlisted_groups = 0;
    for (const auto &words : group_words) {
      shortlisted_groups += words ? 1 : 0;
    }
    std::fprintf(stderr,
                 "[decode-stats] batch=%zu src_len=%zu steps=%zu rows=%zu "
                 "target_tokens=%zu wasted_rows=%zu limit=%zu groups=%zu "
                 "shortlisted_groups=%zu\n",
                 batch_size, source_sequence_length, steps, decoder_rows,
                 target_tokens, wasted_rows, max_seq_length,
                 group_words.size(), shortlisted_groups);
  }

  if (out_deficits != nullptr) {
    *out_deficits = std::move(def_best);
  }

  Histories histories;
  for (size_t i = 0; i < sentences.size(); i++) {
    Hypothesis hypothesis{
        .target = std::move(sentences[i]),      //
        .alignment = std::move(alignments[i]),  //
        .alternatives = alt_acc.empty() ? TokenAlternatives{}
                                        : std::move(alt_acc[i])  //
    };
    auto history = std::make_shared<Hypothesis>(std::move(hypothesis));
    histories.push_back(std::move(history));
  }

  return histories;
}

Histories Model::decode_beam(const Tensor &encoder_out, const Tensor &mask,
                             const RowShortlists &row_shortlists,
                             const std::vector<size_t> &lengths,
                             const std::vector<size_t> &beam_widths,
                             float limit_factor, Arena &arena) const {
  size_t num_sentences = encoder_out.dim(-3);
  size_t source_sequence_length = encoder_out.dim(-2);
  const Decoder &decoder = transformer_.decoder();
  const Vocabulary &target_vocab = target_vocabulary();
  auto eos = static_cast<uint32_t>(target_vocab.eos_id());

  // Replicate each sentence's row into its beam rows. `sentence_of[r]` maps a
  // beam row back to its sentence; `rows_of[s]` lists a sentence's beam rows.
  std::vector<size_t> rep;
  std::vector<size_t> sentence_of;
  std::vector<std::vector<size_t>> rows_of(num_sentences);
  for (size_t s = 0; s < num_sentences; ++s) {
    for (size_t b = 0; b < beam_widths[s]; ++b) {
      rows_of[s].push_back(rep.size());
      sentence_of.push_back(s);
      rep.push_back(s);
    }
  }
  size_t rows = rep.size();

  Tensor enc = select_batch(encoder_out, rep, "encoder_out_beam");
  Tensor msk = select_batch(mask, rep, "mask_beam");

  // Per-row shortlist groups (rows of one sentence share its shortlist).
  std::vector<std::shared_ptr<const Words>> group_words;
  std::vector<SelectedAffine> group_selected;
  std::vector<size_t> row_group(rows);
  for (size_t r = 0; r < rows; ++r) {
    std::shared_ptr<const Words> words =
        row_shortlists.empty() ? nullptr : row_shortlists[rep[r]];
    size_t group = 0;
    while (group < group_words.size() && group_words[group] != words) {
      group++;
    }
    if (group == group_words.size()) {
      group_words.push_back(words);
      group_selected.push_back(
          words ? decoder.prepare_shortlisted_output(*words) : SelectedAffine{});
    }
    row_group[r] = group;
  }

  std::vector<Tensor> states = decoder.start_states(rows);
  std::vector<AttentionContext> contexts = decoder.prepare_contexts(enc);

  // Only beam 0 of each sentence is live at step 0, so the first expansion
  // picks `beam_widths[s]` distinct continuations rather than the same token.
  constexpr double kNegInf = -1e30;
  std::vector<double> score(rows, kNegInf);
  for (size_t s = 0; s < num_sentences; ++s) {
    score[rows_of[s][0]] = 0.0;
  }
  std::vector<Words> seq(rows);
  std::vector<Alignment> align(rows);
  std::vector<bool> done(rows, false);

  std::vector<size_t> caps(num_sentences);
  for (size_t s = 0; s < num_sentences; ++s) {
    caps[s] = limit_factor * lengths[s] + 8;
  }
  size_t max_len = limit_factor * source_sequence_length + 8;

  // A scored continuation of a parent beam row.
  struct Cand {
    size_t parent;
    uint32_t token;
    double score;
  };

  Words previous;  // empty at step 0 -> zero embedding for every row
  arena.reset();
  for (size_t step = 0; step < max_len; ++step) {
    bool all_done = true;
    for (size_t r = 0; r < rows; ++r) {
      all_done = all_done && done[r];
    }
    if (all_done) {
      break;
    }

    std::vector<size_t> parent(rows);
    std::vector<uint32_t> token(rows);
    std::vector<double> next_score(rows);
    std::vector<Words> next_seq(rows);
    std::vector<Alignment> next_align(rows);
    std::vector<bool> next_done(rows);

    {
      ArenaScope arena_scope(arena);
      auto [hidden, attn] =
          decoder.step(enc, msk, states, contexts, previous, step);
      // Per beam row, the source attention distribution (head 0) for this
      // step's token, carried alongside the token through beam reordering so
      // the winning hypothesis gets greedy-equivalent alignments. The pivot
      // remap assumes one alignment row per target token; an empty matrix here
      // makes combine() read out of bounds.
      std::vector<Distribution> row_dist(rows);
      {
        const float *attn_data = attn.data<float>();
        size_t num_heads = attn.dim(-3);
        size_t slice = attn.dim(-2);
        size_t attn_source_length = attn.dim(-1);
        size_t row_stride = num_heads * slice * attn_source_length;
        for (size_t r = 0; r < rows; ++r) {
          size_t length = lengths[sentence_of[r]];
          const float *src = attn_data + r * row_stride;
          row_dist[r].assign(src, src + length);
        }
      }
      size_t hidden_dim = hidden.dim(-1);
      const float *hidden_data = hidden.data<float>();

      // Per beam row, its top candidate tokens (global ids) with log-probs.
      std::vector<std::vector<Cand>> row_cands(rows);
      std::vector<std::vector<size_t>> members(group_words.size());
      for (size_t r = 0; r < rows; ++r) {
        members[row_group[r]].push_back(r);
      }
      for (size_t g = 0; g < members.size(); ++g) {
        if (members[g].empty()) {
          continue;
        }
        Tensor gathered;
        const Tensor *x = &hidden;
        if (members[g].size() != rows) {
          gathered = Tensor(Type::f32, Shape({members[g].size(), 1, hidden_dim}),
                            "grouped_hidden");
          float *dst = gathered.data<float>();
          for (size_t j = 0; j < members[g].size(); ++j) {
            const float *src = hidden_data + members[g][j] * hidden_dim;
            std::copy(src, src + hidden_dim, dst + j * hidden_dim);
          }
          x = &gathered;
        }
        Tensor logits = group_words[g]
                            ? affine_with_selected(group_selected[g], *x, "logits")
                            : decoder.project(*x);
        size_t stride =
            group_words[g] ? group_words[g]->size() : target_vocab.size();
        const float *data = logits.data<float>();
        for (size_t j = 0; j < members[g].size(); ++j) {
          size_t r = members[g][j];
          const float *rowl = data + j * stride;
          float mx = rowl[0];
          for (size_t k = 1; k < stride; ++k) mx = std::max(mx, rowl[k]);
          double denom = 0.0;
          for (size_t k = 0; k < stride; ++k) {
            denom += std::exp(static_cast<double>(rowl[k] - mx));
          }
          double log_denom = std::log(denom);
          size_t want = beam_widths[sentence_of[r]];
          std::vector<size_t> order(stride);
          std::iota(order.begin(), order.end(), 0);
          size_t take = std::min(want, stride);
          std::partial_sort(
              order.begin(), order.begin() + take, order.end(),
              [&](size_t a, size_t b) { return rowl[a] > rowl[b]; });
          for (size_t t = 0; t < take; ++t) {
            size_t local = order[t];
            uint32_t gid =
                group_words[g] ? (*group_words[g])[local] : static_cast<uint32_t>(local);
            double lp = static_cast<double>(rowl[local] - mx) - log_denom;
            row_cands[r].push_back({r, gid, score[r] + lp});
          }
        }
      }

      // Per sentence, pick the top-k continuations across its beams. A finished
      // beam carries forward as a single frozen candidate so it keeps competing.
      for (size_t s = 0; s < num_sentences; ++s) {
        std::vector<Cand> cands;
        for (size_t r : rows_of[s]) {
          if (done[r]) {
            cands.push_back({r, eos, score[r]});
          } else {
            for (const Cand &c : row_cands[r]) {
              cands.push_back(c);
            }
          }
        }
        size_t k = rows_of[s].size();
        std::partial_sort(
            cands.begin(), cands.begin() + std::min(k, cands.size()), cands.end(),
            [](const Cand &a, const Cand &b) { return a.score > b.score; });
        for (size_t b = 0; b < k; ++b) {
          size_t slot = rows_of[s][b];
          const Cand &c = cands[std::min(b, cands.size() - 1)];
          parent[slot] = c.parent;
          token[slot] = c.token;
          next_score[slot] = c.score;
          bool parent_done = done[c.parent];
          next_seq[slot] = seq[c.parent];
          next_align[slot] = align[c.parent];
          if (!parent_done) {
            next_seq[slot].push_back(c.token);
            next_align[slot].push_back(row_dist[c.parent]);
          }
          next_done[slot] =
              parent_done || c.token == eos || next_seq[slot].size() >= caps[s];
        }
      }
    }

    // Gather SSRU states to follow each new beam's parent. Contexts are shared
    // across a sentence's beams (same source), so they need no reordering.
    for (Tensor &state : states) {
      state = select_batch(state, parent, state.name());
    }
    score = std::move(next_score);
    seq = std::move(next_seq);
    align = std::move(next_align);
    done = std::move(next_done);
    previous = Words(token.begin(), token.end());
  }

  // Drop ruy packs of the decode-scoped shortlisted Ws before group_selected
  // frees them, so a later allocation reusing an address can't hit a stale
  // pack (same contract as the greedy decode).
  qmm::clear_standalone_pack_cache();

  Histories histories;
  histories.reserve(num_sentences);
  for (size_t s = 0; s < num_sentences; ++s) {
    size_t best = rows_of[s][0];
    double best_norm = kNegInf;
    for (size_t r : rows_of[s]) {
      double norm = seq[r].empty() ? kNegInf : score[r] / seq[r].size();
      if (norm > best_norm) {
        best_norm = norm;
        best = r;
      }
    }
    Hypothesis hypothesis{.target = std::move(seq[best]),
                          .alignment = std::move(align[best])};
    histories.push_back(std::make_shared<Hypothesis>(std::move(hypothesis)));
  }
  return histories;
}

Histories Model::forward(const Input &input) const {
  const Tensor &indices = input.indices();
  const Tensor &mask = input.mask();

  // uint64_t batch_size = indices.dim(-2);
  // uint64_t sequence_length = indices.dim(-1);
  // uint64_t embed_dim = embedding_.dim(-1);

  // One scratch arena per worker thread, shared by the encoder and the decode
  // loop. Encoder transients (word embedding, per-layer Q/K/V/O and FFN
  // projections, self-attention scores) come from it; encoder_out is cloned
  // out to the heap once the scope closes so it survives into decode(), which
  // then resets and reuses the same arena per step.
  constexpr size_t kArenaInitialBytes = 8 << 20;  // 8 MiB
  static thread_local Arena arena(kArenaInitialBytes);
  arena.reset();
  Tensor encoder_out;
  {
    Tensor arena_encoder_out;
    {
      ArenaScope arena_scope(arena);
      Tensor word_embedding =
          index_select(transformer_.embedding(), indices, "word_embedding");
      transform_embedding(word_embedding);

      // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L570
      // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L133
      arena_encoder_out = transformer_.encoder().forward(word_embedding, mask);
    }
    encoder_out = arena_encoder_out.clone("encoder_out");
  }

  // Two-pass robust decode: greedy first (capturing each sentence's max
  // contiguous log-prob deficit), then re-decode only the low-confidence
  // sentences with batched beam search, reusing the encoder output. The beam
  // width is stratified by deficit. SLIMT_ROBUST_D overrides the re-decode
  // threshold; 0 disables the pass (pure greedy, no deficit overhead).
  static const double robust_d = [] {
    const char *v = std::getenv("SLIMT_ROBUST_D");
    return v != nullptr ? std::strtod(v, nullptr) : 1.0;
  }();
  constexpr double kBeam3Deficit = 1.5;

  // Harvesting per-token alternatives needs the greedy distributions, which the
  // robust beam re-decode would discard. A batch can mix rows that requested
  // alternatives with rows that didn't (the Async batcher packs across
  // requests), so deficits are still computed and the beam still runs — just
  // not for the rows that asked for alternatives, whose greedy hypotheses are
  // kept intact.
  const std::optional<AlternativesConfig> &alt_cfg = input.alternatives();
  bool robust = robust_d > 0.0;

  std::vector<double> deficits;
  Histories histories =
      decode(encoder_out, mask, input.shortlist_rows(), input.lengths(),
             input.limit_factor(), arena, robust ? &deficits : nullptr, alt_cfg,
             input.forced());

  if (robust) {
    const std::vector<bool> &alt_rows = input.alternatives_rows();
    const std::vector<Words> &forced = input.forced();
    std::vector<size_t> flagged;
    std::vector<size_t> widths;
    for (size_t i = 0; i < deficits.size(); ++i) {
      if (deficits[i] <= robust_d) {
        continue;
      }
      if (i < forced.size() && !forced[i].empty()) {
        continue;  // forced-prefix rows must stay on the greedy path
      }
      if (!alt_rows.empty() && alt_rows[i]) {
        continue;
      }
      flagged.push_back(i);
      widths.push_back(deficits[i] >= kBeam3Deficit ? 3 : 2);
    }
    if (!flagged.empty()) {
      Tensor enc = select_batch(encoder_out, flagged, "encoder_out_flagged");
      Tensor msk = select_batch(mask, flagged, "mask_flagged");
      RowShortlists shortlist;
      if (!input.shortlist_rows().empty()) {
        for (size_t i : flagged) {
          shortlist.push_back(input.shortlist_rows()[i]);
        }
      }
      std::vector<size_t> lengths;
      for (size_t i : flagged) {
        lengths.push_back(input.lengths()[i]);
      }
      Histories re_decoded = decode_beam(enc, msk, shortlist, lengths, widths,
                                         input.limit_factor(), arena);
      for (size_t j = 0; j < flagged.size(); ++j) {
        histories[flagged[j]] = std::move(re_decoded[j]);
      }
    }
  }

  return histories;
}

}  // namespace slimt
