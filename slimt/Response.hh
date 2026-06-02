#pragma once

#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/Types.hh"
#include "slimt/Utils.hh"

namespace slimt {

/// Response holds AnnotatedText(s) of source-text and translated text,
/// alignment information between source and target sub-words and sentences.
///
/// AnnotatedText provides an API to access markings of (sub)-word and
/// sentences boundaries, which are required to interpret Quality and
/// Alignment (s) at the moment.
struct Response {
  /// source text and annotations of (sub-)words and sentences.
  AnnotatedText source;

  /// translated text and annotations of (sub-)words and sentences.
  AnnotatedText target;

  /// Alignments between source and target. This is a collection of dense
  /// matrices providing
  ///    P[t][s] = p(source-token s  | target token t)
  /// with an alignment matrix for each sentence.
  std::vector<std::vector<std::vector<float>>> alignments;

  /// Per-sentence, per-target-token top-k decode candidates (global word id +
  /// softmax probability), with the chosen token first. Empty unless
  /// `Options::alternatives` was set. Raw and unfiltered — callers decode the
  /// word ids and decide which to surface.
  std::vector<TokenAlternatives> alternatives;

  /// Convenience function to obtain number of units translated. Same as
  /// `.source.sentence_count()` and `.target.sentence_count().` The processing
  /// of a text of into sentences are handled internally, and this information
  /// can be used to iterate through meaningful units of translation for which
  /// alignment and quality information are available.
  size_t size() const { return source.sentence_count(); }

  void to(Encoding encoding);
};

/// Options dictate how to construct a Response for an input string of
/// text to be translated.
struct Options {
  bool alignment{false};  ///< Include alignments or not.

  /// When set, decode greedily (skipping the robust beam re-decode) and harvest
  /// per-token alternatives into `Response::alternatives`.
  std::optional<AlternativesConfig> alternatives{};

  /// Target tokens to force for the first N decode steps (teacher-forcing),
  /// after which the model free-runs. Used to re-translate with a user-edited
  /// prefix. Empty means no forcing. Implies greedy decode.
  Words forced_prefix{};

  /// Fired once per Request when it finishes translating, from the worker
  /// thread. Used to report incremental progress without slicing the corpus
  /// into many small translate() calls (which starves the batcher). Must be
  /// non-blocking — workers call it inline between forward passes.
  std::function<void()> on_progress{};
};

std::vector<Alignment> remap_alignments(const Response &first,
                                        const Response &second);

// Combines two responses with first.target == second.source mapping alignments
// etc accordingly. There are several constraints which are matched by only the
// pivoting workflow in <>Service source, therefore this function is not for
// external use and in a hidden namespace.
Response combine(Response &&first, Response &&second);

using Responses = std::vector<Response>;

class Request;

/// Rich reporting handle for a request in-flight.   Wraps around an std::future
/// to achieve std::future things, also holds a Ptr<Request> which lets know how
/// many parts of the request are complete.
class Handle {
 public:
  Handle(const Ptr<Request> &request, size_t parts, Future &&future)
      : request_(request), parts_(parts), future_(std::move(future)) {}

  // The following information is picked up and exported via ::info()
  struct Info {
    double wps;
    Fraction parts;
    Fraction words;
    Fraction segments;
  };

  Handle::Info info();
  std::future<Response> &future() { return future_; }

 private:
  Ptr<Request> request_;

  size_t part_ = 0;
  size_t parts_;

  std::future<Response> future_;

  Timer timer_;
};

}  // namespace slimt
