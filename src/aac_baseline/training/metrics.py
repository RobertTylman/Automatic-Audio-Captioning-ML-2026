from collections import Counter
import math
import re


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _closest_reference_length(candidate_length: int, reference_lengths: list[int]) -> int:
    return min(reference_lengths, key=lambda length: (abs(length - candidate_length), length))


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for right_index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[right_index - 1] + 1)
            else:
                current.append(max(previous[right_index], current[-1]))
        previous = current
    return previous[-1]


def compute_caption_metrics(
    predictions: list[str],
    references: list[list[str]],
) -> dict[str, float]:
    """
    Lightweight caption metrics for validation-time monitoring.

    These are not a replacement for the official DCASE/COCO evaluation stack,
    but they are useful smoke signals while training.
    """

    if not predictions:
        return {
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "rouge_l": 0.0,
            "gen_len": 0.0,
        }

    tokenized_predictions = [_tokenize(prediction) for prediction in predictions]
    tokenized_references = [
        [_tokenize(reference) for reference in reference_group]
        for reference_group in references
    ]

    total_candidate_length = 0
    total_reference_length = 0
    clipped_matches = [0, 0, 0, 0]
    total_candidate_ngrams = [0, 0, 0, 0]
    rouge_l_scores = []

    for prediction_tokens, reference_tokens_group in zip(
        tokenized_predictions,
        tokenized_references,
    ):
        reference_tokens_group = [
            reference_tokens
            for reference_tokens in reference_tokens_group
            if reference_tokens
        ]
        if not reference_tokens_group:
            reference_tokens_group = [[]]

        candidate_length = len(prediction_tokens)
        reference_lengths = [len(reference_tokens) for reference_tokens in reference_tokens_group]
        total_candidate_length += candidate_length
        total_reference_length += _closest_reference_length(candidate_length, reference_lengths)

        for ngram_size in range(1, 5):
            candidate_ngrams = _ngrams(prediction_tokens, ngram_size)
            total_candidate_ngrams[ngram_size - 1] += sum(candidate_ngrams.values())

            max_reference_ngrams: Counter[tuple[str, ...]] = Counter()
            for reference_tokens in reference_tokens_group:
                reference_ngrams = _ngrams(reference_tokens, ngram_size)
                for ngram, count in reference_ngrams.items():
                    max_reference_ngrams[ngram] = max(max_reference_ngrams[ngram], count)

            clipped_matches[ngram_size - 1] += sum(
                min(count, max_reference_ngrams[ngram])
                for ngram, count in candidate_ngrams.items()
            )

        rouge_candidates = []
        for reference_tokens in reference_tokens_group:
            lcs = _lcs_length(prediction_tokens, reference_tokens)
            if not prediction_tokens or not reference_tokens:
                rouge_candidates.append(0.0)
                continue
            precision = lcs / len(prediction_tokens)
            recall = lcs / len(reference_tokens)
            rouge_candidates.append(
                0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            )
        rouge_l_scores.append(max(rouge_candidates))

    if total_candidate_length == 0:
        brevity_penalty = 0.0
    elif total_candidate_length > total_reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - total_reference_length / total_candidate_length)

    metrics = {}
    for order in range(1, 5):
        if any(total_candidate_ngrams[index] == 0 for index in range(order)):
            metrics[f"bleu{order}"] = 0.0
            continue

        log_precision_sum = 0.0
        for index in range(order):
            # Add-one smoothing keeps early training from collapsing BLEU to zero
            # whenever a higher-order n-gram has no matches yet.
            precision = (clipped_matches[index] + 1) / (total_candidate_ngrams[index] + 1)
            log_precision_sum += math.log(precision)
        metrics[f"bleu{order}"] = brevity_penalty * math.exp(log_precision_sum / order)

    metrics["rouge_l"] = sum(rouge_l_scores) / len(rouge_l_scores)
    metrics["gen_len"] = total_candidate_length / len(predictions)
    return metrics
