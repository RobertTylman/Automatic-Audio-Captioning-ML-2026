import json
import os
import sys
import time
from string import punctuation
from urllib import error, request

import numpy as np
import pandas as pd


INFERENCE_DIR = sys.argv[1]
OPENAI_MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt-4.1-mini"
TOP_AUDIO_KEEP = int(sys.argv[3]) if len(sys.argv) > 3 else 32
TOP_PAIRWISE_KEEP = int(sys.argv[4]) if len(sys.argv) > 4 else 12
PAIRWISE_MODE = sys.argv[5] if len(sys.argv) > 5 else "instructor"

strip_punct_table = str.maketrans("", "", punctuation)

SAP_PROMPT = """Caption Summarization-Activating Prompt (SAP):
This is a hard problem. Carefully summarize in ONE detailed
sentence the following captions by different (possibly
incorrect) people describing the same audio. Be sure to describe
everything, including the source and background of the sounds,
identify when you're not sure. Do not allude to the existence of
the multiple captions. Do not start your summary with sentence
like "The audio (likely) features", "The audio (likely) captures"
and so on. Focus on describing the content of the audio. Note
that your summary MUST be about ten words and use
subject-predicate-object structure. Your summary NEEDS
to use present continuous tense whenever possible. HERE is
the question, Captions: {audio captions}.
"""


def build_chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def normalize_caption(text: str) -> str:
    # Conform with this repo's inference normalization.
    return text.strip().lower().translate(strip_punct_table)


def call_openai_chat_completion(prompt: str, model: str, retries: int = 3) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    url = build_chat_completions_url(base_url)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 80,
    }
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(retries):
        req = request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=120) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                content = resp_json["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    return "".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    ).strip()
                return str(content).strip()
        except (error.HTTPError, error.URLError, TimeoutError, KeyError) as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)

    raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_err}")


def build_prompt(captions):
    captions_block = "\n".join([f"- {c}" for c in captions])
    return SAP_PROMPT.replace("{audio captions}", captions_block)


def select_top_pairwise_central_captions(instructor_model, caption_dicts, top_k):
    if len(caption_dicts) <= 1:
        return caption_dicts

    if instructor_model is None:
        return caption_dicts[: min(top_k, len(caption_dicts))]

    texts = [c["text"] for c in caption_dicts]
    instructor_inputs = [["Represent the audio caption: ", t] for t in texts]
    text_embeds = instructor_model.encode(
        instructor_inputs,
        show_progress_bar=False,
        batch_size=128,
    )

    from sklearn.metrics.pairwise import cosine_similarity

    # Pairwise caption-caption cosine similarities.
    pairwise = cosine_similarity(text_embeds, text_embeds)
    np.fill_diagonal(pairwise, 0.0)
    denom = max(1, len(caption_dicts) - 1)
    centrality = pairwise.sum(axis=1) / denom

    ranked_idx = np.argsort(-centrality)
    keep_idx = ranked_idx[: min(top_k, len(caption_dicts))]
    return [caption_dicts[int(i)] for i in keep_idx]


if __name__ == "__main__":
    in_path = os.path.join(INFERENCE_DIR, "gen_captions_encoder_reranked.json")
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Missing encoder-reranked file: {in_path}. "
            "Run encoder_rerank_sampling_outputs.py first."
        )

    reranked = json.load(open(in_path))
    instructor_model = None
    if PAIRWISE_MODE != "none":
        try:
            from InstructorEmbedding import INSTRUCTOR

            instructor_model = INSTRUCTOR(
                "hkunlp/instructor-xl", cache_folder="./instructor_pretrained_weights"
            )
            instructor_model.eval()
        except Exception as exc:
            print(
                "[warn] could not load INSTRUCTOR model; "
                "falling back to rank-only selection for prompt candidates:",
                exc,
            )

    csv_out = {"file_name": [], "caption_predicted": []}
    details_out = []

    for i, sample in enumerate(reranked):
        candidates = sample["generated_captions"]
        if len(candidates) == 0:
            final_caption = ""
            selected_caps = []
        else:
            # Keep only the top-half candidates by audio-text similarity score.
            top_audio = candidates[: min(TOP_AUDIO_KEEP, len(candidates))]
            if PAIRWISE_MODE == "none":
                selected = top_audio[: min(TOP_PAIRWISE_KEEP, len(top_audio))]
            else:
                selected = select_top_pairwise_central_captions(
                    instructor_model, top_audio, TOP_PAIRWISE_KEEP
                )
            selected_caps = [x["text"] for x in selected]
            prompt = build_prompt(selected_caps)

            try:
                final_caption = call_openai_chat_completion(prompt, OPENAI_MODEL)
            except RuntimeError as exc:
                print(f"[warn] sample {i}: OpenAI failed, falling back to top caption: {exc}")
                final_caption = selected_caps[0]

        final_caption = normalize_caption(final_caption)
        csv_out["file_name"].append(sample["audio_file"])
        csv_out["caption_predicted"].append(final_caption)
        details_out.append(
            {
                "idx": sample["idx"],
                "audio_file": sample["audio_file"],
                "n_candidates_total": len(candidates),
                "n_candidates_after_audio_filter": min(TOP_AUDIO_KEEP, len(candidates)),
                "n_candidates_for_llm": len(selected_caps),
                "captions_for_llm": selected_caps,
                "llm_caption": final_caption,
            }
        )

        if not (i + 1) % 10:
            print(f"[info] processed {i + 1} / {len(reranked)} samples")

    pd.DataFrame.from_dict(csv_out).to_csv(
        os.path.join(INFERENCE_DIR, "llm_sap_summary_output.csv"), index=False
    )
    with open(os.path.join(INFERENCE_DIR, "llm_sap_summary_details.json"), "w") as f:
        f.write(json.dumps(details_out, indent=4))
        f.write("\n")
