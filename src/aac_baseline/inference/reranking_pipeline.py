import json
import os
import time
from dataclasses import asdict, dataclass
from string import punctuation
from urllib import error, request

import torch
import torch.nn.functional as F
import torchaudio.functional as AF


STRIP_PUNCT_TABLE = str.maketrans("", "", punctuation)

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


@dataclass
class CandidateCaption:
    text: str
    clap_score: float
    normed_nll: float
    hybrid_score: float


@dataclass
class PipelineResult:
    sample_id: str
    nucleus_sampled_captions: list[str]
    candidates: list[CandidateCaption]
    captions_for_llm: list[str]
    final_caption: str


def normalize_caption(text: str) -> str:
    return text.strip().lower().translate(STRIP_PUNCT_TABLE)


def build_prompt(captions: list[str]) -> str:
    captions_block = "\n".join(f"- {caption}" for caption in captions)
    return SAP_PROMPT.replace("{audio captions}", captions_block)


def build_chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def call_openai_chat_completion(
    prompt: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
    max_tokens: int = 80,
    retries: int = 3,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = build_chat_completions_url(
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    )

    last_error = None
    for attempt in range(retries):
        req = request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=120) as response:
                response_json = json.loads(response.read().decode("utf-8"))
                content = response_json["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    return "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    ).strip()
                return str(content).strip()
        except (error.HTTPError, error.URLError, TimeoutError, KeyError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)

    raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_error}")


class ClapSimilarityScorer:
    def __init__(self, model_name: str, device: torch.device) -> None:
        from transformers import AutoTokenizer, ClapFeatureExtractor, ClapModel, ClapProcessor

        self.device = device
        try:
            self.processor = ClapProcessor.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except Exception:
            try:
                feature_extractor = ClapFeatureExtractor.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
            except Exception:
                feature_extractor = ClapFeatureExtractor.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.processor = ClapProcessor(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
            )

        try:
            self.model = ClapModel.from_pretrained(
                model_name,
                local_files_only=True,
                use_safetensors=False,
            ).to(device)
        except Exception:
            self.model = ClapModel.from_pretrained(
                model_name,
                use_safetensors=False,
            ).to(device)
        self.model.eval()

    @staticmethod
    def _feature_tensor(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if hasattr(output, "audio_embeds") and output.audio_embeds is not None:
            return output.audio_embeds
        raise TypeError(f"Unsupported CLAP feature output type: {type(output)!r}")

    @torch.no_grad()
    def score(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        captions: list[str],
    ) -> torch.Tensor:
        if not captions:
            return torch.empty(0, device=self.device)

        target_sample_rate = getattr(
            self.processor.feature_extractor,
            "sampling_rate",
            sample_rate,
        )
        if sample_rate != target_sample_rate:
            waveform = AF.resample(
                waveform.detach().cpu(),
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
            )
            sample_rate = target_sample_rate

        audio_array = waveform.detach().cpu().numpy()
        try:
            audio_inputs = self.processor(
                audio=audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
        except TypeError:
            audio_inputs = self.processor(
                audios=audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
        text_inputs = self.processor(
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        audio_inputs = {key: value.to(self.device) for key, value in audio_inputs.items()}
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        audio_features = self._feature_tensor(
            self.model.get_audio_features(**audio_inputs)
        )
        text_features = self._feature_tensor(
            self.model.get_text_features(**text_inputs)
        )
        audio_features = F.normalize(audio_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return torch.matmul(text_features, audio_features.squeeze(0)).detach().cpu()

    @torch.no_grad()
    def text_embeddings(self, captions: list[str]) -> torch.Tensor | None:
        if not captions:
            return None

        text_inputs = self.processor(
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        text_features = self._feature_tensor(
            self.model.get_text_features(**text_inputs)
        )
        return F.normalize(text_features, dim=-1).detach().cpu()


class CaptionRerankingPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        clap_scorer: ClapSimilarityScorer,
        device: torch.device,
        generation_config: dict | None = None,
        decoder_weight: float = 0.3,
        clap_weight: float = 0.7,
        top_audio_keep: int = 32,
        top_pairwise_keep: int = 12,
        openai_model: str = "gpt-4.1-mini",
        use_llm: bool = True,
        show_sampled_captions: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.clap_scorer = clap_scorer
        self.device = device
        self.generation_config = {
            "do_sample": True,
            "temperature": 0.5,
            "top_p": 0.95,
            "num_return_sequences": 64,
            "min_length": 5,
            "max_length": 30,
            "no_repeat_ngram_size": 3,
        }
        self.generation_config.update(generation_config or {})
        self.decoder_weight = decoder_weight
        self.clap_weight = clap_weight
        self.top_audio_keep = top_audio_keep
        self.top_pairwise_keep = top_pairwise_keep
        self.openai_model = openai_model
        self.use_llm = use_llm
        self.show_sampled_captions = show_sampled_captions

    @torch.no_grad()
    def generate_candidates(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
    ) -> list[str]:
        token_ids = self.model.generate(
            waveforms=waveforms.to(self.device),
            waveform_lengths=waveform_lengths.to(self.device),
            sample_rates=sample_rates.to(self.device),
            **self.generation_config,
        )
        captions = self.tokenizer.batch_decode(
            token_ids.detach().cpu(),
            skip_special_tokens=True,
        )
        return [normalize_caption(caption) for caption in captions]

    @torch.no_grad()
    def score_decoder_nll(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
        sample_rate: torch.Tensor,
        captions: list[str],
    ) -> torch.Tensor:
        tokenized = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.generation_config["max_length"],
            return_tensors="pt",
        )
        labels = tokenized["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return self.model.score_captions(
            waveforms=waveform.unsqueeze(0).to(self.device),
            waveform_lengths=waveform_length.unsqueeze(0).to(self.device),
            sample_rates=sample_rate.unsqueeze(0).to(self.device),
            labels=labels.to(self.device),
        ).detach().cpu()

    def select_pairwise_central(
        self,
        candidates: list[CandidateCaption],
    ) -> list[CandidateCaption]:
        if len(candidates) <= self.top_pairwise_keep:
            return candidates

        embeddings = self.clap_scorer.text_embeddings(
            [candidate.text for candidate in candidates]
        )
        if embeddings is None:
            return candidates[: self.top_pairwise_keep]

        pairwise = torch.matmul(embeddings, embeddings.T)
        pairwise.fill_diagonal_(0.0)
        centrality = pairwise.sum(dim=1) / max(1, len(candidates) - 1)
        keep_indices = torch.argsort(centrality, descending=True)[: self.top_pairwise_keep]
        return [candidates[int(index)] for index in keep_indices]

    def rerank(
        self,
        captions: list[str],
        clap_scores: torch.Tensor,
        decoder_nlls: torch.Tensor,
    ) -> list[CandidateCaption]:
        deduped: dict[str, CandidateCaption] = {}
        for caption, clap_score, decoder_nll in zip(captions, clap_scores, decoder_nlls):
            hybrid_score = (
                self.clap_weight * float(clap_score)
                - self.decoder_weight * float(decoder_nll)
            )
            candidate = CandidateCaption(
                text=caption,
                clap_score=float(clap_score),
                normed_nll=float(decoder_nll),
                hybrid_score=hybrid_score,
            )
            existing = deduped.get(caption)
            if existing is None or candidate.hybrid_score > existing.hybrid_score:
                deduped[caption] = candidate

        return sorted(deduped.values(), key=lambda item: item.hybrid_score, reverse=True)

    def summarize(self, captions: list[str]) -> str:
        if not captions:
            return ""
        if not self.use_llm:
            return normalize_caption(captions[0])

        prompt = build_prompt(captions)
        try:
            return normalize_caption(
                call_openai_chat_completion(prompt=prompt, model=self.openai_model)
            )
        except RuntimeError:
            return normalize_caption(captions[0])

    def process_batch(self, batch: dict) -> list[PipelineResult]:
        waveforms = batch["waveforms"]
        waveform_lengths = batch["waveform_lengths"]
        sample_rates = batch["sample_rates"]
        batch_candidates = self.generate_candidates(
            waveforms=waveforms,
            waveform_lengths=waveform_lengths,
            sample_rates=sample_rates,
        )
        num_return_sequences = self.generation_config["num_return_sequences"]
        results = []

        for index, sample_id in enumerate(batch["sample_ids"]):
            start = index * num_return_sequences
            end = start + num_return_sequences
            captions = batch_candidates[start:end]
            waveform = waveforms[index, : int(waveform_lengths[index].item())]
            clap_scores = self.clap_scorer.score(
                waveform=waveform,
                sample_rate=int(sample_rates[index].item()),
                captions=captions,
            )
            decoder_nlls = self.score_decoder_nll(
                waveform=waveform,
                waveform_length=waveform_lengths[index],
                sample_rate=sample_rates[index],
                captions=captions,
            )
            candidates = self.rerank(
                captions=captions,
                clap_scores=clap_scores,
                decoder_nlls=decoder_nlls,
            )
            top_audio = candidates[: min(self.top_audio_keep, len(candidates))]
            selected = self.select_pairwise_central(top_audio)
            captions_for_llm = [candidate.text for candidate in selected]
            if self.show_sampled_captions:
                print(
                    f"[nucleus] {sample_id}: {len(captions)} sampled captions",
                    flush=True,
                )
                for caption_index, caption in enumerate(captions, start=1):
                    print(f"[nucleus]   {caption_index:02d}. {caption}", flush=True)
            results.append(
                PipelineResult(
                    sample_id=sample_id,
                    nucleus_sampled_captions=captions,
                    candidates=candidates,
                    captions_for_llm=captions_for_llm,
                    final_caption=self.summarize(captions_for_llm),
                )
            )

        return results

    @staticmethod
    def result_to_dict(result: PipelineResult) -> dict:
        output = asdict(result)
        output["generated_captions"] = output.pop("candidates")
        output["audio_file"] = output.pop("sample_id")
        return output
