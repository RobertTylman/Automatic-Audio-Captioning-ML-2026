import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AAC_METRICS_SRC = REPO_ROOT.parent / "aac-metrics" / "src"
STRIP_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_caption(text: str) -> str:
    text = str(text).strip().lower().translate(STRIP_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def import_aac_metrics(aac_metrics_root: Path | None):
    if aac_metrics_root is not None and aac_metrics_root.exists():
        sys.path.insert(0, str(aac_metrics_root))
    try:
        from aac_metrics import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "Could not import aac_metrics. Install it with `pip install aac-metrics` "
            "or pass --aac-metrics-root pointing to the local clone's src folder."
        ) from exc
    return evaluate


def tensor_to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: tensor_to_python(item) for key, item in value.items()}
    return value


def load_predictions(predictions_csv: Path, normalize: bool) -> pd.DataFrame:
    dataframe = pd.read_csv(predictions_csv)
    expected = {"file_name", "caption_predicted"}
    missing = expected.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{predictions_csv} is missing required columns: {sorted(missing)}")
    dataframe = dataframe[["file_name", "caption_predicted"]].copy()
    dataframe["file_name"] = dataframe["file_name"].astype(str)
    if normalize:
        dataframe["caption_predicted"] = dataframe["caption_predicted"].map(normalize_caption)
    return dataframe


def load_references(caption_csv: Path, normalize: bool) -> pd.DataFrame:
    dataframe = pd.read_csv(caption_csv)
    caption_columns = [f"caption_{index}" for index in range(1, 6)]
    expected = {"file_name", *caption_columns}
    missing = expected.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{caption_csv} is missing required columns: {sorted(missing)}")
    dataframe = dataframe[["file_name", *caption_columns]].copy()
    dataframe["file_name"] = dataframe["file_name"].astype(str)
    if normalize:
        for column in caption_columns:
            dataframe[column] = dataframe[column].map(normalize_caption)
    return dataframe


def align_predictions_and_references(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[list[str], list[list[str]], list[str]]:
    merged = predictions.merge(references, on="file_name", how="inner")
    if len(merged) != len(predictions):
        missing = sorted(set(predictions["file_name"]) - set(merged["file_name"]))
        raise ValueError(
            f"{len(missing)} prediction rows have no references. "
            f"First missing files: {missing[:10]}"
        )

    caption_columns = [f"caption_{index}" for index in range(1, 6)]
    candidates = merged["caption_predicted"].astype(str).tolist()
    mult_references = merged[caption_columns].astype(str).values.tolist()
    file_names = merged["file_name"].astype(str).tolist()
    return candidates, mult_references, file_names


def save_sentence_scores(
    output_path: Path,
    file_names: list[str],
    sentence_scores: dict[str, Any],
) -> None:
    rows = {"file_name": file_names}
    for metric_name, values in sentence_scores.items():
        values = tensor_to_python(values)
        if isinstance(values, list) and len(values) == len(file_names):
            rows[metric_name] = values
    pd.DataFrame(rows).to_csv(output_path, index=False)


def parse_metrics(metrics: str) -> str | list[str]:
    if "," not in metrics:
        return metrics
    return [metric.strip() for metric in metrics.split(",") if metric.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=str, required=True)
    parser.add_argument("--references-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metrics", type=str, default="dcase2024")
    parser.add_argument("--device", type=str, default="cuda_if_available")
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--java-path", type=str, default=None)
    parser.add_argument("--tmp-path", type=str, default=None)
    parser.add_argument("--aac-metrics-root", type=str, default=str(DEFAULT_AAC_METRICS_SRC))
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize = not args.no_normalize

    evaluate = import_aac_metrics(
        Path(args.aac_metrics_root).resolve() if args.aac_metrics_root else None
    )
    predictions = load_predictions(Path(args.predictions_csv).resolve(), normalize=normalize)
    references = load_references(Path(args.references_csv).resolve(), normalize=normalize)
    candidates, mult_references, file_names = align_predictions_and_references(
        predictions=predictions,
        references=references,
    )

    metrics = parse_metrics(args.metrics)
    print(f"[score] scoring {len(candidates)} captions with metrics={metrics}", flush=True)
    corpus_scores, sentence_scores = evaluate(
        candidates=candidates,
        mult_references=mult_references,
        metrics=metrics,
        cache_path=args.cache_path,
        java_path=args.java_path,
        tmp_path=args.tmp_path,
        device=args.device,
        verbose=args.verbose,
    )
    corpus_scores = tensor_to_python(corpus_scores)

    corpus_json = output_dir / "corpus_scores.json"
    sentence_csv = output_dir / "sentence_scores.csv"
    with corpus_json.open("w", encoding="utf-8") as handle:
        json.dump(corpus_scores, handle, indent=4, sort_keys=True)
        handle.write("\n")
    save_sentence_scores(sentence_csv, file_names, sentence_scores)

    print(f"[score] corpus scores written to {corpus_json}", flush=True)
    print(f"[score] sentence scores written to {sentence_csv}", flush=True)
    for name, value in corpus_scores.items():
        print(f"{name}: {value}", flush=True)


if __name__ == "__main__":
    main()
