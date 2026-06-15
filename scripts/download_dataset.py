
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config.settings as s

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    s.DATA_DIR.mkdir(parents=True, exist_ok=True)

    out_path = s.DATA_DIR / "financebench_open_source.jsonl"
    if out_path.exists():
        log.info("Dataset already downloaded at %s", out_path)
        return

    log.info("Loading FinanceBench from HuggingFace (%s)…", s.HF_DATASET_ID)
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Run: pip install datasets huggingface-hub")
        raise

    dataset = load_dataset(s.HF_DATASET_ID, split="train")
    log.info("Downloaded %d records", len(dataset))

    # Inspect available fields so we can adapt later scripts if schema differs
    log.info("Fields: %s", list(dataset.features.keys()))

    with open(out_path, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

    log.info("Saved → %s", out_path)

    # Quick breakdown by question type
    qt_counts: dict[str, int] = {}
    for r in dataset:
        qt = r.get("question_type") or r.get("category") or "unknown"
        qt_counts[qt] = qt_counts.get(qt, 0) + 1
    log.info("Question-type breakdown: %s", qt_counts)


if __name__ == "__main__":
    main()