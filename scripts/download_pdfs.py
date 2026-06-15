
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config.settings as s

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (research project; contact: your@email.com)"}
TIMEOUT = 90   # seconds
DELAY   = 0.5  # polite delay between requests

_GITHUB_RAW_PDF = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf"
)

def _unique_docs(jsonl_path: Path) -> dict[str, str]:
    """Return {doc_name: doc_link} for every unique doc in a JSONL file."""
    seen: dict[str, str] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            doc_name = r.get("doc_name", "")
            doc_link = r.get("doc_link", "") or r.get("doc_url", "")
            if doc_name and doc_link and doc_name not in seen:
                seen[doc_name] = doc_link
    return seen


def _fetch_doc_catalogue() -> dict[str, str]:
    """Return {doc_name: doc_link} from the local financebench_document_information.jsonl."""
    catalogue_path = s.DATA_DIR / "financebench_document_information.jsonl"
    log.info("Loading document catalogue from %s", catalogue_path)
    return _unique_docs(catalogue_path)


def _download(doc_name: str, primary_url: str, dest_dir: Path) -> bool:
    """Try primary URL then GitHub raw mirror. Skips if dest already exists."""
    dest = dest_dir / f"{doc_name}.pdf"
    if dest.exists():
        log.debug("Already downloaded: %s", doc_name)
        return True

    for label, url in [("primary", primary_url), ("github", _GITHUB_RAW_PDF.format(doc_name=doc_name))]:
        if not url:
            continue
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
            content = b"".join(resp.iter_content(chunk_size=16_384))
            if len(content) < 1_024 or not content[:4] == b"%PDF":
                log.debug("  not a valid PDF from %s: %s", label, url)
                continue
            dest.write_bytes(content)
            size_mb = dest.stat().st_size / 1_048_576
            log.info("  ✓ %-10s %s  (%.1f MB)", f"[{label}]", doc_name, size_mb)
            return True
        except Exception as exc:
            log.debug("  %s failed for %s: %s", label, doc_name, exc)
            if dest.exists():
                dest.unlink()

    log.error("  ✗ %s  — all sources failed", doc_name)
    return False


def _run_batch(docs: dict[str, str], desc: str) -> list[str]:
    """Download a batch of docs, return list of failed doc_names."""
    failures: list[str] = []
    for doc_name, url in tqdm(docs.items(), desc=desc):
        ok = _download(doc_name, url, s.PDF_DIR)
        if not ok:
            failures.append(doc_name)
        time.sleep(DELAY)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FinanceBench PDFs (idempotent)")
    parser.add_argument(
        "--extra",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Also download N non-eval distractor documents from the broader "
            "financebench_document_information catalogue (default: 0)"
        ),
    )
    args = parser.parse_args()

    jsonl_path = s.DATA_DIR / "financebench_open_source.jsonl"
    if not jsonl_path.exists():
        log.error("Dataset JSONL not found at %s", jsonl_path)
        sys.exit(1)

    s.PDF_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: eval documents (84 unique docs) ─────────────────────────────
    eval_docs = _unique_docs(jsonl_path)
    log.info("Eval documents to download: %d", len(eval_docs))
    failures = _run_batch(eval_docs, desc="Eval PDFs")

    # ── Phase 2: non-eval distractor documents ───────────────────────────────
    if args.extra > 0:
        all_docs = _fetch_doc_catalogue()
        candidates = {
            dn: url
            for dn, url in all_docs.items()
            if dn not in eval_docs and not (s.PDF_DIR / f"{dn}.pdf").exists()
        }
        log.info(
            "Non-eval candidates available: %d  (requesting %d)",
            len(candidates),
            args.extra,
        )
        extra_docs = dict(list(candidates.items())[: args.extra])
        extra_failures = _run_batch(extra_docs, desc=f"Extra PDFs ({args.extra})")
        failures.extend(extra_failures)

    # ── Failure log ──────────────────────────────────────────────────────────
    log.info(
        "Downloaded %d total PDFs  |  %d failures",
        len(list(s.PDF_DIR.glob("*.pdf"))),
        len(failures),
    )
    fail_path = s.DATA_DIR / "download_failures.txt"
    if failures:
        fail_path.write_text("\n".join(sorted(set(failures))))
        log.warning("%d failure(s) logged → %s", len(failures), fail_path)
    else:
        if fail_path.exists():
            fail_path.unlink()
        log.info("All PDFs downloaded successfully.")


if __name__ == "__main__":
    main()