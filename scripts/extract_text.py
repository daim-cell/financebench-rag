
import logging
import sys
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)
import json
import fitz  # PyMuPDF
import config.settings as s


def extract_pdf(pdf_path: Path) -> dict:
    """
    Extract text from every page of a PDF in reading order.

    Returns:
        {
            "doc_name": str,
            "pages": [{"page_num": int, "text": str}, ...]
        }
        Only non-blank pages are included.
    """
    doc_name = pdf_path.stem
    pages: list[dict] = []

    doc = fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc, start=1):
            # sort=True → top-to-bottom, left-to-right reading order
            text = page.get_text("text", sort=True).strip()
            if text:
                pages.append({"page_num": i, "text": text})
    finally:
        doc.close()

    return {"doc_name": doc_name, "pages": pages}


def extract_and_cache(pdf_path: Path, processed_dir: Path = s.PROCESSED_DIR) -> Path:
    """
    Extract a PDF and write the result to processed_dir/{doc_name}.json.

    Idempotent: if the output file already exists the function returns
    immediately without re-processing.

    Returns:
        Path to the output JSON file.
    """
    doc_name = pdf_path.stem
    out_path = processed_dir / f"{doc_name}.json"

    if out_path.exists():
        log.debug("Already extracted: %s", doc_name)
        return out_path

    log.info("Extracting: %s", doc_name)
    data = extract_pdf(pdf_path)

    total_chars = sum(len(p["text"]) for p in data["pages"])
    log.info(
        "  %s — %d pages, %s chars",
        doc_name, len(data["pages"]), f"{total_chars:,}"
    )

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return out_path


def load_processed_docs(processed_dir: Path = s.PROCESSED_DIR) -> list[dict]:
    """
    Load all cached extracted documents from disk.

    Returns a list of dicts matching the schema:
        {doc_name: str, pages: [{page_num: int, text: str}]}
    """
    docs = []
    for json_path in sorted(processed_dir.glob("*.json")):
        docs.append(json.loads(json_path.read_text(encoding="utf-8")))
    return docs

def main() -> None:
    if not s.PDF_DIR.exists() or not any(s.PDF_DIR.glob("*.pdf")):
        log.error("No PDFs found in %s. Run download_pdfs.py first.", s.PDF_DIR)
        sys.exit(1)

    s.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(s.PDF_DIR.glob("*.pdf"))
    log.info("Found %d PDFs to process", len(pdfs))

    ok, failures = 0, []
    for pdf_path in tqdm(pdfs, desc="Extracting text"):
        try:
            extract_and_cache(pdf_path, s.PROCESSED_DIR)
            ok += 1
        except Exception as exc:
            log.error("Failed on %s: %s", pdf_path.name, exc)
            failures.append(pdf_path.name)

    log.info("Extracted %d / %d documents", ok, len(pdfs))
    if failures:
        log.warning("Failed: %s", failures)


if __name__ == "__main__":
    main()