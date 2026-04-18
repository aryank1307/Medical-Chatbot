import argparse
import json
import mimetypes
import tempfile
from pathlib import Path

import requests


def safe_print(message: str) -> None:
    try:
        print(message)
    except OSError:
        # Some Windows shells can throw intermittent stdout errors.
        pass


def load_catalog(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    urls: list[str] = []
    for entry in payload.get("sources", []):
        urls.extend(entry.get("urls", []))
    seen = set()
    deduped = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def looks_like_pdf(url: str, content_type: str) -> bool:
    if url.lower().endswith(".pdf"):
        return True
    if "application/pdf" in content_type.lower():
        return True
    return False


def upload_pdf(pdf_path: Path, api_url: str) -> tuple[int, str]:
    with pdf_path.open("rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        response = requests.post(f"{api_url}/upload_pdfs/", files=files, timeout=180)
    return response.status_code, response.text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download direct PDF links from a source catalog and ingest into backend."
    )
    parser.add_argument(
        "--catalog",
        default="server/knowledge_sources.json",
        help="Path to source catalog JSON.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Backend API base URL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max number of direct PDF URLs to ingest in one run.",
    )
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    urls = load_catalog(catalog_path)
    safe_print(f"Loaded {len(urls)} URLs from catalog.")

    ingested = 0
    skipped = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        for url in urls:
            if ingested >= args.limit:
                break

            try:
                head = requests.head(url, allow_redirects=True, timeout=30)
                content_type = head.headers.get("Content-Type", "")
            except requests.RequestException:
                content_type = ""

            if not looks_like_pdf(url, content_type):
                skipped += 1
                safe_print(f"SKIP (not direct PDF): {url}")
                continue

            try:
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                filename = (
                    Path(url.split("?")[0]).name
                    or f"source_{ingested + 1}.pdf"
                )
                if not filename.lower().endswith(".pdf"):
                    filename = f"{filename}.pdf"
                pdf_path = temp_dir / filename
                with pdf_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if mimetypes.guess_type(str(pdf_path))[0] != "application/pdf":
                    # Keep soft validation; many valid PDFs still pass with .pdf extension.
                    pass

                status, text = upload_pdf(pdf_path, args.api_url)
                if status == 200:
                    ingested += 1
                    safe_print(f"OK  ({ingested}/{args.limit}) {url}")
                else:
                    failed += 1
                    safe_print(f"FAIL upload [{status}] {url} -> {text[:180]}")

            except Exception as exc:
                failed += 1
                safe_print(f"FAIL fetch {url} -> {exc}")

    safe_print(
        f"Done. ingested={ingested}, skipped={skipped}, failed={failed}, "
        f"limit={args.limit}"
    )
    safe_print(
        "Note: Most links in this catalog are landing pages. "
        "For best results, replace them with direct PDF URLs."
    )


if __name__ == "__main__":
    main()
