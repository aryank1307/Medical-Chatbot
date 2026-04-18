import argparse
import re
import tempfile
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests


PDF_RE = re.compile(r"https?://[^\s\"'<>]+?\.pdf(?:\?[^\s\"'<>]*)?", re.IGNORECASE)
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


def same_domain(url: str, domain: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == domain or host.endswith("." + domain)


def discover_pdf_links(
    seeds: list[str], allowed_domains: list[str], max_pages: int = 200
) -> list[str]:
    queue = deque(seeds)
    visited: set[str] = set()
    pdfs: set[str] = set()

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            text = resp.text
        except requests.RequestException:
            continue

        for match in PDF_RE.findall(text):
            pdfs.add(match)

        for href in HREF_RE.findall(text):
            nxt = urljoin(url, href)
            if not nxt.startswith("http"):
                continue
            if any(same_domain(nxt, d) for d in allowed_domains):
                if nxt not in visited and len(queue) < max_pages * 4:
                    queue.append(nxt)

    return sorted(pdfs)


def upload_pdf_from_url(
    url: str, api_url: str, namespace: str, specialty: str
) -> tuple[bool, str]:
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                tmp_path = Path(tf.name)
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tf.write(chunk)

        with tmp_path.open("rb") as f:
            files = {"files": (tmp_path.name, f, "application/pdf")}
            data = {"namespace": namespace, "specialty": specialty}
            upload = requests.post(
                f"{api_url}/upload_pdfs/",
                files=files,
                data=data,
                timeout=300,
            )
        tmp_path.unlink(missing_ok=True)
        if upload.status_code == 200:
            return True, upload.text
        return False, f"[{upload.status_code}] {upload.text[:200]}"
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover public PDF links and ingest into RAG backend."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--namespace", default="medical-guidelines")
    parser.add_argument("--specialty", default="general")
    parser.add_argument("--max-pages", type=int, default=160)
    parser.add_argument("--max-pdfs", type=int, default=30)
    args = parser.parse_args()

    seeds = [
        "https://www.who.int/publications",
        "https://iris.who.int",
        "https://www.nih.gov/health-information",
        "https://medlineplus.gov/healthtopics.html",
        "https://www.cdc.gov/diseasesconditions",
    ]
    domains = ["who.int", "iris.who.int", "nih.gov", "medlineplus.gov", "cdc.gov"]

    print("Discovering public PDF links...")
    pdf_links = discover_pdf_links(seeds, domains, max_pages=args.max_pages)
    print(f"Discovered {len(pdf_links)} candidate PDF links.")

    ok = 0
    fail = 0
    for link in pdf_links[: args.max_pdfs]:
        success, msg = upload_pdf_from_url(
            link,
            api_url=args.api_url,
            namespace=args.namespace,
            specialty=args.specialty,
        )
        if success:
            ok += 1
            print(f"OK {ok}: {link}")
        else:
            fail += 1
            print(f"FAIL {fail}: {link} -> {msg}")

    print(
        f"Done. uploaded_ok={ok}, uploaded_fail={fail}, attempted={min(len(pdf_links), args.max_pdfs)}"
    )


if __name__ == "__main__":
    main()
