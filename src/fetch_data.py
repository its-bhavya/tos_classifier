# src/fetch_data.py
import requests
from bs4 import BeautifulSoup
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.segment import segment_into_clauses


def fetch_and_segment(url: str) -> tuple:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Could not fetch URL: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["nav", "footer", "header", "script", "style",
                     "aside", "form", "button", "iframe", "noscript"]):
        tag.decompose()

    boilerplate_patterns = ["nav", "navbar", "footer", "header", "sidebar",
                            "cookie", "banner", "popup", "menu", "breadcrumb"]
    for pattern in boilerplate_patterns:
        for tag in soup.find_all(class_=lambda c: c and pattern in c.lower()):
            tag.decompose()
        for tag in soup.find_all(id=lambda i: i and pattern in i.lower()):
            tag.decompose()

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(id="main-content")
        or soup.find(class_="content")
        or soup.find("body")
    )

    raw_text = main_content.get_text(separator="\n") if main_content else soup.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines()]
    cleaned_text = "\n".join(line for line in lines if line)

    clauses = segment_into_clauses(cleaned_text)
    return clauses, cleaned_text


def save_segmented_tos(url: str, output_name: str, save_dir: str = "data/raw/sample_tos") -> list:
    os.makedirs(save_dir, exist_ok=True)
    clauses, _ = fetch_and_segment(url)

    output_path = os.path.join(save_dir, f"{output_name}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {url}\n")
        f.write(f"Total clauses: {len(clauses)}\n")
        f.write("=" * 60 + "\n\n")
        for i, clause in enumerate(clauses, 1):
            f.write(f"[{i}] {clause}\n\n")

    print(f"Saved {len(clauses)} clauses to {output_path}")
    return clauses


if __name__ == "__main__":
    test_docs = [
        ("https://policies.google.com/terms", "google_tos"),
        ("https://www.spotify.com/us/legal/end-user-agreement/", "spotify_tos"),
        ("https://www.anthropic.com/legal/consumer-terms", "anthropic_tos"),
    ]
    for url, name in test_docs:
        print(f"\nFetching: {url}")
        try:
            clauses = save_segmented_tos(url, name)
            print(f"First 3 clauses:")
            for c in clauses[:3]:
                print(f"  → {c[:100]}...")
        except Exception as e:
            print(f"  ERROR: {e}")