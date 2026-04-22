# src/segment.py
import re
import spacy

nlp = spacy.load("en_core_web_sm")


def segment_into_clauses(text: str) -> list:
    if not text or not text.strip():
        return []

    structural_split_pattern = re.compile(
        r'(?<!\w)'
        r'(?:'
        r'\n\s*\d+\.\s+'
        r'|\n\s*\(\w\)\s+'
        r'|\n\s*[ivxlcdm]+\.\s+'
        r'|\n\s*[•\-\*]\s+'
        r')',
        re.IGNORECASE
    )

    chunks = structural_split_pattern.split(text)
    refined_chunks = []
    for chunk in chunks:
        sub_chunks = re.split(r'\n\s*\n', chunk)
        refined_chunks.extend(sub_chunks)

    raw_sentences = []
    for chunk in refined_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        doc = nlp(chunk)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                raw_sentences.append(sent_text)

    merged = []
    for sent in raw_sentences:
        word_count = len(sent.split())
        if word_count < 8 and merged:
            merged[-1] = merged[-1] + " " + sent
        else:
            merged.append(sent)

    clauses = [c.strip() for c in merged if c.strip()]
    return clauses


def segment_file(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return segment_into_clauses(text)


if __name__ == "__main__":
    sample_text = """
    Welcome to our service. By using this service, you agree to these terms.

    1. Data Collection. We collect your personal data including name, email, and usage patterns.
    We may share this with third parties at our sole discretion.

    2. Arbitration. You waive your right to a jury trial. All disputes will be resolved through
    binding arbitration. You also waive your right to participate in class action lawsuits.

    3. You can delete your account at any time by contacting support.

    (a) We reserve the right to modify these terms at any time without notice.
    (b) Continued use constitutes acceptance of modified terms.
    """
    clauses = segment_into_clauses(sample_text)
    print(f"Found {len(clauses)} clauses:\n")
    for i, clause in enumerate(clauses, 1):
        print(f"[{i}] {clause}\n")