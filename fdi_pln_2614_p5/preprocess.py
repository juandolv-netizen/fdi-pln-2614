# preprocess.py
import os
import re
from pathlib import Path


def clean_gutenberg_markers(text):
    start_match = re.search(r"\*\*\* START OF.+?\*\*\*", text)
    end_match = re.search(r"\*\*\* END OF.+?\*\*\*", text)

    if start_match:
        text = text[start_match.end() :]
    if end_match:
        text = text[: end_match.start()]
    return text


def normalize_text(text):
    text = re.sub(r"[«»“”]", '"', text)
    text = re.sub(r"[—–]", "-", text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,;:\'"!?()-]', "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def run_preprocessing(
    input_dir="resources", output_dir="resources_clean", exclude_files=None
):
    if exclude_files is None:
        exclude_files = set()

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for file_path in in_path.glob("*.txt"):
        if file_path.name in exclude_files:
            continue

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        text = clean_gutenberg_markers(raw_text)
        text = normalize_text(text)
        text += "\n\n<|endofdoc|>\n\n"

        with open(out_path / file_path.name, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    run_preprocessing(exclude_files={"Natural_Language_Processing_with_Python.txt"})
