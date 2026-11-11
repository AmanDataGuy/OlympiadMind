"""
structured_parsers.py
---------------------
Structured Format Parsers

Parses non-standard document formats into structured JSON:

  - Science Bowl MCQ: PHYS-91; Multiple Choice → question + w/x/y/z options + answer
  - CBSE Assertion-Reason: Assertion(A) + Reason(R) + A/B/C/D answer codes
  - NIST Constants Table: fixed-width columns → Quantity/Value/Uncertainty/Unit

Output: data/extracted/structured/{doc_id}_structured.json
"""

import json
import re
import fitz
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

INVENTORY_PATH      = Path("data/manifests/inventory.csv")
OUTPUT_DIR          = Path("data/extracted/structured")

SOURCE_ROOTS = {
    "science_bowl": Path("data/raw/science_bowl"),
    "cbse":         Path("data/raw/cbse"),
    "tabular":      Path("data/raw/tabular"),
}


# ── science bowl mcq parser ───────────────────────────────────────────────────

# matches: PHYS-91; Multiple Choice:
QUESTION_HEADER = re.compile(
    r"(PHYS-\d+;\s+(?:Multiple Choice|Short Answer)[:\s]+)"
)

# matches answer line: ANSWER:   X -- THE RAILROAD CAR IS...
ANSWER_LINE = re.compile(
    r"ANSWER:\s+([WXYZ])\s*--\s*(.+?)(?=\n|$)",
    re.IGNORECASE
)

# matches options: w) ... x) ... y) ... z) ...
OPTION_PATTERN = re.compile(
    r"([wxyz])\)\s+(.+?)(?=[wxyz]\)|ANSWER:|$)",
    re.IGNORECASE | re.DOTALL
)


def parse_science_bowl(pdf_path: Path) -> list:
    """
    Parse Science Bowl MCQ PDF into structured question records.

    Each record contains:
    - question_id: PHYS-91 style identifier
    - question_type: Multiple Choice or Short Answer
    - question_text: the problem statement
    - options: dict of w/x/y/z → option text
    - correct_answer: letter of correct option
    - answer_text: full answer explanation
    """
    doc       = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    questions = []

    # split on question headers
    parts = QUESTION_HEADER.split(full_text)

    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body   = parts[i + 1] if i + 1 < len(parts) else ""

        # parse question id and type from header
        id_match   = re.search(r"(PHYS-\d+)", header)
        type_match = re.search(r"(Multiple Choice|Short Answer)", header, re.IGNORECASE)

        question_id   = id_match.group(1)   if id_match   else "UNKNOWN"
        question_type = type_match.group(1) if type_match else "Unknown"

        # extract answer line first — everything before it is question + options
        answer_match = ANSWER_LINE.search(body)

        if answer_match:
            question_block = body[:answer_match.start()]
            correct_answer = answer_match.group(1).upper()
            answer_text    = answer_match.group(2).strip()
        else:
            question_block = body
            correct_answer = ""
            answer_text    = ""

        # extract options
        options = {}
        for opt_match in OPTION_PATTERN.finditer(question_block):
            letter = opt_match.group(1).lower()
            text   = opt_match.group(2).strip().replace("\n", " ")
            options[letter] = text

        # question text is everything before first option
        first_option_pos = OPTION_PATTERN.search(question_block)
        if first_option_pos:
            question_text = question_block[:first_option_pos.start()].strip()
        else:
            question_text = question_block.strip()

        question_text = question_text.replace("\n", " ").strip()

        if question_text:
            questions.append({
                "question_id":    question_id,
                "question_type":  question_type,
                "question_text":  question_text,
                "options":        options,
                "correct_answer": correct_answer,
                "answer_text":    answer_text,
            })

        i += 2

    return questions


# ── cbse assertion-reason parser ──────────────────────────────────────────────

# matches: Assertion(A): ... Reason(R): ...
ASSERTION_PATTERN = re.compile(
    r"Assertion\s*\(A\)\s*:\s*(.+?)(?=Reason\s*\(R\)|$)",
    re.IGNORECASE | re.DOTALL
)
REASON_PATTERN = re.compile(
    r"Reason\s*\(R\)\s*:\s*(.+?)(?=\n\d+\s|Assertion|$)",
    re.IGNORECASE | re.DOTALL
)

# CBSE answer codes:
# A — both assertion and reason true, reason is correct explanation
# B — both true but reason is not correct explanation
# C — assertion true, reason false
# D — assertion false, reason true
ANSWER_CODE_MAP = {
    "A": "Both A and R true, R is correct explanation",
    "B": "Both A and R true, R is not correct explanation",
    "C": "A is true, R is false",
    "D": "A is false, R is true",
}


def parse_cbse_assertion_reason(pdf_path: Path) -> list:
    """
    Parse CBSE Assertion-Reason PDF into structured question records.

    Each record contains:
    - sl_no: question serial number
    - chapter: chapter name
    - assertion: assertion statement
    - reason: reason statement
    - answer_code: A/B/C/D
    - answer_meaning: plain English meaning of the code
    """
    doc       = fitz.open(pdf_path)
    questions = []
    chapter   = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()

        # detect chapter heading
        chapter_match = re.search(r"CHAPTER[–\-]\d+:\s*(.+?)(?:\n|SYLLABUS)", text)
        if chapter_match:
            chapter = chapter_match.group(1).strip()

        # find all assertion blocks on this page
        assertion_matches = list(ASSERTION_PATTERN.finditer(text))

        for i, a_match in enumerate(assertion_matches):
            assertion_text = a_match.group(1).strip().replace("\n", " ")

            # find corresponding reason — starts after assertion ends
            reason_search = REASON_PATTERN.search(text, a_match.end())
            reason_text   = reason_search.group(1).strip().replace("\n", " ") if reason_search else ""

            # look for answer code — single letter A/B/C/D after the reason
            answer_search = re.search(r"\b([ABCD])\b", text[a_match.start():a_match.start() + 500])
            answer_code   = answer_search.group(1) if answer_search else ""

            if assertion_text:
                questions.append({
                    "chapter":        chapter,
                    "page":           page_num,
                    "assertion":      assertion_text,
                    "reason":         reason_text,
                    "answer_code":    answer_code,
                    "answer_meaning": ANSWER_CODE_MAP.get(answer_code, ""),
                })

    doc.close()
    return questions


# ── nist constants table parser ───────────────────────────────────────────────

def parse_nist_table(pdf_path: Path) -> list:
    """
    Parse NIST physical constants table into structured records.

    Each record contains:
    - quantity: name of the physical constant
    - value: numerical value
    - uncertainty: uncertainty in the last digits
    - unit: SI unit string
    """
    doc     = fitz.open(pdf_path)
    records = []

    for page in doc:
        # use "words" extraction — better for tabular data
        words = page.get_text("words", sort=True)

        if not words:
            continue

        # group words by y-coordinate (same line = same row)
        lines = {}
        for word in words:
            x0, y0, x1, y1, text, block, line, word_idx = word
            y_key = round(y0 / 5) * 5  # group within 5px tolerance
            if y_key not in lines:
                lines[y_key] = []
            lines[y_key].append((x0, text))

        # process each line
        for y_key in sorted(lines.keys()):
            line_words = sorted(lines[y_key], key=lambda w: w[0])
            line_text  = " ".join(w[1] for w in line_words)

            # skip headers and empty lines
            if len(line_text) < 5:
                continue
            if any(h in line_text for h in ["Quantity", "Value", "Uncertainty", "Unit"]):
                continue

            # try to parse: quantity | value | uncertainty | unit
            # value and uncertainty are numeric, unit contains letters
            tokens = line_text.split()
            if len(tokens) < 3:
                continue

            # find where numbers start (value column)
            num_start = None
            for j, token in enumerate(tokens):
                if re.match(r"[\d\-\+]", token):
                    num_start = j
                    break

            if num_start is None or num_start == 0:
                continue

            quantity    = " ".join(tokens[:num_start])
            numeric_part = tokens[num_start:]

            # last token(s) that aren't numbers are the unit
            unit_tokens = []
            value_tokens = []
            for token in reversed(numeric_part):
                if re.match(r"[a-zA-Z]", token) and not re.match(r"\d", token):
                    unit_tokens.insert(0, token)
                else:
                    break

            value_end    = len(numeric_part) - len(unit_tokens)
            value_tokens = numeric_part[:value_end]
            unit         = " ".join(unit_tokens)

            # value is first numeric token, uncertainty is second if present
            value       = value_tokens[0] if value_tokens else ""
            uncertainty = value_tokens[1] if len(value_tokens) > 1 else "exact"

            if quantity and value:
                records.append({
                    "quantity":    quantity,
                    "value":       value,
                    "uncertainty": uncertainty,
                    "unit":        unit,
                })

    doc.close()
    return records


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INVENTORY_PATH)
    df = df[df["source"].isin(SOURCE_ROOTS.keys())].reset_index(drop=True)

    print(f"Structured format documents: {len(df)}\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing structured docs"):
        doc_id   = row["filename"].replace(".pdf", "")
        source   = row["source"]
        pdf_path = SOURCE_ROOTS[source] / row["filename"]

        if not pdf_path.exists():
            continue

        try:
            if source == "science_bowl":
                records = parse_science_bowl(pdf_path)
                output  = {
                    "doc_id":         doc_id,
                    "format":         "mcq_wxyz",
                    "question_count": len(records),
                    "questions":      records,
                }

            elif source == "cbse":
                records = parse_cbse_assertion_reason(pdf_path)
                output  = {
                    "doc_id":         doc_id,
                    "format":         "assertion_reason",
                    "question_count": len(records),
                    "questions":      records,
                }

            elif source == "tabular":
                records = parse_nist_table(pdf_path)
                output  = {
                    "doc_id":       doc_id,
                    "format":       "constants_table",
                    "record_count": len(records),
                    "constants":    records,
                }

            else:
                continue

        except Exception as e:
            print(f"\n  ERROR — {doc_id}: {e}")
            continue

        out_path = OUTPUT_DIR / f"{doc_id}_structured.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        count = output.get("question_count") or output.get("record_count", 0)
        print(f"  {doc_id}: {count} records parsed")

    print(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()