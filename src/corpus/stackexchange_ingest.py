"""
stackexchange_ingest.py
-----------------------
Step 3.2 — Physics Stack Exchange Ingestion

Parses the Physics Stack Exchange XML dump and extracts high-quality
question-answer pairs for the knowledge corpus.

Download Posts.xml from:
  https://archive.org/download/stackexchange/physics.stackexchange.com.7z

Filter criteria:
  - Questions with score >= 5
  - Accepted answers with score >= 3
  - Tags matching physics topics relevant to olympiad content

Output: data/corpus/stackexchange_chunks.jsonl
"""

import json
import re
import html
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

# update this path to where you extracted the dump
POSTS_XML_PATH = Path(r"C:\Users\AMAND\Downloads\physics.stackexchange.com\Posts.xml")

OUTPUT_FILE    = Path("data/corpus/stackexchange_chunks.jsonl")

MIN_QUESTION_SCORE = 5
MIN_ANSWER_SCORE   = 3

# only keep posts with these tags
RELEVANT_TAGS = {
    "mechanics", "electromagnetism", "thermodynamics", "optics",
    "quantum-mechanics", "special-relativity", "waves", "fluid-dynamics",
    "classical-mechanics", "electromagnetic-radiation", "electric-fields",
    "magnetic-fields", "energy", "momentum", "forces", "gravity",
    "oscillators", "harmonic-oscillator", "rotational-dynamics",
    "circuit", "capacitance", "inductance", "interference", "diffraction",
    "lagrangian-formalism", "newtonian-mechanics", "kinematics",
    "angular-momentum", "torque", "potential-energy", "kinetic-energy",
    "electric-circuits", "magnetic-moment", "optics", "acoustics",
}


# ── html stripping ────────────────────────────────────────────────────────────

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MATH_PATTERN     = re.compile(r"\$\$?.+?\$\$?", re.DOTALL)


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities, preserve math."""
    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_relevant_tag(tags_str: str) -> bool:
    """Check if post tags overlap with our relevant tag set."""
    if not tags_str:
        return False
    post_tags = set(re.findall(r"\|([^|]+)\|", tags_str))
    return bool(post_tags & RELEVANT_TAGS)


# ── parser ────────────────────────────────────────────────────────────────────

def parse_posts(posts_xml_path: Path) -> list:
    """
    Parse Stack Exchange Posts.xml and extract QA pairs.

    Two-pass approach:
    Pass 1 — collect all questions meeting score threshold
    Pass 2 — collect accepted answers for those questions
    """
    print("Pass 1: collecting questions...")
    questions = {}

    # use iterparse for memory efficiency on large XML
    for event, elem in tqdm(ET.iterparse(str(posts_xml_path), events=["end"])):
        if elem.tag != "row":
            elem.clear()
            continue

        post_type = elem.get("PostTypeId")
        score     = int(elem.get("Score", 0))
        tags      = elem.get("Tags", "")

        if post_type == "1" and score >= MIN_QUESTION_SCORE and has_relevant_tag(tags):
            questions[elem.get("Id")] = {
                "title":      elem.get("Title", ""),
                "body":       strip_html(elem.get("Body", "")),
                "tags":       tags,
                "score":      score,
                "accepted_id": elem.get("AcceptedAnswerId", ""),
            }

        elem.clear()

    print(f"  Found {len(questions)} relevant questions")

    print("Pass 2: collecting accepted answers...")
    qa_pairs = []

    for event, elem in tqdm(ET.iterparse(str(posts_xml_path), events=["end"])):
        if elem.tag != "row":
            elem.clear()
            continue

        post_type = elem.get("PostTypeId")
        score     = int(elem.get("Score", 0))
        parent_id = elem.get("ParentId", "")

        if post_type == "2" and score >= MIN_ANSWER_SCORE and parent_id in questions:
            q   = questions[parent_id]
            ans = strip_html(elem.get("Body", ""))

            if len(ans.split()) < 20:
                elem.clear()
                continue

            # combine question title + body + answer as the chunk
            text = f"Question: {q['title']}. {q['body'][:300]} Answer: {ans[:600]}"
            text = re.sub(r"\s+", " ", text).strip()

            qa_pairs.append({
                "chunk_id":      f"se_{elem.get('Id')}",
                "source":        "stackexchange",
                "question":      q["title"],
                "question_body": q["body"][:300],
                "answer":        ans[:600],
                "text":          text,
                "tags":          q["tags"],
                "q_score":       q["score"],
                "a_score":       score,
                "word_count":    len(text.split()),
            })

        elem.clear()

    return qa_pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not POSTS_XML_PATH.exists():
        print(f"ERROR: Posts.xml not found at {POSTS_XML_PATH}")
        print("Update POSTS_XML_PATH in this script to the correct location.")
        return

    print(f"Parsing {POSTS_XML_PATH}...\n")
    qa_pairs = parse_posts(POSTS_XML_PATH)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  QA pairs extracted : {len(qa_pairs)}")
    print(f"  Output             : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()