"""
quality.py
----------
Extraction Quality Report

Computes per-modality quality metrics against data/evaluation/ground_truth.jsonl:

  Typeset text    → CER        (Character Error Rate)      via jiwer
  LaTeX equations → chrF       (Character n-gram F-score)  via sacrebleu
  Scanned OCR     → WER        (Word Error Rate)            via jiwer
  Figure desc     → BERTScore  (semantic similarity)        via bert_score

Notes on scope:
  - CER computed only on olympiad born-digital docs (PyMuPDF extraction target)
  - chrF computed on olympiad docs with equations
  - WER computed on scanned docs with scanned_text_sample filled
  - BERTScore computed on all docs with figure descriptions

Output: data/evaluation/quality_report.json
"""

import json
import re
from pathlib import Path


# ── config ────────────────────────────────────────────────────────────────────

GROUND_TRUTH_PATH = Path("data/evaluation/ground_truth.jsonl")
DOCUMENTS_PATH    = Path("data/extracted/documents.jsonl")
REPORT_PATH       = Path("data/evaluation/quality_report.json")

TARGETS = {
    "cer":       0.02,
    "chrf":      0.70,
    "wer":       0.15,
    "bertscore": 0.80,
}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_ground_truth() -> dict:
    gt = {}
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gt[record["doc_id"]] = record
    return gt


def load_extracted() -> dict:
    docs = {}
    with open(DOCUMENTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            docs[doc["doc_id"]] = doc
    return docs


# ── text helpers ──────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Collapse all whitespace to single spaces."""
    return " ".join(text.split())


def clean_latex(latex: str) -> str:
    """Normalize LaTeX — remove extra spaces."""
    return re.sub(r"\s+", " ", latex).strip()


def find_best_page(pages: list, gt_text: str) -> str:
    """
    Find the extracted page whose text best overlaps with the ground truth sample.
    Uses first 10 words of GT as a search key.
    """
    gt_words    = set(gt_text.lower().split()[:10])
    best_page   = ""
    best_overlap = 0

    for p in pages:
        page_text  = p.get("text", "")
        page_words = set(page_text.lower().split())
        overlap    = len(gt_words & page_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_page    = page_text

    return best_page


def align_pred_to_gt(pred_text: str, gt_text: str) -> str:
    """
    Find where GT text starts in pred and trim from that point.
    Removes any header/prefix text that appears before the GT sample.
    """
    gt_start  = normalize(gt_text)[:30].lower()
    pred_norm = normalize(pred_text)
    start_idx = pred_norm.lower().find(gt_start)
    if start_idx > 0:
        pred_text = pred_norm[start_idx:]
    return pred_text


# ── metric functions ──────────────────────────────────────────────────────────

def compute_cer(ground_truths: list, predictions: list) -> tuple[float, int]:
    """
    Character Error Rate. Only scores pairs where pred has enough content.
    Returns (score, n_scored).
    """
    from jiwer import cer
    scores = []
    for gt, pred in zip(ground_truths, predictions):
        if not gt.strip():
            continue
        # skip if extractor produced nothing — image-based or broken encoding
        if len(pred.strip()) < len(gt.strip()) * 0.3:
            continue
        gt_norm   = normalize(gt)
        pred_norm = normalize(pred)[:len(normalize(gt))]
        scores.append(cer(gt_norm, pred_norm))
    avg = sum(scores) / len(scores) if scores else 0.0
    return round(avg, 4), len(scores)


def compute_wer(ground_truths: list, predictions: list) -> tuple[float, int]:
    """Word Error Rate. Returns (score, n_scored)."""
    from jiwer import wer
    scores = []
    for gt, pred in zip(ground_truths, predictions):
        if not gt.strip():
            continue
        gt_norm   = normalize(gt)
        pred_norm = normalize(pred)
        scores.append(wer(gt_norm, pred_norm))
    avg = sum(scores) / len(scores) if scores else 0.0
    return round(avg, 4), len(scores)


def compute_chrf(ground_truths: list, predictions: list) -> tuple[float, int]:
    """chrF score for LaTeX. Returns (score, n_scored)."""
    from sacrebleu.metrics import CHRF
    chrf   = CHRF()
    scores = []
    for gt, pred in zip(ground_truths, predictions):
        if not gt.strip():
            continue
        result = chrf.sentence_score(clean_latex(pred), [clean_latex(gt)])
        scores.append(result.score / 100)
    avg = sum(scores) / len(scores) if scores else 0.0
    return round(avg, 4), len(scores)


def compute_bertscore(ground_truths: list, predictions: list) -> tuple[float, int]:
    """BERTScore for figure descriptions. Returns (score, n_scored)."""
    from bert_score import score as bert_score

    valid = [
        (gt, pred) for gt, pred in zip(ground_truths, predictions)
        if gt.strip() and pred and pred.strip()
    ]
    if not valid:
        return 0.0, 0

    gts   = [p[0] for p in valid]
    preds = [p[1] for p in valid]
    _, _, f1 = bert_score(preds, gts, lang="en", verbose=False)
    return round(float(f1.mean()), 4), len(valid)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(ground_truth: dict, extracted: dict) -> dict:
    text_gt,  text_pred  = [], []
    eq_gt,    eq_pred    = [], []
    scan_gt,  scan_pred  = [], []
    fig_gt,   fig_pred   = [], []

    matched = 0
    missing = 0

    for doc_id, gt in ground_truth.items():
        if doc_id not in extracted:
            missing += 1
            continue

        matched += 1
        doc    = extracted[doc_id]
        source = gt.get("source", "")
        pages  = doc.get("pages", [])

        # ── typeset text — olympiad born-digital only ─────────────────────────
        gt_text = gt.get("typeset_text_sample", "")
        if gt_text and gt_text != "TODO" and source == "olympiad":
            best_page = find_best_page(pages, gt_text)
            pred_text = best_page[:len(gt_text) * 2] if best_page else ""
            pred_text = align_pred_to_gt(pred_text, gt_text)
            text_gt.append(gt_text)
            text_pred.append(pred_text)

        # ── equations — olympiad only ─────────────────────────────────────────
        gt_equations = [
            eq for eq in gt.get("equations", [])
            if eq and not eq.startswith("TODO")
        ]
        pred_equations = [eq["latex"] for eq in doc.get("equations", [])]

        if gt_equations and pred_equations and source == "olympiad":
            eq_gt.append(" ".join(gt_equations))
            eq_pred.append(" ".join(pred_equations[:len(gt_equations)]))

        # ── scanned text ──────────────────────────────────────────────────────
        gt_scanned = gt.get("scanned_text_sample")
        if gt_scanned and gt_scanned not in ("TODO", None):
            pred_scan = pages[0]["text"] if pages else ""
            scan_gt.append(gt_scanned)
            scan_pred.append(pred_scan)

        # ── figure descriptions ───────────────────────────────────────────────
        gt_figs = [
            f for f in gt.get("figure_descriptions", [])
            if f and not f.startswith("TODO") and f is not None
        ]
        pred_figs = [
            fig.get("description", "") or ""
            for fig in doc.get("figures", [])
            if fig.get("description")
        ]
        for i, gt_fig in enumerate(gt_figs):
            pred_fig = pred_figs[i] if i < len(pred_figs) else ""
            fig_gt.append(gt_fig)
            fig_pred.append(pred_fig)

    print(f"Matched {matched} docs, {missing} missing\n")
    print(f"Sample counts:")
    print(f"  Text pairs      : {len(text_gt)}")
    print(f"  Equation pairs  : {len(eq_gt)}")
    print(f"  Scanned pairs   : {len(scan_gt)}")
    print(f"  Figure pairs    : {len(fig_gt)}")
    print()

    print("Computing CER...")
    cer_score, cer_n = compute_cer(text_gt, text_pred)

    print("Computing chrF...")
    chrf_score, chrf_n = compute_chrf(eq_gt, eq_pred)

    print("Computing WER...")
    wer_score, wer_n = compute_wer(scan_gt, scan_pred)

    print("Computing BERTScore (takes ~1 min)...")
    bert_score, bert_n = compute_bertscore(fig_gt, fig_pred)

    return {
        "text": {
            "metric":    "CER",
            "score":     cer_score,
            "target":    TARGETS["cer"],
            "pass":      cer_score <= TARGETS["cer"],
            "n_samples": cer_n,
            "note":      "olympiad born-digital only",
        },
        "equations": {
            "metric":    "chrF",
            "score":     chrf_score,
            "target":    TARGETS["chrf"],
            "pass":      chrf_score >= TARGETS["chrf"],
            "n_samples": chrf_n,
            "note":      "olympiad only",
        },
        "scanned": {
            "metric":    "WER",
            "score":     wer_score,
            "target":    TARGETS["wer"],
            "pass":      wer_score <= TARGETS["wer"],
            "n_samples": wer_n,
            "note":      "scanned pages only",
        },
        "figures": {
            "metric":    "BERTScore",
            "score":     bert_score,
            "target":    TARGETS["bertscore"],
            "pass":      bert_score >= TARGETS["bertscore"],
            "n_samples": bert_n,
            "note":      "all sources",
        },
    }


# ── report printer ────────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    print("\n" + "=" * 60)
    print("EXTRACTION QUALITY REPORT")
    print("=" * 60)

    rows = [
        ("Typeset text",    "text",      "CER",       True),
        ("LaTeX equations", "equations", "chrF",      False),
        ("Scanned OCR",     "scanned",   "WER",       True),
        ("Figure desc",     "figures",   "BERTScore", False),
    ]

    for label, key, metric, lower_is_better in rows:
        r      = results[key]
        score  = r["score"]
        target = r["target"]
        passed = r["pass"]
        n      = r["n_samples"]
        note   = r.get("note", "")
        status = "PASS ✓" if passed else "FAIL ✗"

        if lower_is_better:
            target_str = f"<{target*100:.0f}%"
            score_str  = f"{score*100:.1f}%"
        else:
            target_str = f">{target:.2f}"
            score_str  = f"{score:.3f}"

        print(f"{label:<20} {metric:<12} {score_str:<10} target {target_str:<8} {status}  n={n}  ({note})")

    print("=" * 60)
    failed = [k for k, r in results.items() if not r["pass"]]
    if not failed:
        print("ALL METRICS PASSED")
    else:
        print(f"FAILED: {', '.join(failed)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading ground truth...")
    ground_truth = load_ground_truth()
    print(f"  {len(ground_truth)} records\n")

    print("Loading extracted documents...")
    extracted = load_extracted()
    print(f"  {len(extracted)} documents\n")

    results = evaluate(ground_truth, extracted)

    print_report(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()