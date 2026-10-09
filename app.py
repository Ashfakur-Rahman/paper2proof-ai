import re
import json
from datetime import datetime
from typing import Any

import fitz
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Paper2Proof AI — Day 2
# Transparent, rule-based extraction with page-level evidence.
# This does not use an LLM or claim to verify reproducibility.
# --------------------------------------------------

st.set_page_config(
    page_title="Paper2Proof AI",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0b1020; color: #f2f5fc; }
    [data-testid="stSidebar"] { background-color: #10182a; }
    div[data-testid="stMetric"] {
        background: #141f35; border: 1px solid #273652;
        padding: 14px; border-radius: 12px;
    }
    .hero {
        padding: 22px; border: 1px solid #263956; border-radius: 16px;
        background: linear-gradient(120deg, #111d34, #101525);
        margin-bottom: 18px;
    }
    .muted { color: #a8b5cc; }
    </style>
    """,
    unsafe_allow_html=True,
)


METRIC_PATTERNS = {
    "accuracy": r"\baccuracy\b",
    "precision": r"\bprecision\b",
    "recall": r"\brecall\b",
    "f1-score": r"\bf1[\s-]*score\b",
    "AUC": r"\bauc\b",
    "AUROC": r"\bauroc\b",
    "mAP@0.5": r"\bmap\s*@?\s*0?\.?5\b",
    "mAP50": r"\bmap50\b",
    "RMSE": r"\brmse\b",
    "MAE": r"\bmae\b",
    "BLEU": r"\bbleu\b",
    "ROUGE": r"\brouge(?:-?\d+)?\b",
    "R-squared": r"\br(?:2|²)\b|r-squared",
}

FIELD_PATTERNS = {
    "Research problem / objective": [
        r"\b(?:we|this paper|this study|our work)\s+(?:propose|present|introduce|investigate|address|aim|focus)",
        r"\bobjective(?:s)?\b",
        r"\bresearch question(?:s)?\b",
        r"\bproblem statement\b",
    ],
    "Dataset": [
        r"\bdataset\b", r"\bdata set\b", r"\bbenchmark\b",
        r"\bcollected from\b", r"\bpublicly available data\b",
    ],
    "Model / method": [
        r"\bmodel\b", r"\bclassifier\b", r"\bneural network\b",
        r"\btransformer\b", r"\bregression\b", r"\balgorithm\b",
        r"\bproposed method\b", r"\barchitecture\b",
    ],
    "Data split / validation": [
        r"\btrain(?:ing)?(?:/| and )test\b", r"\btraining set\b",
        r"\btest set\b", r"\bvalidation set\b", r"\bcross-validation\b",
        r"\bk-fold\b",
    ],
    "Hyperparameters / training": [
        r"\blearning rate\b", r"\bbatch size\b", r"\bepochs?\b",
        r"\boptimizer\b", r"\brandom[_ -]?state\b",
        r"\bhyperparameters?\b", r"\bweight decay\b",
    ],
    "Environment / implementation": [
        r"\bGPU\b", r"\bCPU\b", r"\bCUDA\b", r"\bPyTorch\b",
        r"\bTensorFlow\b", r"\bscikit-learn\b", r"\bimplementation details\b",
    ],
}


def clean_text(text: str) -> str:
    """Normalize extracted PDF text without changing its intended meaning."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(pdf_bytes: bytes) -> dict[str, Any]:
    """Extract selectable text from a PDF in memory, retaining page boundaries."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        metadata = doc.metadata or {}
        page_count = len(doc)
        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            pages.append({
                "page": page_number,
                "text": text,
                "characters": len(text),
            })

    full_text = "\n\n".join(
        f"[Page {page['page']}]\n{page['text']}" for page in pages
    )
    return {
        "metadata": metadata,
        "page_count": page_count,
        "pages": pages,
        "text": clean_text(full_text),
    }


def find_section(text: str, heading_pattern: str, next_headings: list[str]) -> str:
    """Heuristic section extraction; inspect the source PDF to confirm boundaries."""
    start_pattern = rf"(?im)^\s*(?:{heading_pattern})\s*:?\s*$"
    match = re.search(start_pattern, text)

    if not match:
        same_line_pattern = rf"(?im)^\s*(?:{heading_pattern})\s*:?\s+"
        match = re.search(same_line_pattern, text)

    if not match:
        return ""

    start = match.end()
    end = len(text)
    for next_heading in next_headings:
        next_match = re.search(
            rf"(?im)^\s*(?:{next_heading})\s*:?\s*$",
            text[start:],
        )
        if next_match:
            end = min(end, start + next_match.start())

    return text[start:end].strip()[:5000]


def sentence_like_lines(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Reconstruct sentence-level evidence from PDF text while retaining page numbers.

    PDF extraction often inserts a newline at the end of every visual line.
    Join wrapped lines before splitting sentences, and repair words split by
    a line-ending hyphen. This remains heuristic and may not resolve every
    multi-column layout, heading, or OCR/PDF text issue.
    """
    rows = []
    sentence_boundary = re.compile(
        r"""(?<=[.!?])\s+(?=(?:[A-Z0-9"'“‘(\[]))"""
    )

    for page in pages:
        raw_text = page.get("text", "")
        if not raw_text.strip():
            continue

        # Repair words split across visual lines, e.g. "charac-\\nters".
        text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", raw_text)

        # Single newlines usually indicate visual line wrapping; blank lines
        # are retained as paragraph boundaries.
        text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
        paragraphs = re.split(r"\n\s*\n", text)

        for paragraph in paragraphs:
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if not paragraph:
                continue

            for candidate in sentence_boundary.split(paragraph):
                sentence = candidate.strip(" \t\r\n")
                if len(sentence) < 25:
                    continue
                rows.append({"page": page["page"], "text": sentence})

    return rows



def extract_metric_evidence(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return candidate metric mentions with page number and exact text snippet."""
    found = []
    seen = set()
    for item in lines:
        line = item["text"]
        matches = [
            name for name, pattern in METRIC_PATTERNS.items()
            if re.search(pattern, line, flags=re.IGNORECASE)
        ]
        if not matches:
            continue

        key = (item["page"], line)
        if key in seen:
            continue
        seen.add(key)
        found.append({
            "page": item["page"],
            "candidate_metrics": ", ".join(matches),
            "text_evidence": line[:500],
            "verification_status": "Candidate mention — manually verify",
        })
        if len(found) >= 100:
            break
    return found


def extract_field_evidence(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find page-level text evidence for research specification fields."""
    evidence = []
    for field, patterns in FIELD_PATTERNS.items():
        hits = []
        for item in lines:
            if any(re.search(pattern, item["text"], re.IGNORECASE) for pattern in patterns):
                hits.append(item)
        # Keep a small, useful set of distinct lines per field.
        unique = []
        seen_text = set()
        for hit in hits:
            normalized = hit["text"].lower()
            if normalized in seen_text:
                continue
            seen_text.add(normalized)
            unique.append(hit)
            if len(unique) == 5:
                break

        evidence.append({
            "field": field,
            "status": "Candidate evidence found" if unique else "Not detected",
            "evidence_count": len(unique),
            "evidence": unique,
        })
    return evidence


def build_blueprint(extracted: dict[str, Any]) -> dict[str, Any]:
    text = extracted["text"]
    pages = extracted["pages"]
    lines = sentence_like_lines(pages)

    abstract = find_section(
        text, r"Abstract", [r"Keywords", r"1\s+Introduction", r"Introduction"]
    )
    methodology = find_section(
        text,
        r"Methodology|Methods|Materials and Methods|Experimental Setup",
        [r"Results", r"Experiments", r"Discussion", r"Conclusion", r"References"],
    )

    field_evidence = extract_field_evidence(lines)
    metric_evidence = extract_metric_evidence(lines)
    missing = [row["field"] for row in field_evidence if row["status"] == "Not detected"]

    return {
        "project": "Paper2Proof AI",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "Preliminary — requires human verification",
        "pages_processed": extracted["page_count"],
        "abstract_excerpt": abstract[:2000] if abstract else "",
        "methodology_excerpt": methodology[:3000] if methodology else "",
        "field_evidence": field_evidence,
        "metric_evidence": metric_evidence,
        "missing_or_unconfirmed_fields": missing,
        "important_note": (
            "Keyword matches are candidate evidence, not proof of a complete "
            "experimental specification or successful reproduction. Review each "
            "page reference and confirm details in the original PDF."
        ),
    }


# ---------------- Interface ----------------

st.markdown(
    """
    <div class="hero">
      <h1>🧪 Paper2Proof AI</h1>
      <p class="muted">From research papers to experiment blueprints.
      Inspect the evidence before you run the experiment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Day 2: rule-based extraction with page-level evidence. "
    "No external API, no invented results, and no automatic code execution."
)

with st.sidebar:
    st.header("Project")
    st.write("Version: Day 2 — Evidence extraction")
    st.write("Mode: Local processing")
    st.write("AI API: Not required")
    st.divider()
    st.caption(
        "Scanned PDFs may require OCR. Page-level matches and extracted sections "
        "are candidates and must be checked against the original paper."
    )

uploaded_file = st.file_uploader(
    "Upload a research paper (PDF)",
    type=["pdf"],
    help="The PDF is processed locally by this application.",
)

if uploaded_file is None:
    st.markdown("### Start with a research paper")
    st.write(
        "Upload a selectable-text PDF to inspect candidate research fields, "
        "metric mentions, page references, and missing information."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("PDF parsing", "Ready")
    c2.metric("Evidence tracking", "Page-level")
    c3.metric("Experiment runner", "Coming next")
    st.stop()

try:
    extracted = extract_pdf(uploaded_file.getvalue())
except Exception as exc:
    st.error(
        "The PDF could not be read. Check that it is a valid, unencrypted PDF "
        "and try again."
    )
    st.caption(f"Technical detail: {type(exc).__name__}")
    st.stop()

paper_text = extracted["text"]
metadata = extracted["metadata"]
page_count = extracted["page_count"]

if not paper_text.strip():
    st.warning(
        "No selectable text was found. This may be a scanned PDF. OCR support "
        "is not included yet."
    )
    st.stop()

blueprint = build_blueprint(extracted)

title_guess = (metadata.get("title") or "").strip()
if not title_guess:
    first_lines = [
        line.strip() for line in paper_text.splitlines()
        if len(line.strip()) > 12 and not line.strip().startswith("[Page ")
    ]
    title_guess = first_lines[0] if first_lines else "Title not detected"

st.subheader("Paper overview")
c1, c2, c3 = st.columns(3)
c1.metric("Pages", page_count)
c2.metric("Extracted characters", f"{len(paper_text):,}")
c3.metric("Candidate metric mentions", len(blueprint["metric_evidence"]))
st.markdown(f"**Title candidate:** {title_guess}")
st.caption("The title is a candidate, not a verified bibliographic record.")

tab_paper, tab_blueprint, tab_metrics, tab_export = st.tabs(
    ["📄 Extracted paper", "🧭 Blueprint & evidence", "📊 Metric evidence", "⬇️ Export"]
)

with tab_paper:
    st.subheader("Extracted text")
    st.caption("Page markers help you locate source passages.")
    st.text_area(
        "PDF text",
        value=paper_text,
        height=420,
        label_visibility="collapsed",
    )
    with st.expander("PDF metadata"):
        st.json({key: value for key, value in metadata.items() if value})

with tab_blueprint:
    st.subheader("Preliminary experiment blueprint")
    st.warning(
        "A keyword match is not proof that the paper provides a complete "
        "experimental specification."
    )

    st.markdown("#### Abstract excerpt")
    st.write(blueprint["abstract_excerpt"] or "Abstract section not detected reliably.")

    st.markdown("#### Methodology excerpt")
    st.write(
        blueprint["methodology_excerpt"]
        or "Methodology section not detected reliably."
    )

    st.markdown("#### Field evidence (with page references)")
    for row in blueprint["field_evidence"]:
        with st.expander(f"{row['field']} — {row['status']}"):
            if not row["evidence"]:
                st.write("No matching evidence line detected. This does not prove absence.")
            for hit in row["evidence"]:
                st.markdown(f"**Page {hit['page']}**")
                st.write(hit["text"])

    st.markdown("#### Missing or unconfirmed fields")
    if blueprint["missing_or_unconfirmed_fields"]:
        for field in blueprint["missing_or_unconfirmed_fields"]:
            st.markdown(f"- **{field}** — no matching text detected; inspect the PDF manually.")
    else:
        st.write(
            "Candidate evidence was found for every listed field. Completeness "
            "and correctness still require manual verification."
        )

with tab_metrics:
    st.subheader("Candidate metric mentions")
    metric_rows = blueprint["metric_evidence"]
    if metric_rows:
        st.dataframe(
            pd.DataFrame(metric_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "These are text matches, not verified reported scores. The value, "
            "dataset, units, comparison baseline, and evaluation conditions "
            "must be checked in the original paper."
        )
    else:
        st.write("No supported metric keywords were detected.")

with tab_export:
    st.subheader("Export your analysis")

    st.download_button(
        "Download extracted text",
        data=paper_text,
        file_name="extracted_paper.txt",
        mime="text/plain",
    )

    st.download_button(
        "Download preliminary blueprint (JSON)",
        data=json.dumps(blueprint, ensure_ascii=False, indent=2),
        file_name="paper2proof_blueprint.json",
        mime="application/json",
    )

    flat_field_rows = []
    for field_row in blueprint["field_evidence"]:
        if field_row["evidence"]:
            for hit in field_row["evidence"]:
                flat_field_rows.append({
                    "field": field_row["field"],
                    "status": field_row["status"],
                    "page": hit["page"],
                    "evidence_text": hit["text"],
                })
        else:
            flat_field_rows.append({
                "field": field_row["field"],
                "status": field_row["status"],
                "page": "",
                "evidence_text": "",
            })

    st.download_button(
        "Download field evidence (CSV)",
        data=pd.DataFrame(flat_field_rows).to_csv(index=False).encode("utf-8-sig"),
        file_name="paper2proof_field_evidence.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download metric evidence (CSV)",
        data=pd.DataFrame(blueprint["metric_evidence"]).to_csv(index=False).encode("utf-8-sig"),
        file_name="paper2proof_metric_evidence.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Paper2Proof AI · Day 2 · Evidence candidates, not a validated reproduction "
    "of published results."
)
