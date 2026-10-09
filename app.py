
import re
import json
from collections import Counter
from datetime import datetime

import fitz
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Paper2Proof AI — Day 1 MVP
# This version uses transparent, rule-based extraction.
# It does not claim to use an LLM or verify reproduction.
# --------------------------------------------------

st.set_page_config(
    page_title="Paper2Proof AI",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1020;
        color: #f2f5fc;
    }
    [data-testid="stSidebar"] {
        background-color: #10182a;
    }
    div[data-testid="stMetric"] {
        background: #141f35;
        border: 1px solid #273652;
        padding: 14px;
        border-radius: 12px;
    }
    .hero {
        padding: 22px;
        border: 1px solid #263956;
        border-radius: 16px;
        background: linear-gradient(120deg, #111d34, #101525);
        margin-bottom: 18px;
    }
    .muted {
        color: #a8b5cc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(text: str) -> str:
    """Normalize extracted PDF text without changing its meaning."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(pdf_bytes: bytes) -> dict:
    """Extract selectable text from a PDF in memory."""
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        metadata = doc.metadata or {}
        page_count = len(doc)

        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                    "characters": len(text),
                }
            )

    full_text = "\n\n".join(
        f"[Page {p['page']}]\n{p['text']}" for p in pages
    )

    return {
        "metadata": metadata,
        "page_count": page_count,
        "pages": pages,
        "text": clean_text(full_text),
    }


def find_section(text: str, heading: str, next_headings: list[str]) -> str:
    """
    Heuristic section extraction.
    Results must be checked against the original paper.
    """
    pattern = rf"(?im)^\s*{heading}\s*:?\s*$"
    match = re.search(pattern, text)

    if not match:
        # Some papers place a section heading at the start of a line
        # followed by content on the same line.
        pattern = rf"(?im)^\s*{heading}\s*:?\s+"
        match = re.search(pattern, text)

    if not match:
        return ""

    start = match.end()
    end = len(text)

    for next_heading in next_headings:
        next_match = re.search(
            rf"(?im)^\s*{next_heading}\s*:?\s*$",
            text[start:],
        )
        if next_match:
            end = min(end, start + next_match.start())

    section = text[start:end].strip()
    return section[:5000]


def extract_metrics(text: str) -> list[dict]:
    """Find candidate metric mentions; these are not verified results."""
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1-score",
        "f1 score",
        "auc",
        "auroc",
        "map@0.5",
        "map50",
        "rmse",
        "mae",
        "bleu",
        "rouge",
    ]

    found = []
    lines = text.splitlines()

    for page_line in lines:
        line = page_line.strip()
        lower = line.lower()

        if not line:
            continue

        matched = [name for name in metric_names if name in lower]

        if matched:
            found.append(
                {
                    "candidate_metrics": ", ".join(matched),
                    "text_evidence": line[:350],
                    "verification": "Needs manual verification",
                }
            )

        if len(found) >= 30:
            break

    return found


def build_blueprint(paper_text: str, page_count: int) -> dict:
    abstract = find_section(
        paper_text,
        "Abstract",
        ["Keywords", "1 Introduction", "Introduction"],
    )

    methodology = find_section(
        paper_text,
        "Methodology|Methods|Materials and Methods|Experimental Setup",
        ["Results", "Experiments", "Discussion", "Conclusion", "References"],
    )

    lower = paper_text.lower()

    candidates = {
        "dataset": any(
            word in lower
            for word in ["dataset", "data set", "benchmark"]
        ),
        "model": any(
            word in lower
            for word in [
                "model", "classifier", "neural network",
                "transformer", "regression",
            ]
        ),
        "split": any(
            word in lower
            for word in [
                "train/test", "training set", "test set",
                "validation set", "cross-validation",
            ]
        ),
        "metrics": any(
            word in lower
            for word in [
                "accuracy", "precision", "recall",
                "f1-score", "f1 score", "rmse", "mae",
            ]
        ),
        "hyperparameters": any(
            word in lower
            for word in [
                "learning rate", "batch size", "epochs",
                "random_state", "hyperparameter",
            ]
        ),
    }

    missing = [
        item for item, present in candidates.items() if not present
    ]

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "Preliminary — requires human verification",
        "pages_processed": page_count,
        "abstract_excerpt": abstract[:2000] if abstract else "",
        "methodology_excerpt": methodology[:3000] if methodology else "",
        "signals_detected": candidates,
        "missing_or_unconfirmed_signals": missing,
        "metric_candidates": extract_metrics(paper_text),
        "important_note": (
            "Keyword detection is not proof that a paper specifies "
            "a dataset, model, split, metric, or hyperparameter. "
            "Inspect the cited evidence and original PDF."
        ),
    }


# --------------------------------------------------
# Interface
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🧪 Paper2Proof AI</h1>
        <p class="muted">
            From research papers to experiment blueprints.
            Inspect the evidence before you run the experiment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Day 1 MVP: transparent PDF text extraction and heuristic analysis. "
    "No external API, no invented results, and no automatic code execution."
)

with st.sidebar:
    st.header("Project")
    st.write("Version: Day 1 MVP")
    st.write("Mode: Local processing")
    st.write("AI API: Not required")
    st.divider()
    st.caption(
        "Scanned PDFs may require OCR. Extracted sections and metric "
        "mentions must be checked against the source paper."
    )

uploaded_file = st.file_uploader(
    "Upload a research paper (PDF)",
    type=["pdf"],
    help="The PDF is processed locally by this application.",
)

if uploaded_file is None:
    st.markdown("### Start with a research paper")
    st.write(
        "Upload a selectable-text PDF to extract its content, "
        "inspect candidate metrics, and prepare a preliminary blueprint."
    )

    st.markdown("#### What this MVP can do")
    c1, c2, c3 = st.columns(3)
    c1.metric("PDF parsing", "Ready")
    c2.metric("Blueprint", "Preliminary")
    c3.metric("Experiment runner", "Coming next")

    st.stop()

try:
    pdf_bytes = uploaded_file.getvalue()
    extracted = extract_pdf(pdf_bytes)

except Exception as exc:
    st.error(
        "The PDF could not be read. Check that the file is a valid, "
        "unencrypted PDF and try again."
    )
    st.caption(f"Technical detail: {type(exc).__name__}")
    st.stop()

paper_text = extracted["text"]
metadata = extracted["metadata"]
page_count = extracted["page_count"]

if not paper_text.strip():
    st.warning(
        "No selectable text was found. This may be a scanned PDF. "
        "OCR support is not included in Day 1."
    )
    st.stop()

blueprint = build_blueprint(paper_text, page_count)

title_guess = metadata.get("title") or ""
if not title_guess.strip():
    first_lines = [
        line.strip()
        for line in paper_text.splitlines()
        if len(line.strip()) > 12
    ]
    title_guess = first_lines[1] if len(first_lines) > 1 else (
        first_lines[0] if first_lines else "Title not detected"
    )

st.subheader("Paper overview")

c1, c2, c3 = st.columns(3)
c1.metric("Pages", page_count)
c2.metric("Extracted characters", f"{len(paper_text):,}")
c3.metric("Metric mentions", len(blueprint["metric_candidates"]))

st.markdown(f"**Title candidate:** {title_guess}")
st.caption(
    "Title is a candidate, not a verified bibliographic record."
)

tab_paper, tab_blueprint, tab_metrics, tab_download = st.tabs(
    ["📄 Extracted paper", "🧭 Blueprint", "📊 Metric evidence", "⬇️ Export"]
)

with tab_paper:
    st.subheader("Extracted text")
    st.caption("Page markers are included to help locate source passages.")
    st.text_area(
        "PDF text",
        value=paper_text,
        height=420,
        label_visibility="collapsed",
    )

    with st.expander("PDF metadata"):
        st.json(
            {
                key: value
                for key, value in metadata.items()
                if value
            }
        )

with tab_blueprint:
    st.subheader("Preliminary experiment blueprint")
    st.warning(
        "Detected keywords are signals only. A signal does not prove "
        "that the paper provides a complete experimental specification."
    )

    st.markdown("#### Abstract excerpt")
    if blueprint["abstract_excerpt"]:
        st.write(blueprint["abstract_excerpt"])
    else:
        st.write("Abstract section not detected reliably.")

    st.markdown("#### Methodology excerpt")
    if blueprint["methodology_excerpt"]:
        st.write(blueprint["methodology_excerpt"])
    else:
        st.write("Methodology section not detected reliably.")

    st.markdown("#### Specification checklist")

    checklist_rows = [
        {
            "Item": key.replace("_", " ").title(),
            "Keyword signal detected": (
                "Yes" if value else "Not detected"
            ),
        }
        for key, value in blueprint["signals_detected"].items()
    ]

    st.dataframe(
        pd.DataFrame(checklist_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Missing or unconfirmed")
    if blueprint["missing_or_unconfirmed_signals"]:
        for item in blueprint["missing_or_unconfirmed_signals"]:
            st.markdown(f"- **{item.replace('_', ' ').title()}**")
    else:
        st.write(
            "Keyword signals were found for all checklist items. "
            "This does not mean they are fully specified."
        )

with tab_metrics:
    st.subheader("Candidate metric mentions")

    metric_rows = blueprint["metric_candidates"]

    if metric_rows:
        st.dataframe(
            pd.DataFrame(metric_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "These are text matches, not verified reported scores. "
            "Tables, units, datasets, and evaluation conditions "
            "must be inspected before using a value."
        )
    else:
        st.write("No supported metric keywords were detected.")

with tab_download:
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

st.divider()
st.caption(
    "Paper2Proof AI · Day 1 · Preliminary analysis, "
    "not a validated reproduction of published results."
)
