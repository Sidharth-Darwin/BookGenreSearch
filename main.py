"""
Streamlit Genre Classifier App with Semantic‑plus‑Keyword Scoring
-----------------------------------------------------------------
• Loads `genres.json`.
• Embeds each genre once and caches via `st.cache_resource`.
• Cosine similarity (Sentence‑Transformers MiniLM) + RapidFuzz keyword hits.
• Each keyword hit contributes 0 .20 to the score (capped at 0 .60).
• UI: text area → top‑k genres with score & bonus breakdown, multiselect for manual tags.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz   # NEW
from dataclasses import dataclass

# ---------------------- CONFIG ------------------------------------
GENRE_FILE     = Path(__file__).with_name("genres.json")
MODEL_NAME     = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K          = 7
THRESHOLD      = 0.35           # apply to combined score
FUZZ_THRESHOLD = 80             # RapidFuzz partial_ratio cut‑off
KW_WEIGHT      = 0.20           # per‑hit bonus
KW_CAP         = 0.60           # max cumulative bonus
# ------------------------------------------------------------------

@dataclass
class GenreRow:
    name: str
    description: str
    keywords: List[str]
    emb: np.ndarray

# ------------------------------------------------------------------
# ⏳ CACHE EMBEDDINGS & GENRE DATA ----------------------------------
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embeddings …")
def load_resources() -> Tuple[SentenceTransformer, List[GenreRow], np.ndarray]:
    model = SentenceTransformer(MODEL_NAME)

    with GENRE_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    genres: List[GenreRow] = []
    texts = []
    for g in raw:
        keywords = g.get("keywords", [])  + [kw.strip().lower() for kw in g["name"].split("/")]
        row = GenreRow(
            name=g["name"],
            description=g["description"],
            keywords=keywords,
            emb=None  # placeholder
        )
        genres.append(row)
        texts.append(f"{g['name']}: {g['description']}")

    # embed & normalise
    mat = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)

    # attach embeddings back
    for row, e in zip(genres, mat):
        row.emb = e

    return model, genres, mat

model, GENRES, GENRE_MAT = load_resources()
GENRE_NAMES  = [g.name for g in GENRES]
GENRE_DESCR  = [g.description for g in GENRES]

# ------------------------------------------------------------------
# 🔍 SCORING HELPERS -------------------------------------------------
# ------------------------------------------------------------------
def keyword_bonus(text: str, keywords: List[str]) -> float:
    """Return cumulative bonus from RapidFuzz keyword matches."""
    t = text.lower()
    if len(keywords) == 1:
        return 0.2 if (fuzz.partial_ratio(keywords[0], t) >= FUZZ_THRESHOLD) else 0
    hits = sum(1 for kw in keywords if fuzz.partial_token_set_ratio(kw, t) >= FUZZ_THRESHOLD)
    return min(hits * KW_WEIGHT, KW_CAP)

@st.cache_resource(show_spinner=False)
def classify(text: str, k: int = TOP_K) -> List[Tuple[str, float, float, float]]:
    """
    Return list of (genre, semantic_sim, kw_bonus, combined_score)
    sorted by combined_score desc, trimmed to top‑k.
    """
    q = model.encode([text], convert_to_numpy=True)[0]
    q /= np.linalg.norm(q)
    sem_sims = GENRE_MAT @ q

    results = []
    for idx, sim in enumerate(sem_sims):
        bonus = keyword_bonus(text, GENRES[idx].keywords)
        combined = sim + bonus
        results.append((idx, sim, bonus, combined))

    results.sort(key=lambda r: r[3], reverse=True)
    top = [(GENRES[i].name, s, b, c) for i, s, b, c in results if c >= THRESHOLD][:k]
    return top or [(GENRES[i].name, s, b, c) for i, s, b, c in results[:k]]

# ------------------------------------------------------------------
# 🖼️ Streamlit UI ---------------------------------------------------
# ------------------------------------------------------------------
st.set_page_config(page_title="Genre Classifier", page_icon="📚", layout="centered")
st.title("📚 Zero‑Shot Genre Classifier (+ Keyword Fuzzy Bonus)")

user_input = st.text_area(
    "Enter a book/show summary, review, or subject list:",
    height=170,
)

if st.button("Classify", type="primary") and user_input.strip():
    hits = classify(user_input.strip())
    st.subheader("Top matches:")
    for g, sim, bonus, score in hits:
        st.markdown(
            f"**{g}**  "
            f"— semantic `{sim:.3f}`  "
            f"+ bonus `{bonus:.2f}`  "
            f"→ **`{score:.3f}`**"
        )

# Manual tag editor -------------------------------------------------
st.divider()
st.markdown("### 🏷️ Add / remove genres manually")
default_tags = [g for g, *_ in classify(user_input)] if user_input else None
selected = st.multiselect(
    "Search and select genres:",
    options=GENRE_NAMES,
    default=default_tags,
)
if selected:
    st.success("Tags saved: " + ", ".join(selected))
    st.markdown("**Selected genres:**")
    for g in selected:
        idx = GENRE_NAMES.index(g)
        st.write(f"- {GENRES[idx].name}: {GENRES[idx].description}")
else:
    st.info("No genres selected. Use the search to find and select genres.")
    st.markdown("**All available genres:**")
    for g, d in zip(GENRE_NAMES, GENRE_DESCR):
        st.write(f"- {g}: {d}")
