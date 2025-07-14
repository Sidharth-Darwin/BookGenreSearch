"""
Streamlit Genre Classifier App with Cached Embeddings
-----------------------------------------------------
• Loads `genres.json` (same folder).
• Embeds each genre once and caches the matrix with `st.cache_resource` → nearly instant hot‑reload.
• MiniLM (paraphrase‑MiniLM‑L6‑v2) for CPU‑friendly inference.
• User enters a blurb ➜ shows top‑k genres.
• Multiselect to add/remove tags with live search.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

GENRE_FILE = Path(__file__).with_name("genres.json")
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 7
THRESHOLD = 0.35

# ---------------------------------------------------------------------
# ⏳ CACHE EMBEDDINGS --------------------------------------------------
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embeddings …")
def load_embeddings() -> Tuple[SentenceTransformer, List[str], List[str], np.ndarray]:
    """Load model, genres, and pre‑compute normalized embeddings (cached)."""
    model = SentenceTransformer(MODEL_NAME)

    with GENRE_FILE.open("r", encoding="utf-8") as f:
        genres = json.load(f)

    texts = [f"{g['name']}: {g['description']}" for g in genres]
    names = [g["name"] for g in genres]
    descriptions = [g["description"] for g in genres]

    mat = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return model, names, descriptions, mat

model, genre_names, genre_descriptions, genre_mat = load_embeddings()

# ---------------------------------------------------------------------
# 🔍 Classifier --------------------------------------------------------
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def classify(text: str, k: int = TOP_K) -> List[Tuple[str, float]]:
    q = model.encode([text], convert_to_numpy=True)[0]
    q /= np.linalg.norm(q)
    sims = genre_mat @ q
    idx = sims.argsort()[-k:][::-1]
    return [(genre_names[i], float(sims[i])) for i in idx if sims[i] >= THRESHOLD]

# ---------------------------------------------------------------------
# 🖼️ Streamlit UI ------------------------------------------------------
# ---------------------------------------------------------------------
st.set_page_config(page_title="Genre Classifier", page_icon="📚", layout="centered")
st.title("📚 Zero‑Shot Genre Classifier")

user_input = st.text_area("Enter a book/show summary, review, or subject list:", height=170)
if st.button("Classify", type="primary") and user_input.strip():
    top_genres = classify(user_input.strip())
    if top_genres:
        st.subheader("Top matches:")
        for g, sc in top_genres:
            st.write(f"**{g}** – score `{sc:.3f}`")
    else:
        st.info("No genre scored above the threshold; showing closest anyway.")
        alt = classify(user_input.strip(), k=TOP_K)
        for g, sc in alt:
            st.write(f"**{g}** – score `{sc:.3f}`")

# Manual tag editor ---------------------------------------------------
st.divider()
st.markdown("### 🏷️ Add / remove genres manually")
selected = st.multiselect(
    "Search and select genres:",
    options=genre_names,
    default=[g for g, _ in classify(user_input)] if user_input else None,
)
if selected:
    selected_indices = [genre_names.index(g) for g in selected]
    genres_and_descriptions = [f"{genre_names[g]}: {genre_descriptions[g]}" for g in selected_indices]
    st.success("Tags saved: " + ", ".join(selected))
    st.markdown("**Selected genres:**")
    for gd in genres_and_descriptions:
        st.write(f"- {gd}")
else:
    st.info("No genres selected. Use the search to find and select genres.")
    st.markdown("**All available genres:**")
    for g, d in zip(genre_names, genre_descriptions):
        st.write(f"- {g}: {d}")

