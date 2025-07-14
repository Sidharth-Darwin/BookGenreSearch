"""
Streamlit Genre Classifier App
----------------------------------

•  Load `genres.json` (must be present in the same folder).
•  Embed each genre (name + description) once at start‑up
•  Allow the user to enter a synopsis / blurb.
•  Return the top‑k matching genres (cosine similarity).
•  Provide a multiselect so users can manually tag items.
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 1 / Load genres --------------------------------------------------------------
# -----------------------------------------------------------------------------
GENRE_FILE = Path(__file__).with_name("genres.json")
if not GENRE_FILE.exists():
    st.error("genres.json not found – place it in the same directory as app.py")
    st.stop()

with GENRE_FILE.open("r", encoding="utf-8") as f:
    genres = json.load(f)

# Build text used for embedding (name + description)
genre_texts = [f"{g['name']}: {g['description']}" for g in genres]
genre_names = [g["name"] for g in genres]

# -----------------------------------------------------------------------------
# 2 / Sentence‑BERT model ------------------------------------------------------
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_embeddings():
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # ~80 MB, CPU‑friendly
    embs = model.encode(genre_texts, normalize_embeddings=True)
    return model, embs

model, genre_embs = load_model_and_embeddings()

# -----------------------------------------------------------------------------
# 3 / Helper – classify --------------------------------------------------------
# -----------------------------------------------------------------------------

def classify(text: str, top_k: int = 5, threshold: float = 0.35) -> List[Tuple[str, float]]:
    if not text.strip():
        return []
    query_emb = model.encode([text], normalize_embeddings=True)
    sims = np.dot(genre_embs, query_emb.T).flatten()
    # keep those above threshold; otherwise top‑k anyway
    idx = np.where(sims >= threshold)[0]
    if len(idx) == 0:
        idx = sims.argsort()[-top_k:][::-1]
    else:
        idx = idx[np.argsort(sims[idx])[::-1]]
    return [(genre_names[i], float(sims[i])) for i in idx[:top_k]]

# -----------------------------------------------------------------------------
# 4 / Streamlit UI -------------------------------------------------------------
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Genre Classifier", page_icon="📚", layout="centered")

st.title("📚 Genre Classifier (Sentence‑BERT + Cosine Similarity)")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top‑k genres", 1, 10, 5)
    thresh = st.slider("Similarity threshold", 0.0, 1.0, 0.35, 0.01)
    st.markdown("---")
    st.write("**Manual Tagging** – choose any genres to attach manually:")
    manual_tags = st.multiselect("Select genres", genre_names)

# Main input
user_text = st.text_area("Enter a synopsis / blurb / summary", height=150, placeholder="Paste or type text here…")

if st.button("Classify"):
    if not user_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Embedding & matching…"):
            results = classify(user_text, top_k=k, threshold=thresh)
        st.subheader("🔎 Suggested Genres")
        if results:
            for name, score in results:
                st.write(f"**{name}** — {score:.3f}")
        else:
            st.info("No genres passed the similarity threshold. Try lowering the threshold or adding keywords.")

if manual_tags:
    st.subheader("🖊️  Manual Tags Added")
    st.write(", ".join(manual_tags))
