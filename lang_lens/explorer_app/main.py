import sys
import dill as pickle
import streamlit as st
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS


def load_explorer(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_explorer(path, explorer):
    with open(path, "wb") as f:
        pickle.dump(explorer, f)


def extract_keywords(texts, top_k=10):
    """Return top keywords from a list of texts using TF-IDF, removing stopwords."""
    if not texts:
        return []

    # Basic cleaning: lowercase, remove punctuation
    cleaned = [re.sub(f"[{string.punctuation}]", "", t.lower()) for t in texts]

    # Build TF-IDF on unigrams + bigrams, removing stopwords
    vectorizer = TfidfVectorizer(
        stop_words='english',  # explicit stopword removal
        ngram_range=(1, 2)
    )
    tfidf = vectorizer.fit_transform(cleaned)

    # Average tf-idf scores across documents
    scores = tfidf.toarray().mean(axis=0)
    terms = vectorizer.get_feature_names_out()

    # Remove any remaining stopwords (in case bigrams contain them)
    filtered_terms_scores = [
        (term, score) for term, score in zip(terms, scores)
        if all(word not in ENGLISH_STOP_WORDS for word in term.split())
    ]

    # Sort and return top_k
    filtered_terms_scores.sort(key=lambda x: x[1], reverse=True)
    top_terms = [term for term, _ in filtered_terms_scores[:top_k]]

    return top_terms

# Expect pickle path as CLI arg
if len(sys.argv) < 2:
    st.error("No pickle path provided!")
    st.stop()

pickle_path = sys.argv[-1]
explorer = load_explorer(pickle_path)
text_store = explorer.lens.text_store
lens_axes = explorer.lens.axis_discovery.get_axes()

st.title("Lang-Lens Explorer")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;  /* increase width */
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.5, 1])

with left:
    if explorer.inspections:
        st.subheader("Inspections")

        # Dropdown to pick inspection
        inspection_labels = [f"Inspection {i+1}" for i in range(len(explorer.inspections))]
        selected_inspection_label = st.selectbox(
            "Select an inspection",
            inspection_labels,
            key="inspection_select",
        )

        selected_idx = inspection_labels.index(selected_inspection_label)
        ins = explorer.inspections[selected_idx]

        st.markdown("**Axis Projections**")
        axis_labels = [axis.label for axis in ins.axes]
        projections = ins.projection

        cols = st.columns(len(axis_labels))
        for col, label, value in zip(cols, axis_labels, projections):
            with col:
                st.metric(label=label, value=f"{value:.3f}")

    else:
        st.info("No inspections yet — run `explorer.inspect(vec)` in Python before launching.")

with right:
    st.subheader("Axis Explorer")

    # Dropdown to pick axis
    axis_labels = [axis.label for axis in lens_axes]
    selected_axis_label = st.selectbox(
        "Select an axis", axis_labels, key="axis_select"
    )
    selected_axis = next(axis for axis in lens_axes if axis.label == selected_axis_label)

    # Rename axis
    new_label = st.text_input(
        "Rename axis", selected_axis.label, key=f"rename_{selected_axis.label}"
    )
    if new_label and new_label != selected_axis.label:
        selected_axis.set_label(new_label)
        save_explorer(pickle_path, explorer)  # persist change
        st.rerun()  # refresh UI everywhere with new label

    # All vectors + texts from store
    vectors = np.array(text_store.get_vectors())
    texts = text_store.get_texts()

    # Project all vectors onto this axis
    projections = selected_axis.transform(vectors)

    # Slider over projection range
    min_val, max_val = float(np.min(projections)), float(np.max(projections))
    if f"slider_{selected_axis.label}" not in st.session_state:
        st.session_state[f"slider_{selected_axis.label}"] = float(np.median(projections))

    slider_val = st.slider(
        f"Target value for {selected_axis.label}",
        min_val,
        max_val,
        value=st.session_state[f"slider_{selected_axis.label}"],
        key=f"slider_{selected_axis.label}",
    )

    # Find closest vectors to slider value
    diffs = np.abs(np.array(projections) - slider_val)

    # Take closest 5% of samples (at least 1)
    n_samples = max(1, int(0.05 * len(diffs)))
    closest_idx = np.argsort(diffs)[:n_samples]
    closest_texts = [texts[idx] for idx in closest_idx]

    # Keyword extraction
    keywords = extract_keywords(closest_texts, top_k=n_samples)

    st.markdown("**Top Keywords at this axis value**")
    st.write(", ".join(keywords))

    # Optional: still show texts
    with st.expander("See sample texts"):
        for idx in closest_idx[:5]:  # just show 5 examples
            st.write(f"`{projections[idx]:.3f}` — {texts[idx]}")
