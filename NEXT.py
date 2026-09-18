#!/usr/bin/env python3
"""ATLAS 3D Word Geometry Lab.

Run:
    pip install streamlit numpy pandas plotly scikit-learn
    pip install sentence-transformers nltk   # recommended
    streamlit run atlas_3d_word_geometry.py

The graph geometry is derived from declared coordinates and a declared metric.
Lexical relations are overlays; they do not create node positions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.manifold import MDS


APP_TITLE = "ATLAS 3D Word Geometry Lab"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_AXES = {
    "AFFECT_VALENCE": {
        "positive": ["love", "joy", "benefit", "kindness", "hope"],
        "negative": ["hate", "grief", "harm", "cruelty", "despair"],
    },
    "EPI_CERTAINTY": {
        "positive": ["certain", "known", "verified", "evidence", "confidence"],
        "negative": ["uncertain", "unknown", "doubt", "ambiguous", "possibly"],
    },
    "ABSTRACTION": {
        "positive": ["concept", "principle", "meaning", "theory", "abstraction"],
        "negative": ["stone", "table", "hammer", "body", "object"],
    },
    "AGENCY": {
        "positive": ["act", "choose", "control", "cause", "decide"],
        "negative": ["passive", "receive", "undergo", "constraint", "inert"],
    },
    "SELF_OTHER": {
        "positive": ["self", "myself", "identity", "internal", "own"],
        "negative": ["other", "you", "external", "stranger", "they"],
    },
    "TEMPORAL_DIRECTION": {
        "positive": ["future", "after", "later", "prediction", "becoming"],
        "negative": ["past", "before", "earlier", "memory", "history"],
    },
    "SOCIAL_ORIENTATION": {
        "positive": ["collective", "community", "cooperate", "shared", "together"],
        "negative": ["individual", "solitary", "private", "independent", "alone"],
    },
    "CAUSAL_DIRECTION": {
        "positive": ["cause", "condition", "trigger", "source", "produce"],
        "negative": ["effect", "outcome", "consequence", "result", "response"],
    },
    "LOG_POLARITY": {
        "positive": ["true", "valid", "consistent", "affirm", "yes"],
        "negative": ["false", "invalid", "contradiction", "negate", "no"],
    },
    "UNCERTAINTY": {
        "positive": ["uncertain", "ambiguous", "variable", "unresolved", "unknown"],
        "negative": ["stable", "precise", "resolved", "definite", "fixed"],
    },
}

RELATION_WEIGHTS = {
    "SYNONYM": 0.92,
    "ANTONYM": 0.96,
    "HYPERNYM": 0.88,
    "HYPONYM": 0.84,
    "MERONYM": 0.82,
    "HOLONYM": 0.82,
    "DERIVATIONALLY_RELATED": 0.82,
    "PERTAINYM": 0.76,
    "COORDINATE_TERM": 0.68,
}


@dataclass
class LexicalEdge:
    source: str
    target: str
    relationship: str
    relation_weight: float
    evidence: str
    provenance: str
    typed_distance: float | None = None
    projected_distance: float | None = None
    projection_residual: float | None = None


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


@st.cache_resource(show_spinner=False)
def load_sentence_transformer(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def embed_texts(texts: list[str], model_name: str, offline: bool, dims: int):
    if not offline:
        try:
            model = load_sentence_transformer(model_name)
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return normalize_rows(np.asarray(vectors, dtype=float)), model_name, "LEARNED_EMBEDDING"
        except Exception as exc:
            st.warning(f"Embedding model unavailable; using hashing proxy: {exc}")
    vectorizer = HashingVectorizer(
        n_features=dims, analyzer="char_wb", ngram_range=(2, 5),
        alternate_sign=False, norm="l2",
    )
    vectors = vectorizer.transform(texts).toarray()
    return normalize_rows(vectors), f"offline-hashed-{dims}d", "DERIVED_HASH_PROXY"


def parse_words(raw: str) -> list[str]:
    words = [x.strip() for x in re.split(r"[,\n;]+", raw) if x.strip()]
    return list(dict.fromkeys(words))


def parse_axes(raw: str) -> dict[str, dict[str, list[str]]]:
    value = json.loads(raw)
    clean = {}
    for name, poles in value.items():
        positive = [str(x).strip() for x in poles.get("positive", []) if str(x).strip()]
        negative = [str(x).strip() for x in poles.get("negative", []) if str(x).strip()]
        if positive and negative:
            clean[str(name)] = {"positive": positive, "negative": negative}
    if len(clean) < 3:
        raise ValueError("At least three axes with positive and negative anchors are required.")
    return clean


@st.cache_data(show_spinner=False)
def wordnet_neighborhood(seed_words: tuple[str, ...], relation_types: tuple[str, ...], limit: int):
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("word")
    except Exception:
        return [], "WordNet unavailable; only entered words will be graphed."

    edges: list[LexicalEdge] = []
    seen = set()

    def add(source: str, target: str, relation: str, evidence: str):
        target = target.replace("_", " ").strip()
        if not target or source.casefold() == target.casefold() or relation not in relation_types:
            return
        key = (source.casefold(), target.casefold(), relation)
        if key in seen:
            return
        seen.add(key)
        edges.append(LexicalEdge(
            source, target, relation, RELATION_WEIGHTS[relation], evidence, "WORDNET"
        ))

    for source in seed_words:
        synsets = wn.synsets(source.replace(" ", "_"))[:12]
        before = len(edges)
        for syn in synsets:
            definition = syn.definition()
            for lemma in syn.lemmas()[:8]:
                add(source, lemma.name(), "SYNONYM", definition)
                for ant in lemma.antonyms()[:4]:
                    add(source, ant.name(), "ANTONYM", "WordNet antonym")
                for item in lemma.derivationally_related_forms()[:4]:
                    add(source, item.name(), "DERIVATIONALLY_RELATED", "WordNet derivation")
                for item in lemma.pertainyms()[:4]:
                    add(source, item.name(), "PERTAINYM", "WordNet pertainym")
            for target_syn in syn.hypernyms()[:4]:
                for name in target_syn.lemma_names()[:3]:
                    add(source, name, "HYPERNYM", target_syn.definition())
            for target_syn in syn.hyponyms()[:6]:
                for name in target_syn.lemma_names()[:2]:
                    add(source, name, "HYPONYM", target_syn.definition())
            for target_syn in (syn.part_meronyms() + syn.member_meronyms() + syn.substance_meronyms())[:4]:
                for name in target_syn.lemma_names()[:2]:
                    add(source, name, "MERONYM", target_syn.definition())
            for target_syn in (syn.part_holonyms() + syn.member_holonyms() + syn.substance_holonyms())[:4]:
                for name in target_syn.lemma_names()[:2]:
                    add(source, name, "HOLONYM", target_syn.definition())
            for parent in syn.hypernyms()[:2]:
                for sibling in parent.hyponyms()[:8]:
                    if sibling != syn:
                        for name in sibling.lemma_names()[:1]:
                            add(source, name, "COORDINATE_TERM", parent.definition())
            if len(edges) - before >= limit:
                break
        source_edges = [e for e in edges if e.source == source]
        keep = set((e.source, e.target, e.relationship) for e in source_edges[:limit])
        edges = [e for e in edges if e.source != source or (e.source, e.target, e.relationship) in keep]
    return [asdict(e) for e in edges], ""


def build_axis_coordinates(words: list[str], axes: dict[str, Any], model: str, offline: bool, dims: int):
    anchor_texts = []
    for poles in axes.values():
        anchor_texts.extend(poles["positive"])
        anchor_texts.extend(poles["negative"])
    all_texts = list(dict.fromkeys(words + anchor_texts))
    vectors, used_model, status = embed_texts(all_texts, model, offline, dims)
    by_text = {text: vectors[i] for i, text in enumerate(all_texts)}
    rows = []
    for word in words:
        row = []
        for poles in axes.values():
            positive = normalize_rows(np.mean([by_text[x] for x in poles["positive"]], axis=0)[None, :])[0]
            negative = normalize_rows(np.mean([by_text[x] for x in poles["negative"]], axis=0)[None, :])[0]
            value = float(np.dot(by_text[word], positive) - np.dot(by_text[word], negative))
            row.append(value)
        rows.append(row)
    return np.asarray(rows), used_model, status


def scale_coordinates(x: np.ndarray) -> np.ndarray:
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    stds[stds < 1e-9] = 1.0
    return (x - means) / stds


def distance_matrix(x: np.ndarray, metric: str, weights: np.ndarray) -> np.ndarray:
    weighted = x * np.sqrt(weights)[None, :]
    n = len(x)
    out = np.zeros((n, n), dtype=float)
    if metric == "Cosine":
        z = normalize_rows(weighted)
        return np.clip(1.0 - z @ z.T, 0.0, 2.0)
    if metric == "Mahalanobis":
        covariance = np.cov(weighted, rowvar=False)
        inverse = np.linalg.pinv(np.atleast_2d(covariance) + np.eye(weighted.shape[1]) * 1e-6)
        for i in range(n):
            for j in range(i + 1, n):
                delta = weighted[i] - weighted[j]
                out[i, j] = out[j, i] = math.sqrt(max(0.0, float(delta @ inverse @ delta)))
        return out
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = float(np.linalg.norm(weighted[i] - weighted[j]))
    return out


def project_3d(x: np.ndarray, distances: np.ndarray, method: str, selected_axes: list[int], seed: int):
    n = len(x)
    if method == "Explicit typed axes":
        result = x[:, selected_axes[:3]]
        return result, "DIRECT_AXIS_VIEW"
    if n < 4:
        padded = np.zeros((n, 3))
        padded[:, :min(3, x.shape[1])] = x[:, :min(3, x.shape[1])]
        return padded, "PADDED_DIRECT_VIEW"
    if method == "PCA":
        return PCA(n_components=3, random_state=seed).fit_transform(x), "PCA"
    if method == "UMAP":
        try:
            import umap
            reducer = umap.UMAP(n_components=3, metric="precomputed", random_state=seed)
            return reducer.fit_transform(distances), "UMAP_PRECOMPUTED"
        except Exception as exc:
            st.warning(f"UMAP unavailable; using metric MDS: {exc}")
    model = MDS(
        n_components=3, dissimilarity="precomputed", random_state=seed,
        n_init=4, max_iter=500, eps=1e-6,
    )
    return model.fit_transform(distances), "METRIC_MDS"


def aligned_projected_distances(points: np.ndarray, native: np.ndarray):
    n = len(points)
    raw = np.zeros((n, n))
    upper_raw, upper_native = [], []
    for i in range(n):
        for j in range(i + 1, n):
            raw[i, j] = raw[j, i] = np.linalg.norm(points[i] - points[j])
            upper_raw.append(raw[i, j])
            upper_native.append(native[i, j])
    a = np.asarray(upper_raw)
    b = np.asarray(upper_native)
    scale = float((a @ b) / max(a @ a, 1e-12))
    aligned = raw * scale
    residuals = np.abs(aligned - native)
    stress = math.sqrt(float(np.sum((aligned - native) ** 2)) / max(float(np.sum(native ** 2)), 1e-12))
    return aligned, residuals, scale, stress


def geometry_figure(words, seed_set, points, edges, index, show_edges, max_edge_distance):
    fig = go.Figure()
    if show_edges:
        for edge in edges:
            if edge["source"] not in index or edge["target"] not in index:
                continue
            if edge["typed_distance"] is not None and edge["typed_distance"] > max_edge_distance:
                continue
            a, b = points[index[edge["source"]]], points[index[edge["target"]]]
            hover = (
                f"<b>{edge['source']} —{edge['relationship']}→ {edge['target']}</b><br>"
                f"relation weight: {edge['relation_weight']:.4f}<br>"
                f"typed distance: {edge['typed_distance']:.4f}<br>"
                f"projected distance: {edge['projected_distance']:.4f}<br>"
                f"projection residual: {edge['projection_residual']:.4f}<br>"
                f"provenance: {edge['provenance']}"
            )
            fig.add_trace(go.Scatter3d(
                x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
                mode="lines", line=dict(width=2, color="rgba(99,102,241,.35)"),
                hoverinfo="skip", showlegend=False,
            ))
            middle = (a + b) / 2
            fig.add_trace(go.Scatter3d(
                x=[middle[0]], y=[middle[1]], z=[middle[2]], mode="markers",
                marker=dict(size=5, color="rgba(0,0,0,0)"),
                customdata=[hover], hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            ))
    colors = ["#ff5c4d" if word in seed_set else "#29dfc1" for word in words]
    sizes = [9 if word in seed_set else 6 for word in words]
    hover = [f"<b>{word}</b><br>{'SEED' if word in seed_set else 'LEXICAL NEIGHBOR'}" for word in words]
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers+text",
        text=words, textposition="top center", customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        marker=dict(size=sizes, color=colors, line=dict(width=0.5, color="#d8e6ff")),
        showlegend=False,
    ))
    fig.update_layout(
        template="plotly_dark", height=820, title="ATLAS measured multi-axis word geometry",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Coordinates and declared metrics create the geometry. Lexical relationships are evidence overlays.")

with st.sidebar:
    st.header("Carrier words")
    raw_words = st.text_area(
        "Words or phrases — comma or newline separated",
        "freedom, liberty, autonomy, independence, constraint, responsibility",
        height=130,
    )
    expand = st.toggle("Expand with WordNet relations", True)
    relation_types = st.multiselect(
        "Relationship overlays", list(RELATION_WEIGHTS),
        default=["SYNONYM", "ANTONYM", "HYPERNYM", "HYPONYM", "COORDINATE_TERM"],
    )
    neighbor_limit = st.slider("Maximum neighbors per seed", 1, 40, 12)
    st.divider()
    st.header("Geometry")
    metric = st.selectbox("Native metric", ["Weighted Euclidean", "Cosine", "Mahalanobis"])
    projection = st.selectbox("3D projection", ["Metric MDS", "PCA", "Explicit typed axes", "UMAP"])
    standardize = st.toggle("Standardize typed axes", True)
    seed = int(st.number_input("Random seed", 0, 999999, 42))
    st.divider()
    st.header("Embedding observation layer")
    offline = st.toggle("Offline hashing proxy", False)
    model_name = st.text_input("Embedding model", DEFAULT_MODEL)
    hash_dims = st.slider("Offline dimensions", 64, 1024, 384, 64)

with st.expander("Typed-axis registry and anchors", expanded=False):
    axes_json = st.text_area(
        "Edit axis poles as JSON",
        json.dumps(DEFAULT_AXES, indent=2), height=520,
    )

try:
    axes = parse_axes(axes_json)
except Exception as exc:
    st.error(f"Axis registry error: {exc}")
    st.stop()

axis_names = list(axes)
st.subheader("Axis weights")
weight_columns = st.columns(min(5, len(axis_names)))
axis_weights = []
for i, axis in enumerate(axis_names):
    with weight_columns[i % len(weight_columns)]:
        axis_weights.append(st.slider(axis, 0.0, 3.0, 1.0, 0.05, key=f"weight::{axis}"))

explicit_axes = []
if projection == "Explicit typed axes":
    cols = st.columns(3)
    defaults = axis_names[:3]
    for i, label in enumerate(("X axis", "Y axis", "Z axis")):
        with cols[i]:
            selected = st.selectbox(label, axis_names, index=min(i, len(axis_names)-1), key=f"explicit::{i}")
            explicit_axes.append(axis_names.index(selected))
    if len(set(explicit_axes)) != 3:
        st.error("Choose three different explicit axes.")
        st.stop()

run = st.button("Construct 3D word geometry", type="primary", width="stretch")

if run:
    seeds = parse_words(raw_words)
    if len(seeds) < 2:
        st.error("Enter at least two words or phrases.")
        st.stop()
    edge_dicts, resource_note = wordnet_neighborhood(
        tuple(seeds), tuple(relation_types if expand else []), neighbor_limit
    )
    edges = [LexicalEdge(**item) for item in edge_dicts]
    words = list(dict.fromkeys(seeds + [e.target for e in edges]))
    coordinates, used_model, observation_status = build_axis_coordinates(
        words, axes, model_name, offline, hash_dims
    )
    working = scale_coordinates(coordinates) if standardize else coordinates.copy()
    weights = np.asarray(axis_weights, dtype=float)
    if not np.any(weights > 0):
        st.error("At least one axis weight must be greater than zero.")
        st.stop()
    distances = distance_matrix(working, metric, weights)
    selected = explicit_axes if explicit_axes else [0, 1, 2]
    points, projection_used = project_3d(working, distances, projection, selected, seed)
    projected, residuals, projection_scale, stress = aligned_projected_distances(points, distances)
    index = {word: i for i, word in enumerate(words)}
    for edge in edges:
        i, j = index[edge.source], index[edge.target]
        edge.typed_distance = round(float(distances[i, j]), 6)
        edge.projected_distance = round(float(projected[i, j]), 6)
        edge.projection_residual = round(float(residuals[i, j]), 6)
    coordinate_rows = []
    for i, word in enumerate(words):
        row = {"word": word, "role": "SEED" if word in seeds else "LEXICAL_NEIGHBOR"}
        row.update({axis: round(float(coordinates[i, j]), 6) for j, axis in enumerate(axis_names)})
        row.update({"projection_x": points[i, 0], "projection_y": points[i, 1], "projection_z": points[i, 2]})
        coordinate_rows.append(row)
    result = {
        "schema": "ATLAS_3D_WORD_GEOMETRY_v1",
        "words": words, "seeds": seeds, "axes": axes, "axis_weights": dict(zip(axis_names, axis_weights)),
        "metric": metric, "projection_requested": projection, "projection_used": projection_used,
        "projection_scale": projection_scale, "normalized_stress": stress,
        "embedding_model": used_model, "observation_status": observation_status,
        "coordinates": coordinate_rows, "relations": [asdict(e) for e in edges],
        "epistemic_note": "Relation weight, typed distance, and projected distance are distinct quantities.",
        "resource_note": resource_note,
    }
    st.session_state["atlas_3d_result"] = result

if "atlas_3d_result" in st.session_state:
    result = st.session_state["atlas_3d_result"]
    coordinate_df = pd.DataFrame(result["coordinates"])
    edge_df = pd.DataFrame(result["relations"])
    words = result["words"]
    points = coordinate_df[["projection_x", "projection_y", "projection_z"]].to_numpy()
    index = {word: i for i, word in enumerate(words)}
    m = st.columns(5)
    m[0].metric("Words", len(words))
    m[1].metric("Typed axes", len(result["axes"]))
    m[2].metric("Relations", len(edge_df))
    m[3].metric("Projection", result["projection_used"])
    m[4].metric("Normalized stress", f"{result['normalized_stress']:.4f}")
    if result["resource_note"]:
        st.info(result["resource_note"])

    max_native = float(edge_df["typed_distance"].max()) if not edge_df.empty else 1.0
    c1, c2 = st.columns([1, 3])
    with c1:
        show_edges = st.toggle("Show relationship overlays", True)
        threshold = st.slider("Maximum typed edge distance", 0.0, max(0.01, max_native), max_native)
        st.markdown(
            "**Geometry contract**\n\n"
            "- Node positions: typed coordinates + declared metric\n"
            "- Edges: lexical evidence overlays\n"
            "- Relation weight: evidence strength\n"
            "- Residual: projection distortion"
        )
    with c2:
        st.plotly_chart(
            geometry_figure(
                words, set(result["seeds"]), points, result["relations"], index,
                show_edges, threshold,
            ),
            width="stretch", config={"scrollZoom": True, "displaylogo": False},
        )

    tabs = st.tabs(["Typed coordinates", "Lexical relations", "Axis registry", "Export"])
    with tabs[0]:
        st.dataframe(coordinate_df, hide_index=True, width="stretch", height=520)
    with tabs[1]:
        if edge_df.empty:
            st.info("No lexical relationships were returned. The word geometry remains valid.")
        else:
            st.dataframe(edge_df, hide_index=True, width="stretch", height=520)
    with tabs[2]:
        st.json({
            "axes": result["axes"], "weights": result["axis_weights"],
            "metric": result["metric"], "projection": result["projection_used"],
            "embedding_model": result["embedding_model"],
            "observation_status": result["observation_status"],
        })
    with tabs[3]:
        st.download_button(
            "Download complete geometry JSON",
            json.dumps(result, indent=2, ensure_ascii=False),
            "atlas_3d_word_geometry.json", "application/json", width="stretch",
        )
        st.download_button(
            "Download typed coordinates CSV", coordinate_df.to_csv(index=False),
            "atlas_3d_coordinates.csv", "text/csv", width="stretch",
        )
        st.download_button(
            "Download lexical relations CSV", edge_df.to_csv(index=False),
            "atlas_3d_relations.csv", "text/csv", width="stretch",
        )

