from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.knn_utils import (
    cross_validate_k,
    feature_bounds,
    get_nearest_neighbors,
    predict_result,
    vote_scores,
)


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "students.csv"

st.set_page_config(page_title="KNN Interactive Classroom", page_icon="🎓", layout="wide")

# ─────────────────────────── colour palette ───────────────────────────────────
PLOTLY_PALETTE = [
    "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
    "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
]


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop fully-empty columns, then drop rows that still have any NaN.
    Returns the cleaned dataframe and a list of human-readable notes."""
    notes: list[str] = []

    # 1. Drop columns that are entirely empty
    fully_empty = [c for c in data.columns if data[c].isna().all()]
    if fully_empty:
        data = data.drop(columns=fully_empty)
        notes.append(f"Removed {len(fully_empty)} fully-empty column(s): {', '.join(fully_empty)}.")

    # 2. Drop rows that have any remaining NaN
    rows_before = len(data)
    data = data.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(data)
    if rows_dropped:
        notes.append(
            f"Removed {rows_dropped} row(s) with missing values "
            f"({rows_before - rows_dropped} rows kept)."
        )

    return data, notes


def validate_dataset(data: pd.DataFrame) -> tuple[bool, str]:
    if data.empty:
        return False, "The dataset is empty (no rows remain after cleaning)."
    if len(data.columns) < 3:
        return False, "The CSV must contain at least two numeric feature columns and one label column."
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    if len(numeric_columns) < 2:
        return False, "The CSV must contain at least two numeric columns."
    return True, ""


# ──────────────────────────── 2-D plot (matplotlib) ───────────────────────────

def draw_knn_plot_2d(
    data: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    new_student: list[float],
    neighbors: list[dict],
) -> plt.Figure:
    """Classic 2-D scatter with KNN neighborhood highlighted."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = data[label_column].astype(str).unique().tolist()
    cmap = plt.get_cmap("tab10", max(len(labels), 1))
    colors = {label: cmap(i) for i, label in enumerate(labels)}

    for label in labels:
        subset = data[data[label_column].astype(str) == label]
        ax.scatter(
            subset[feature_columns[0]], subset[feature_columns[1]],
            color=colors[label], s=90, alpha=0.78,
            edgecolors="white", linewidths=0.8, label=label,
        )

    if neighbors:
        ndf = pd.DataFrame(neighbors)
        ax.scatter(
            ndf[feature_columns[0]], ndf[feature_columns[1]],
            facecolors="none", edgecolors="#2563eb",
            linewidths=2.4, s=260, label="K nearest neighbors", zorder=4,
        )

    ax.scatter(
        new_student[0], new_student[1],
        color="#111827", marker="*", s=360,
        edgecolors="white", linewidths=1.2,
        label="New student", zorder=5,
    )
    ax.set_title("KNN — 2D feature space", fontsize=14, pad=12)
    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel(feature_columns[1])
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def draw_decision_regions_2d(
    data: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    k: int,
    weighting: str,
    gaussian_sigma: float,
) -> plt.Figure:
    bounds = feature_bounds(data, feature_columns)
    x_min, x_max = bounds[feature_columns[0]]
    y_min, y_max = bounds[feature_columns[1]]
    x_values = np.linspace(x_min, x_max, 55)
    y_values = np.linspace(y_min, y_max, 55)
    labels = sorted(data[label_column].astype(str).unique())
    label_to_number = {label: i for i, label in enumerate(labels)}
    grid_predictions = []
    for y_value in y_values:
        row_predictions = []
        for x_value in x_values:
            nbrs = get_nearest_neighbors(data, [x_value, y_value], k, feature_columns, label_column, bounds)
            row_predictions.append(label_to_number[predict_result(nbrs, weighting, gaussian_sigma)])
        grid_predictions.append(row_predictions)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10", max(len(labels), 1))
    ax.contourf(x_values, y_values, grid_predictions, levels=len(labels), cmap=cmap, alpha=0.20)
    for i, label in enumerate(labels):
        subset = data[data[label_column].astype(str) == label]
        ax.scatter(
            subset[feature_columns[0]], subset[feature_columns[1]],
            color=cmap(i), s=65, edgecolors="white", linewidths=0.7, label=label,
        )
    ax.set_title(f"Decision regions — {weighting} weighting, K={k}")
    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel(feature_columns[1])
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ──────────────────────────── 3-D Plotly scatter ──────────────────────────────

def draw_knn_plot_3d(
    data: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    new_student: list[float],
    neighbors: list[dict],
) -> go.Figure:
    """Interactive 3-D Plotly scatter — rotate, zoom, hover."""
    labels = sorted(data[label_column].astype(str).unique())
    fig = go.Figure()

    neighbor_indices = {n["Index"] for n in neighbors}

    for i, label in enumerate(labels):
        subset = data[data[label_column].astype(str) == label].reset_index()
        color = PLOTLY_PALETTE[i % len(PLOTLY_PALETTE)]

        # split into neighbors vs regular
        is_neighbor = subset["index"].isin(neighbor_indices)
        for flag, marker_symbol, size, opacity, suffix in [
            (False, "circle", 7, 0.75, ""),
            (True,  "circle-open", 14, 1.0, " (neighbor)"),
        ]:
            sub = subset[is_neighbor == flag]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter3d(
                x=sub[feature_columns[0]],
                y=sub[feature_columns[1]],
                z=sub[feature_columns[2]],
                mode="markers",
                marker=dict(size=size, color=color, symbol=marker_symbol,
                            line=dict(width=2, color=color), opacity=opacity),
                name=f"{label}{suffix}",
                hovertemplate=(
                    f"<b>{label}{suffix}</b><br>"
                    f"{feature_columns[0]}: %{{x}}<br>"
                    f"{feature_columns[1]}: %{{y}}<br>"
                    f"{feature_columns[2]}: %{{z}}<extra></extra>"
                ),
            ))

    # new student marker
    fig.add_trace(go.Scatter3d(
        x=[new_student[0]], y=[new_student[1]], z=[new_student[2]],
        mode="markers",
        marker=dict(size=14, color="#111827", symbol="diamond",
                    line=dict(width=2, color="white")),
        name="New student",
        hovertemplate=(
            f"<b>New student</b><br>"
            f"{feature_columns[0]}: %{{x}}<br>"
            f"{feature_columns[1]}: %{{y}}<br>"
            f"{feature_columns[2]}: %{{z}}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="KNN — 3D feature space (drag to rotate)",
        scene=dict(
            xaxis_title=feature_columns[0],
            yaxis_title=feature_columns[1],
            zaxis_title=feature_columns[2],
            bgcolor="rgba(248,249,250,1)",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
        paper_bgcolor="white",
    )
    return fig


# ──────────────────────────── 4-D Plotly scatter ──────────────────────────────

def draw_knn_plot_4d(
    data: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    new_student: list[float],
    neighbors: list[dict],
) -> go.Figure:
    """4-D interactive scatter: X/Y/Z axes + marker SIZE encodes 4th feature."""
    labels = sorted(data[label_column].astype(str).unique())
    fig = go.Figure()

    neighbor_indices = {n["Index"] for n in neighbors}

    # normalise 4th feature → marker size range 6–20
    f4_col = feature_columns[3]
    f4_min = float(data[f4_col].min())
    f4_max = float(data[f4_col].max())
    f4_range = f4_max - f4_min if f4_max != f4_min else 1.0

    def size_for(val: float) -> float:
        return 6.0 + 14.0 * (val - f4_min) / f4_range

    for i, label in enumerate(labels):
        subset = data[data[label_column].astype(str) == label].reset_index()
        color = PLOTLY_PALETTE[i % len(PLOTLY_PALETTE)]
        is_neighbor = subset["index"].isin(neighbor_indices)

        for flag, opacity, suffix in [(False, 0.7, ""), (True, 1.0, " (neighbor)")]:
            sub = subset[is_neighbor == flag]
            if sub.empty:
                continue
            sizes = [size_for(v) for v in sub[f4_col]]
            fig.add_trace(go.Scatter3d(
                x=sub[feature_columns[0]],
                y=sub[feature_columns[1]],
                z=sub[feature_columns[2]],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=color,
                    line=dict(width=1.5 if not flag else 2.5, color="white"),
                    opacity=opacity,
                ),
                name=f"{label}{suffix}",
                hovertemplate=(
                    f"<b>{label}{suffix}</b><br>"
                    f"{feature_columns[0]}: %{{x}}<br>"
                    f"{feature_columns[1]}: %{{y}}<br>"
                    f"{feature_columns[2]}: %{{z}}<br>"
                    f"{f4_col}: %{{customdata}}<extra></extra>"
                ),
                customdata=sub[f4_col].tolist(),
            ))

    # new student
    ns_size = size_for(new_student[3])
    fig.add_trace(go.Scatter3d(
        x=[new_student[0]], y=[new_student[1]], z=[new_student[2]],
        mode="markers",
        marker=dict(size=ns_size + 2, color="#111827", symbol="diamond",
                    line=dict(width=2, color="white")),
        name="New student",
        hovertemplate=(
            f"<b>New student</b><br>"
            f"{feature_columns[0]}: %{{x}}<br>"
            f"{feature_columns[1]}: %{{y}}<br>"
            f"{feature_columns[2]}: %{{z}}<br>"
            f"{f4_col}: {new_student[3]}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=f"KNN — 4D feature space (X/Y/Z + marker size = {f4_col})",
        scene=dict(
            xaxis_title=feature_columns[0],
            yaxis_title=feature_columns[1],
            zaxis_title=feature_columns[2],
            bgcolor="rgba(248,249,250,1)",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=55, b=0),
        height=620,
        paper_bgcolor="white",
    )
    return fig


# ─────────────────────── cached cross-validation ──────────────────────────────

@st.cache_data
def cached_cross_validation(
    data: pd.DataFrame,
    k_values: tuple[int, ...],
    feature_columns: tuple,
    label_column: str,
    weighting: str,
    gaussian_sigma: float,
) -> pd.DataFrame:
    return cross_validate_k(data, k_values, list(feature_columns), label_column, weighting, gaussian_sigma)


# ────────────────────────── prediction panel ──────────────────────────────────

def show_prediction(
    data: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    new_student: list[float],
    k: int,
    weighting: str,
    gaussian_sigma: float,
    viz_mode: str,
) -> None:
    bounds = feature_bounds(data, feature_columns)
    neighbors = get_nearest_neighbors(data, new_student, k, feature_columns, label_column, bounds)
    prediction = predict_result(neighbors, weighting, gaussian_sigma)
    neighbors_df = pd.DataFrame(neighbors)
    vote_counts = neighbors_df["Result"].value_counts()
    scores = vote_scores(neighbors, weighting, gaussian_sigma)

    neighbor_weights = []
    for distance in neighbors_df["Distance"]:
        distance = float(distance)
        if weighting == "uniform":
            weight = 1.0
        elif weighting == "distance":
            weight = 1.0 / max(distance, 1e-12)
        else:
            weight = float(np.exp(-(distance**2) / (2.0 * gaussian_sigma**2)))
        neighbor_weights.append(weight)
    neighbors_df["Vote weight"] = neighbor_weights

    st.divider()
    st.subheader("Prediction result")
    result_col, explanation_col = st.columns([1, 2])
    with result_col:
        st.success(f"Predicted class: **{prediction}**")
        st.metric("Selected K", k)
        st.metric("Visualization mode", viz_mode)
    with explanation_col:
        if weighting == "distance":
            vote_text = ", ".join(f"**{lbl}**: {sc:.2f}" for lbl, sc in scores.items())
            vote_description = "inverse-distance weighted scores"
        elif weighting == "gaussian":
            vote_text = ", ".join(f"**{lbl}**: {sc:.2f}" for lbl, sc in scores.items())
            vote_description = f"Gaussian weighted scores (sigma={gaussian_sigma:.2f})"
        else:
            vote_text = ", ".join(f"**{lbl}**: {cnt}" for lbl, cnt in vote_counts.items())
            vote_description = "uniform vote counts"
        st.write(
            f"The algorithm selected the **{k} closest examples** and used {vote_description}: {vote_text}. "
            f"The winning class is **{prediction}**."
        )

    st.subheader(f"{k} nearest neighbors")
    display_columns = ["Student", *feature_columns, "Result", "Distance", "Vote weight"]
    display_df = neighbors_df[display_columns].rename(
        columns={"Distance": "Scaled distance"}
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Class vote scores")
    metric_columns = st.columns(max(1, min(len(scores), 5)))
    for idx, (lbl, sc) in enumerate(scores.items()):
        value = f"{sc:.2f}" if weighting in {"distance", "gaussian"} else str(int(sc))
        metric_columns[idx % len(metric_columns)].metric(str(lbl), value)

    st.caption(
        "Distance is min-max scaled so every selected feature contributes on a comparable 0–1 scale. "
        "With inverse-distance weighting, a closer neighbor contributes more: weight = 1 / distance. "
        "With Gaussian weighting, the contribution decays smoothly as exp(-distance² / (2 sigma²))."
    )

    # ── Visualization ──
    st.subheader("Visualization")
    if viz_mode == "2D":
        st.pyplot(draw_knn_plot_2d(data, feature_columns, label_column, new_student, neighbors), clear_figure=True)
        st.subheader("Decision-region comparison (2D only)")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(draw_decision_regions_2d(data, feature_columns, label_column, k, "distance", gaussian_sigma), clear_figure=True)
        with c2:
            st.pyplot(draw_decision_regions_2d(data, feature_columns, label_column, k, "gaussian", gaussian_sigma), clear_figure=True)
    elif viz_mode == "3D":
        st.plotly_chart(
            draw_knn_plot_3d(data, feature_columns, label_column, new_student, neighbors),
            use_container_width=True,
        )
        st.caption("💡 Drag to rotate · Scroll to zoom · Double-click to reset · Click legend to toggle classes.")
    else:  # 4D
        st.plotly_chart(
            draw_knn_plot_4d(data, feature_columns, label_column, new_student, neighbors),
            use_container_width=True,
        )
        st.caption(
            f"💡 **4D encoding:** X / Y / Z axes carry the first three features. "
            f"**Marker size** encodes **{feature_columns[3]}** (larger = higher value). "
            "Drag to rotate · Scroll to zoom · Double-click to reset."
        )


# ══════════════════════════════ APP LAYOUT ════════════════════════════════════

st.title("🎓 KNN Interactive Classroom")
st.markdown(
    "Explore **multi-class K-Nearest Neighbors** in **2D**, **3D**, or **4D**. "
    "Switch visualization modes in the sidebar — the 3D and 4D charts are fully interactive "
    "(rotate, zoom, hover). Upload your own CSV or use the included student dataset."
)

# ── Sidebar: dataset ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded_file is None:
        data = load_default_data()
        st.caption("Using the included `data/students.csv` dataset.")
    else:
        try:
            raw = pd.read_csv(uploaded_file)
            data, clean_notes = clean_dataset(raw)
            if clean_notes:
                st.warning("**Auto-cleaned your CSV:**\n" + "\n".join(f"- {n}" for n in clean_notes))
            else:
                st.caption(f"Using uploaded file: `{uploaded_file.name}`")
        except Exception as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            st.stop()

valid, validation_message = validate_dataset(data)
if not valid:
    st.error(validation_message)
    st.info("Required shape: at least two numeric feature columns plus one label column.")
    st.stop()

numeric_columns = data.select_dtypes(include="number").columns.tolist()
all_columns = data.columns.tolist()

# ── Sidebar: visualization mode + column mapping ──────────────────────────────
with st.sidebar:
    st.header("Visualization mode")
    viz_mode = st.radio(
        "Select dimensions",
        options=["2D", "3D", "4D"],
        horizontal=True,
        help=(
            "2D — classic scatter (matplotlib) + decision boundary\n"
            "3D — interactive Plotly 3-D scatter (rotate/zoom)\n"
            "4D — interactive Plotly 3-D scatter where marker size encodes a 4th feature"
        ),
    )

    required_features = {"2D": 2, "3D": 3, "4D": 4}[viz_mode]

    st.header("Column mapping")
    if len(numeric_columns) < required_features:
        st.warning(f"{viz_mode} mode requires at least {required_features} numeric columns. Your dataset has {len(numeric_columns)}.")
        st.stop()

    default_features = numeric_columns[:required_features]
    feature_columns = st.multiselect(
        f"Choose exactly {required_features} numeric features",
        options=numeric_columns,
        default=default_features,
        max_selections=required_features,
    )
    label_options = [col for col in all_columns if col not in feature_columns]
    label_column = st.selectbox("Choose the class/label column", options=label_options)

if len(feature_columns) != required_features:
    st.warning(f"Select exactly {required_features} numeric features in the sidebar to continue.")
    st.stop()

# ── Sidebar: new example inputs ───────────────────────────────────────────────
with st.sidebar:
    st.header("New example")
    bounds = feature_bounds(data, feature_columns)
    new_student = []
    for column in feature_columns:
        minimum, maximum = bounds[column]
        default = float(data[column].median())
        new_student.append(
            st.number_input(
                column,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=(maximum - minimum) / 100 if maximum > minimum else 1.0,
            )
        )

    max_k = min(9, len(data))
    k_options = list(range(1, max_k + 1, 2))
    if max_k not in k_options:
        k_options.append(max_k)

    auto_k = st.checkbox("Automatically select K with cross-validation", value=False)
    weighting = st.selectbox(
        "Voting method",
        options=["uniform", "distance", "gaussian"],
        format_func=lambda o: {
            "uniform": "Uniform voting",
            "distance": "Inverse-distance weighting",
            "gaussian": "Gaussian distance weighting",
        }[o],
    )
    gaussian_sigma = st.slider(
        "Gaussian sigma (normalized distance)",
        min_value=0.03, max_value=0.60, value=0.15, step=0.01,
        disabled=weighting != "gaussian",
    )

    if auto_k:
        cv_k_values = tuple(v for v in k_options if v < len(data))
        cv_results = cached_cross_validation(
            data, cv_k_values, tuple(feature_columns),
            label_column, weighting, gaussian_sigma,
        )
        best_accuracy = cv_results["Accuracy"].max()
        k = int(cv_results.loc[cv_results["Accuracy"].idxmax(), "K"])
        st.caption(f"Cross-validation selected K={k} with accuracy {best_accuracy:.1%}.")
    else:
        k = st.selectbox("Choose K", options=k_options, index=min(1, len(k_options) - 1))

    predict_clicked = st.button("🚀 Predict class", type="primary", use_container_width=True)

# ── Main: dataset preview ─────────────────────────────────────────────────────
st.subheader("Dataset preview")
preview_col, summary_col = st.columns([1.5, 1])
with preview_col:
    st.dataframe(data, use_container_width=True, hide_index=True, height=260)
with summary_col:
    st.metric("Training examples", len(data))
    st.metric("Available classes", data[label_column].nunique())
    st.write("**Class distribution**")
    st.bar_chart(data[label_column].astype(str).value_counts())

if auto_k:
    st.subheader("Cross-validation results")
    cv_display = cv_results.copy()
    cv_display["Accuracy"] = cv_display["Accuracy"].map(lambda v: f"{v:.1%}")
    st.dataframe(cv_display, use_container_width=True, hide_index=True)
    st.line_chart(cv_results.set_index("K")["Accuracy"])

# ── Main: prediction or idle plot ────────────────────────────────────────────
if predict_clicked:
    show_prediction(data, feature_columns, label_column, new_student, k, weighting, gaussian_sigma, viz_mode)
else:
    st.divider()
    st.subheader("Ready to explore")
    st.write(
        "Adjust the controls in the sidebar and click **Predict class**. "
        "Switch between **2D**, **3D**, and **4D** modes to see the same neighbourhood from different angles."
    )
    # idle preview
    if viz_mode == "2D":
        st.pyplot(draw_knn_plot_2d(data, feature_columns, label_column, new_student, []), clear_figure=True)
    elif viz_mode == "3D":
        st.plotly_chart(
            draw_knn_plot_3d(data, feature_columns, label_column, new_student, []),
            use_container_width=True,
        )
        st.caption("💡 Drag to rotate · Scroll to zoom · Double-click to reset · Click legend to toggle classes.")
    else:
        st.plotly_chart(
            draw_knn_plot_4d(data, feature_columns, label_column, new_student, []),
            use_container_width=True,
        )
        st.caption(
            f"💡 **4D encoding:** marker size = **{feature_columns[3]}**. "
            "Drag to rotate · Scroll to zoom."
        )

# ── Expanders ─────────────────────────────────────────────────────────────────
with st.expander("Distance weighting and decision-boundary interpretation"):
    st.markdown(
        r"""
        **Uniform voting** gives every one of the K neighbors one vote. **Inverse-distance weighting** gives neighbor \(i\) the weight

        \[
        w_i = \frac{1}{d_i+\varepsilon},
        \]

        where \(d_i\) is its scaled Euclidean distance and \(\varepsilon\) is a tiny safeguard against division by zero.
        Gaussian weighting uses \(w_i=\exp(-d_i^2/(2\sigma^2))\). Decision-region plots are only shown in **2D** mode.

        In **3D / 4D** mode the neighborhood is shown by enlarged markers with an open ring (3D) or larger size (4D).
        """
    )

with st.expander("Mathematical formulation: scaled Euclidean distance"):
    st.markdown(
        r"""
        For features \(x_1,\ldots,x_n\), let \(m_j\) and \(M_j\) be the dataset min and max for feature \(j\).
        Min-max scaling maps each raw value to

        \[
        z_j = \frac{x_j - m_j}{M_j - m_j}.
        \]

        The scaled Euclidean distance between a new example \(q\) and training example \(p\) is

        \[
        d(p,q) = \sqrt{\sum_{j=1}^{n}\left(\frac{p_j - q_j}{M_j - m_j}\right)^2}.
        \]

        This app supports \(n \in \{2,3,4\}\) features depending on the selected visualization mode.
        """
    )
