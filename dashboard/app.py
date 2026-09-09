"""Interactive results dashboard for the Federated NIDS project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Federated NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
)

METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Macro Recall",
    "Macro F1",
    "Balanced Accuracy",
]

DEFAULT_PROJECT_ROOT = Path(
    r"C:\MyFiles\Project\Privacy-Preserving-Federated-NIDS"
)


def result_paths(project_root: Path) -> dict[str, Path]:
    """Return the project result files consumed by the dashboard."""
    results = project_root / "results"
    return {
        "comparison": results / "centralized_vs_federated_comparison.csv",
        "tradeoff": results / "xgboost_vs_federated_tradeoff.csv",
        "shap": results / "explainability" / "shap_global_feature_importance.csv",
        "summary": results / "final_project_summary.csv",
    }


@st.cache_data
def load_data(project_root_text: str):
    """Load dashboard data from the saved notebook outputs."""
    paths = result_paths(Path(project_root_text))
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))

    return (
        pd.read_csv(paths["comparison"]),
        pd.read_csv(paths["tradeoff"]),
        pd.read_csv(paths["shap"]),
        pd.read_csv(paths["summary"]),
    )


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


st.title("🛡️ Privacy-Preserving Federated NIDS")
st.caption(
    "Centralized vs federated intrusion-detection performance and XAI results"
)

with st.sidebar:
    st.header("Dashboard settings")
    project_root_text = st.text_input(
        "Project folder",
        value=str(DEFAULT_PROJECT_ROOT),
        help="Folder containing the data, results, notebooks, and dashboard directories.",
    )
    st.caption("The dashboard reads saved results only; it does not access raw traffic records.")

try:
    comparison_df, tradeoff_df, shap_df, summary_df = load_data(project_root_text)
except (FileNotFoundError, pd.errors.ParserError) as error:
    st.error("The dashboard could not load its saved results.")
    st.code(str(error))
    st.stop()

required_columns = {"Model", "Training Approach", *METRICS}
missing_columns = required_columns.difference(comparison_df.columns)
if missing_columns:
    st.error("The comparison file is missing required columns.")
    st.code(", ".join(sorted(missing_columns)))
    st.stop()

xgboost = comparison_df.loc[comparison_df["Model"] == "XGBoost"].iloc[0]
federated = comparison_df.loc[
    comparison_df["Training Approach"] == "Federated Learning"
].iloc[0]
top_feature = shap_df.iloc[0]
accuracy_gap = (xgboost["Accuracy"] - federated["Accuracy"]) * 100

metric_columns = st.columns(4)
metric_columns[0].metric("Best centralized model", "XGBoost")
metric_columns[1].metric("XGBoost accuracy", percent(xgboost["Accuracy"]))
metric_columns[2].metric("FedAvg accuracy", percent(federated["Accuracy"]))
metric_columns[3].metric("Accuracy trade-off", f"{accuracy_gap:.2f} pp")

overview_tab, comparison_tab, xai_tab, summary_tab = st.tabs(
    ["Overview", "Model comparison", "Explainable AI", "Project summary"]
)

with overview_tab:
    st.subheader("Performance overview")
    selected_metrics = ["Accuracy", "F1 Score", "Macro F1", "Balanced Accuracy"]
    overview_data = comparison_df.melt(
        id_vars=["Model", "Training Approach"],
        value_vars=selected_metrics,
        var_name="Metric",
        value_name="Score",
    )
    overview_data["Score"] *= 100

    chart = px.bar(
        overview_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        range_y=[0, 105],
        labels={"Score": "Score (%)"},
        title="Overall model performance",
    )
    chart.update_layout(legend_title_text="Metric")
    st.plotly_chart(chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("XGBoost vs FedAvg")
        st.dataframe(
            tradeoff_df.style.format(
                {
                    "Centralized XGBoost (%)": "{:.2f}",
                    "Federated FedAvg (%)": "{:.2f}",
                    "Difference (percentage points)": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Key interpretation")
        st.info(
            "The centralized XGBoost model has the highest overall accuracy. "
            "FedAvg provides a privacy-preserving alternative by aggregating "
            "model updates instead of sharing raw client records."
        )

with comparison_tab:
    st.subheader("Full model comparison")
    selected_metrics = st.multiselect(
        "Metrics to display",
        METRICS,
        default=METRICS,
    )

    comparison_data = comparison_df.melt(
        id_vars=["Model", "Training Approach"],
        value_vars=selected_metrics,
        var_name="Metric",
        value_name="Score",
    )
    comparison_data["Score"] *= 100

    comparison_chart = px.bar(
        comparison_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        range_y=[0, 105],
        labels={"Score": "Score (%)"},
        title="Centralized and federated model metrics",
    )
    comparison_chart.update_layout(legend_title_text="Metric")
    st.plotly_chart(comparison_chart, use_container_width=True)

    table_df = comparison_df[["Model", "Training Approach", *METRICS]].copy()
    for metric in METRICS:
        table_df[metric] *= 100
    st.dataframe(
        table_df.style.format({metric: "{:.2f}%" for metric in METRICS}),
        use_container_width=True,
        hide_index=True,
    )

with xai_tab:
    st.subheader("Global XGBoost feature importance")
    st.caption(
        "Mean absolute SHAP values show which network-traffic features most influenced XGBoost predictions."
    )

    top_n = st.slider("Number of features", 5, min(25, len(shap_df)), 15)
    top_features = shap_df.head(top_n).sort_values(
        "Mean Absolute SHAP Value",
        ascending=True,
    )

    shap_chart = px.bar(
        top_features,
        x="Mean Absolute SHAP Value",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} influential features",
        color_discrete_sequence=["#4C78A8"],
    )
    shap_chart.update_layout(yaxis_title="Network-traffic feature")
    st.plotly_chart(shap_chart, use_container_width=True)

    st.metric(
        "Most influential feature",
        top_feature["Feature"],
        f"SHAP importance: {top_feature['Mean Absolute SHAP Value']:.4f}",
    )
    st.warning(
        "SHAP explains how the model used features; it does not establish that a feature causes an intrusion."
    )

with summary_tab:
    st.subheader("Report-ready project summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    radar_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Macro F1"]
    radar = go.Figure()
    for row, label, color in [
        (xgboost, "Centralized XGBoost", "#4C78A8"),
        (federated, "Federated FedAvg", "#F58518"),
    ]:
        values = [row[metric] for metric in radar_metrics]
        radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself",
                name=label,
                line_color=color,
            )
        )
    radar.update_layout(
        title="XGBoost and FedAvg profile",
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
    )
    st.plotly_chart(radar, use_container_width=True)

st.divider()
st.caption("Built from the saved Notebook 05–09 outputs.")
