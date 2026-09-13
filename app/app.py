import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from app.utils.contract_loader import (
    extract_contract_text
)

from app.utils.pipeline_runner import (
    analyze_contract,
    MODEL_PATH
)

from src.models.nli_inference import (
    LegalBERTNLI
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="LexConflict",

    page_icon="⚖️",

    layout="wide",

    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .conflict-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    .high-risk {
        border-left: 6px solid #d9534f;
    }

    .medium-risk {
        border-left: 6px solid #f0ad4e;
    }

    .low-risk {
        border-left: 6px solid #5cb85c;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        background: #f7f8fa;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_nli_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Legal-BERT model was not found at:\n"
            f"{MODEL_PATH}\n\n"
            "Make sure models/lexconflict_legalbert "
            "exists in your project."
        )


    return LegalBERTNLI(

        model_path=str(
            MODEL_PATH
        ),

        device="cpu"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_results" not in st.session_state:

    st.session_state.analysis_results = None


if "contract_text" not in st.session_state:

    st.session_state.contract_text = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# ⚖️ LexConflict"
    )

    st.caption(
        "AI-Powered Legal Contract "
        "Conflict Detection"
    )

    st.divider()


    st.markdown(
        "### Analysis Settings"
    )


    top_k = st.slider(

        "Candidate clauses per clause",

        min_value=1,

        max_value=5,

        value=3
    )


    max_clauses = st.slider(

        "Maximum clauses",

        min_value=20,

        max_value=200,

        value=120,

        step=10
    )


    max_pairs = st.slider(

        "Maximum NLI pairs",

        min_value=50,

        max_value=500,

        value=300,

        step=50
    )


    st.divider()


    st.markdown(
        "### Pipeline"
    )

    st.write(
        "✓ Clause preprocessing"
    )

    st.write(
        "✓ TF-IDF retrieval"
    )

    st.write(
        "✓ Legal-BERT NLI"
    )

    st.write(
        "✓ Hybrid reasoning"
    )

    st.write(
        "✓ Conflict graph"
    )

    st.write(
        "✓ Risk scoring"
    )

    st.write(
        "✓ Resolution recommendation"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚖️ LexConflict</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Legal Contract Conflict Detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD AREA
# ============================================================

uploaded_file = st.file_uploader(

    "Upload a legal contract",

    type=[
        "pdf",
        "txt"
    ],

    help=(
        "Upload a text-based PDF or TXT contract."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_file is not None:

    st.info(
        f"Selected contract: "
        f"**{uploaded_file.name}**"
    )


    if st.button(

        "🔍 Analyze Contract",

        type="primary",

        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # Extract text
            # ------------------------------------------------

            with st.spinner(
                "Extracting contract text..."
            ):

                file_bytes = (
                    uploaded_file.getvalue()
                )


                contract_text = (
                    extract_contract_text(

                        file_bytes,

                        uploaded_file.name
                    )
                )


            if not contract_text:

                st.error(
                    "No readable text was found "
                    "in this document."
                )

                st.stop()


            st.session_state.contract_text = (
                contract_text
            )


            # ------------------------------------------------
            # Load model
            # ------------------------------------------------

            with st.spinner(
                "Loading Legal-BERT model..."
            ):

                nli_model = load_nli_model()


            # ------------------------------------------------
            # Run pipeline
            # ------------------------------------------------

            progress = st.progress(
                0
            )


            progress.progress(
                10
            )


            with st.spinner(
                "Running LexConflict analysis..."
            ):

                results = analyze_contract(

                    text=contract_text,

                    document_name=
                        uploaded_file.name,

                    nli_model=nli_model,

                    top_k=top_k,

                    max_clauses=
                        max_clauses,

                    max_pairs=
                        max_pairs
                )


            progress.progress(
                100
            )


            st.session_state.analysis_results = (
                results
            )


            st.success(
                "✓ Contract analysis completed."
            )


        except Exception as error:

            st.error(
                "Analysis failed."
            )

            st.exception(
                error
            )


# ============================================================
# RESULTS
# ============================================================

results = (
    st.session_state.analysis_results
)


if results is None:

    st.markdown(
        """
        ### Upload a contract to begin

        LexConflict analyzes a contract through:

        **Clause Processing → Candidate Retrieval →
        Legal-BERT NLI → Hybrid Conflict Reasoning →
        Conflict Graph → Risk Analysis → Recommendations**

        Upload a PDF or TXT file above and click
        **Analyze Contract**.
        """
    )

    st.stop()


# ============================================================
# SUMMARY
# ============================================================

clauses = results[
    "clauses"
]

conflicts = results[
    "conflicts"
]

risk_summary = results[
    "risk_summary"
]

risk_results = results[
    "risk_results"
]


maximum_risk = float(
    risk_summary.get(
        "maximum_risk",
        0
    )
)


average_risk = float(
    risk_summary.get(
        "average_risk",
        0
    )
)


risk_level = risk_summary.get(
    "overall_risk_level",
    "minimal"
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

st.markdown(
    "## 📊 Contract Overview"
)


col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Clauses",
        len(clauses)
    )


with col2:

    st.metric(
        "Conflicts",
        len(conflicts)
    )


with col3:

    st.metric(
        "Maximum Risk",
        f"{maximum_risk * 100:.1f}%"
    )


with col4:

    st.metric(
        "Risk Level",
        risk_level.upper()
    )


st.divider()


# ============================================================
# TABS
# ============================================================

(
    overview_tab,
    conflicts_tab,
    clauses_tab,
    graph_tab,
    recommendations_tab
) = st.tabs(
    [
        "📊 Overview",
        "⚠️ Conflicts",
        "📄 Clauses",
        "🕸️ Conflict Graph",
        "💡 Recommendations"
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with overview_tab:

    left, right = st.columns(
        2
    )


    # --------------------------------------------------------
    # Risk summary
    # --------------------------------------------------------

    with left:

        st.markdown(
            "### Contract Risk"
        )


        st.metric(

            "Average Clause Risk",

            f"{average_risk * 100:.1f}%"
        )


        st.metric(

            "Maximum Clause Risk",

            f"{maximum_risk * 100:.1f}%"
        )


        risk_counts = (
            risk_summary.get(
                "risk_level_counts",
                {}
            )
        )


        if risk_counts:

            risk_df = pd.DataFrame({

                "Risk Level":
                    list(
                        risk_counts.keys()
                    ),

                "Clauses":
                    list(
                        risk_counts.values()
                    )
            })


            fig = go.Figure(

                data=[

                    go.Bar(

                        x=risk_df[
                            "Risk Level"
                        ],

                        y=risk_df[
                            "Clauses"
                        ]
                    )
                ]
            )


            fig.update_layout(

                title="Risk Distribution",

                xaxis_title="Risk Level",

                yaxis_title="Number of Clauses",

                height=350
            )


            st.plotly_chart(

                fig,

                use_container_width=True
            )


    # --------------------------------------------------------
    # Top risk clauses
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### Highest-Risk Clauses"
        )


        for result in risk_results[:10]:

            score = float(
                result[
                    "risk_score"
                ]
            )


            level = result[
                "risk_level"
            ]


            st.markdown(

                f"""
                **{result['node_id']}**

                Risk: **{score * 100:.1f}%**

                Level: **{level.upper()}**

                Conflict connections:
                **{result['conflict_degree']}**
                """
            )


            st.divider()


# ============================================================
# CONFLICTS TAB
# ============================================================

with conflicts_tab:

    st.markdown(
        "## ⚠️ Detected Conflicts"
    )


    if not conflicts:

        st.success(
            "No conflicts were detected "
            "above the configured threshold."
        )


    else:

        st.warning(
            f"{len(conflicts)} conflict pair(s) detected."
        )


        conflict_options = []


        for index, conflict in enumerate(
            conflicts
        ):

            conflict_options.append(

                f"Conflict {index + 1}: "
                f"{conflict['query_clause_id']} ↔ "
                f"{conflict['candidate_clause_id']}"
            )


        selected_conflict = st.selectbox(

            "Select a conflict",

            conflict_options
        )


        selected_index = (
            conflict_options.index(
                selected_conflict
            )
        )


        conflict = conflicts[
            selected_index
        ]


        col_a, col_b = st.columns(
            2
        )


        with col_a:

            st.markdown(
                "### Clause A"
            )


            st.info(
                f"**{conflict['query_clause_id']}**"
            )


            st.write(
                conflict[
                    "query_text"
                ]
            )


        with col_b:

            st.markdown(
                "### Clause B"
            )


            st.info(
                f"**{conflict['candidate_clause_id']}**"
            )


            st.write(
                conflict[
                    "candidate_text"
                ]
            )


        st.divider()


        st.markdown(
            "### Conflict Analysis"
        )


        m1, m2, m3, m4 = (
            st.columns(4)
        )


        with m1:

            st.metric(

                "NLI",

                conflict[
                    "nli_label"
                ]
            )


        with m2:

            st.metric(

                "NLI Confidence",

                f"{conflict['nli_confidence'] * 100:.1f}%"
            )


        with m3:

            st.metric(

                "Hybrid Score",

                f"{conflict['hybrid_conflict_score'] * 100:.1f}%"
            )


        with m4:

            st.metric(

                "Severity",

                conflict[
                    "confidence_level"
                ]
            )


        st.markdown(
            "### Conflict Type"
        )


        st.code(
            conflict[
                "conflict_type"
            ]
        )


        st.markdown(
            "### Why was this flagged?"
        )


        st.write(
            conflict[
                "explanation"
            ]
        )


# ============================================================
# CLAUSES TAB
# ============================================================

with clauses_tab:

    st.markdown(
        "## 📄 Contract Clauses"
    )


    clause_df = pd.DataFrame(
        clauses
    )


    display_columns = [

        "clause_id",

        "text",

        "modality",

        "party",

        "risk_score",

        "risk_level",

        "conflict_degree"
    ]


    display_columns = [

        column

        for column in display_columns

        if column in clause_df.columns
    ]


    st.dataframe(

        clause_df[
            display_columns
        ],

        use_container_width=True,

        hide_index=True,

        column_config={

            "risk_score":
                st.column_config.ProgressColumn(

                    "Risk Score",

                    min_value=0,

                    max_value=1
                )
        }
    )


    st.markdown(
        "### Clause Details"
    )


    clause_ids = [

        clause["clause_id"]

        for clause in clauses
    ]


    selected_clause_id = st.selectbox(

        "Select a clause",

        clause_ids
    )


    selected_clause = next(

        clause

        for clause in clauses

        if clause["clause_id"]
        == selected_clause_id
    )


    st.markdown(
        f"#### {selected_clause_id}"
    )


    st.write(
        selected_clause[
            "text"
        ]
    )


    c1, c2, c3 = st.columns(
        3
    )


    with c1:

        st.write(
            "**Modality**"
        )

        st.write(
            selected_clause.get(
                "modality"
            ) or "Not detected"
        )


    with c2:

        st.write(
            "**Party**"
        )

        st.write(
            selected_clause.get(
                "party"
            ) or "Not detected"
        )


    with c3:

        st.write(
            "**Risk**"
        )

        st.write(

            f"{selected_clause.get('risk_score', 0) * 100:.1f}% "
            f"({selected_clause.get('risk_level', 'minimal').upper()})"
        )


# ============================================================
# CONFLICT GRAPH TAB
# ============================================================

with graph_tab:

    st.markdown(
        "## 🕸️ Conflict Graph"
    )


    edges = results[
        "graph"
    ][
        "edges"
    ]


    if not edges:

        st.info(
            "No conflict edges were detected, "
            "so there is no conflict graph to display."
        )


    else:

        graph = nx.Graph()


        for edge in edges:

            graph.add_edge(

                edge[
                    "source_id"
                ],

                edge[
                    "target_id"
                ],

                score=edge[
                    "hybrid_conflict_score"
                ],

                conflict_type=edge[
                    "conflict_type"
                ]
            )


        positions = nx.spring_layout(

            graph,

            seed=42
        )


        # ----------------------------------------------------
        # Edges
        # ----------------------------------------------------

        edge_x = []

        edge_y = []


        for source, target in graph.edges():

            x0, y0 = positions[
                source
            ]

            x1, y1 = positions[
                target
            ]


            edge_x.extend(
                [x0, x1, None]
            )

            edge_y.extend(
                [y0, y1, None]
            )


        edge_trace = go.Scatter(

            x=edge_x,

            y=edge_y,

            mode="lines",

            hoverinfo="none"
        )


        # ----------------------------------------------------
        # Nodes
        # ----------------------------------------------------

        node_x = []

        node_y = []

        node_text = []


        risk_lookup = {

            result[
                "node_id"
            ]:
                result

            for result in risk_results
        }


        for node in graph.nodes():

            x, y = positions[
                node
            ]


            node_x.append(
                x
            )

            node_y.append(
                y
            )


            risk = risk_lookup.get(
                node,
                {}
            )


            score = float(
                risk.get(
                    "risk_score",
                    0
                )
            )


            node_text.append(

                f"{node}<br>"
                f"Risk: {score * 100:.1f}%"
            )


        node_trace = go.Scatter(

            x=node_x,

            y=node_y,

            mode="markers+text",

            text=list(
                graph.nodes()
            ),

            textposition="top center",

            hovertext=node_text,

            hoverinfo="text",

            marker=dict(

                size=25
            )
        )


        fig = go.Figure(

            data=[
                edge_trace,
                node_trace
            ]
        )


        fig.update_layout(

            title=(
                "Detected Clause Conflict Network"
            ),

            showlegend=False,

            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),

            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),

            height=650
        )


        st.plotly_chart(

            fig,

            use_container_width=True
        )


        st.markdown(
            "### Graph Interpretation"
        )


        st.write(

            "Each node represents a clause. "
            "An edge represents a detected conflict "
            "between two clauses. Clauses with multiple "
            "connections may represent conflict hotspots "
            "within the contract."
        )


# ============================================================
# RECOMMENDATIONS TAB
# ============================================================

with recommendations_tab:

    st.markdown(
        "## 💡 Resolution Recommendations"
    )


    resolutions = results[
        "resolutions"
    ]


    if not resolutions:

        st.success(
            "No conflict-specific recommendations "
            "are required."
        )


    else:

        for index, recommendation in enumerate(
            resolutions,

            start=1
        ):

            st.markdown(
                f"### Recommendation {index}"
            )


            st.write(

                f"**Clauses:** "
                f"{recommendation['source_id']} ↔ "
                f"{recommendation['target_id']}"
            )


            st.write(

                f"**Severity:** "
                f"{recommendation['severity']}"
            )


            st.write(
                recommendation[
                    "recommendation"
                ]
            )


            st.markdown(
                "**Suggested revision:**"
            )


            st.info(
                recommendation[
                    "suggested_revision"
                ]
            )


            st.divider()


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.markdown(
    "## 📥 Export Analysis"
)


json_data = json.dumps(

    results,

    indent=2,

    ensure_ascii=False
)


st.download_button(

    label="Download Analysis JSON",

    data=json_data,

    file_name=(
        f"{results['document_id']}_"
        "lexconflict_analysis.json"
    ),

    mime="application/json"
)