import os
import numpy as np
import joblib
import torch
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.train import MLP


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ClothAI | Research Dashboard",
    page_icon="🧥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(80,120,255,0.10), transparent 30%),
            radial-gradient(circle at 85% 10%, rgba(255,80,120,0.08), transparent 28%),
            #080c16;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #0d1220;
        border-right: 1px solid #202a3d;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ---------- HERO ---------- */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background:
            linear-gradient(135deg, #17223c 0%, #0e1628 55%, #101725 100%);
        border: 1px solid #293957;
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
        margin-bottom: 1.4rem;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -100px;
        top: -100px;
        border-radius: 50%;
        background: rgba(99,102,241,0.12);
        filter: blur(10px);
    }

    .eyebrow {
        color: #8ab4ff;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        color: #aab7ce;
        font-size: 1rem;
        margin-top: 0.7rem;
        max-width: 900px;
        line-height: 1.6;
    }

    /* ---------- KPI CARDS ---------- */

    .kpi {
        background: rgba(17,25,42,0.88);
        border: 1px solid #26334d;
        border-radius: 18px;
        padding: 1.1rem 1.15rem;
        min-height: 115px;
    }

    .kpi-label {
        color: #8d9bb5;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
    }

    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 0.25rem;
        color: #f5f7fb;
    }

    .kpi-sub {
        color: #74829d;
        font-size: 0.76rem;
        margin-top: 0.15rem;
    }

    /* ---------- SECTION ---------- */

    .section-label {
        color: #8ab4ff;
        font-size: 0.73rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.15rem;
        margin-bottom: 0.3rem;
    }

    .muted {
        color: #8d9bb5;
    }

    /* ---------- PIPELINE ---------- */

    .pipeline-card {
        background: #101827;
        border: 1px solid #26334d;
        border-radius: 16px;
        padding: 1rem 0.65rem;
        text-align: center;
        min-height: 105px;
    }

    .pipeline-number {
        color: #8ab4ff;
        font-size: 0.7rem;
        font-weight: 800;
    }

    .pipeline-title {
        font-weight: 750;
        margin-top: 0.3rem;
    }

    .pipeline-desc {
        color: #8290aa;
        font-size: 0.72rem;
        margin-top: 0.25rem;
    }

    /* ---------- INFO BOX ---------- */

    .info-box {
        background: #111a2b;
        border: 1px solid #293a59;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        color: #aebbd1;
        line-height: 1.55;
    }

    /* ---------- STATUS ---------- */

    .status {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.25);
        color: #75e6a2;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #4ade80;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #65728a;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1e293b;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

required = [
    "models/x_scaler.joblib",
    "models/pca.joblib",
    "models/mlp.pt",
    "results/metrics.txt",
]

missing = [p for p in required if not os.path.exists(p)]

if missing:
    st.error("Required model artifacts are missing.")
    st.code(
        "python src/generate_data.py\n"
        "python src/train.py\n"
        "python src/evaluate.py"
    )
    st.stop()


x_scaler = joblib.load("models/x_scaler.joblib")
pca = joblib.load("models/pca.joblib")

model = MLP()
model.load_state_dict(
    torch.load("models/mlp.pt", map_location="cpu")
)
model.eval()


# ============================================================
# LOAD METRICS
# ============================================================

metrics = {}

with open("results/metrics.txt", "r", encoding="utf-8") as f:
    for line in f:
        if ":" in line:
            key, value = line.rsplit(":", 1)

            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                pass


nn_mse = metrics.get("Neural Network Test MSE", np.nan)
nn_rmse = metrics.get("Neural Network Test RMSE", np.nan)
nn_mae = metrics.get("Neural Network Test MAE", np.nan)
lr_mse = metrics.get("Linear Regression Test MSE", np.nan)

pca_variance = metrics.get(
    "PCA explained variance (128 components)",
    float(np.sum(pca.explained_variance_ratio_))
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:1.4rem;font-weight:800;">
        🧥 ClothAI
        </div>
        <div style="color:#8290aa;font-size:.78rem;margin-top:.25rem;">
        Data-Driven Clothing Prediction
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Experiment")

    st.markdown(
        """
        <div class="status">
            <span class="dot"></span>
            Pipeline operational
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown("**Architecture**")
    st.code("56 → 128 → 256 → 128")

    st.markdown("**Compression**")
    st.write(f"PCA → {pca.n_components_} components")

    st.markdown("**Training**")
    st.write("Adam • MSE • Early stopping")

    st.divider()

    st.markdown("### Dataset scope")

    st.info(
        "The original physics-simulated dataset from the reference paper "
        "was not supplied with the assignment. This implementation uses "
        "a synthetic surrogate preserving the same dimensional structure."
    )

    st.divider()

    st.caption("Reference-paper-inspired ML implementation")


# ============================================================
# HERO
# ============================================================
st.html(
    f"""
<div class="hero">

    <div class="eyebrow">
        Machine Learning Mini-Project • Research Prototype
    </div>

    <div class="hero-title">
        🧥 ClothAI — Data-Driven Clothing Prediction
    </div>

    <div class="hero-subtitle">
        A PCA-based neural network pipeline that learns a mapping
        from a 56-dimensional human-pose representation to a
        high-dimensional clothing deformation representation.
    </div>

    <br>

    <div class="status">
        <span class="dot"></span>
        End-to-end prediction pipeline ready
    </div>

</div>
"""
)
# ============================================================
# NAVIGATION
# ============================================================

tabs = st.tabs(
    [
        "🔮 Live Prediction",
        "📊 Results",
        "🧠 Methodology",
        "📁 Dataset & Scope",
    ]
)

tab_predict, tab_results, tab_method, tab_dataset = tabs


# ============================================================
# LIVE PREDICTION
# ============================================================

with tab_predict:

    st.markdown(
        '<div class="section-label">Inference</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Live clothing deformation prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="muted">Provide a 56-dimensional pose vector and run the trained model.</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    preset = st.selectbox(
        "Pose configuration",
        [
            "Zero pose",
            "Random pose — seed 7",
            "Random pose — seed 42",
            "Custom",
        ],
    )

    if preset == "Zero pose":

        initial = np.zeros(56, dtype=np.float32)

    elif preset == "Random pose — seed 7":

        rng = np.random.default_rng(7)
        initial = rng.normal(size=56).astype(np.float32)
        initial /= max(np.linalg.norm(initial), 1e-6)

    elif preset == "Random pose — seed 42":

        rng = np.random.default_rng(42)
        initial = rng.normal(size=56).astype(np.float32)
        initial /= max(np.linalg.norm(initial), 1e-6)

    else:

        initial = np.zeros(56, dtype=np.float32)


    text = st.text_area(
        "56 comma-separated pose values",
        value=",".join(f"{v:.5f}" for v in initial),
        height=105,
        help="56 values = 14 joints × 4 quaternion parameters.",
    )


    run_prediction = st.button(
        "🚀 Run prediction",
        type="primary",
        width="stretch",
    )


    if run_prediction or "last_prediction" not in st.session_state:

        try:

            x = np.array(
                [
                    float(v.strip())
                    for v in text.split(",")
                    if v.strip()
                ],
                dtype=np.float32,
            )

            if len(x) != 56:

                st.error(
                    f"Expected exactly 56 values. Received {len(x)}."
                )

            else:

                xs = x_scaler.transform(
                    x.reshape(1, -1)
                )

                with torch.no_grad():

                    z = model(
                        torch.tensor(
                            xs,
                            dtype=torch.float32
                        )
                    ).numpy()

                y = pca.inverse_transform(z)[0]

                st.session_state.last_prediction = y
                st.session_state.last_pose = x

                st.success(
                    "Prediction completed successfully."
                )

        except ValueError:

            st.error(
                "Please enter only numeric comma-separated values."
            )


    # --------------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------------

    if "last_prediction" in st.session_state:

        y = st.session_state.last_prediction

        st.write("")

        left, right = st.columns([1.55, 1])


        # ---------- PROFILE ----------

        with left:

            st.markdown("### Predicted deformation profile")

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    y=y[:500],
                    mode="lines",
                    line=dict(
                        width=1.5
                    ),
                    name="Predicted offsets",
                )
            )

            fig.update_layout(
                height=390,
                template="plotly_dark",
                margin=dict(
                    l=20,
                    r=20,
                    t=35,
                    b=30,
                ),
                xaxis_title="Output coordinate index",
                yaxis_title="Offset value",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )


        # ---------- SUMMARY ----------

        with right:

            st.markdown("### Prediction summary")

            c1, c2 = st.columns(2)

            c1.metric(
                "Output values",
                f"{len(y):,}"
            )

            c2.metric(
                "Vertices",
                f"{len(y)//3:,}"
            )

            c1.metric(
                "Mean offset",
                f"{np.mean(y):.6f}"
            )

            c2.metric(
                "Std. deviation",
                f"{np.std(y):.6f}"
            )

            st.write("")

            csv_data = pd.DataFrame(
                {
                    "predicted_offset": y
                }
            ).to_csv(index=False)

            st.download_button(
                "⬇️ Download prediction CSV",
                data=csv_data,
                file_name="predicted_clothing_offsets.csv",
                mime="text/csv",
                width="stretch",
            )


        # ====================================================
        # 3D DEFORMATION VIEW
        # ====================================================

        st.write("")

        st.markdown(
            "### 3D deformation representation"
        )

        st.caption(
            "The 6,510 predicted values are reshaped into "
            "2,170 × 3 coordinates for an interactive point-cloud "
            "view. This is a deformation representation, not a "
            "reconstructed garment mesh because the original mesh "
            "topology was not supplied."
        )


        points = y.reshape(-1, 3)

        # Downsample for a responsive browser visualization
        max_points = 1000

        if len(points) > max_points:

            indices = np.linspace(
                0,
                len(points) - 1,
                max_points
            ).astype(int)

            display_points = points[indices]

        else:

            display_points = points


        fig3d = go.Figure()

        fig3d.add_trace(
            go.Scatter3d(
                x=display_points[:, 0],
                y=display_points[:, 1],
                z=display_points[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    opacity=0.75,
                    color=display_points[:, 2],
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(
                        title="Z offset"
                    ),
                ),
                name="Predicted deformation",
            )
        )

        fig3d.update_layout(
            height=600,
            template="plotly_dark",
            margin=dict(
                l=0,
                r=0,
                t=30,
                b=0,
            ),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
            title="Predicted 3D offset field",
        )

        st.plotly_chart(
            fig3d,
            width="stretch",
        )


        # ====================================================
        # FIRST VALUES
        # ====================================================

        with st.expander(
            "View first 30 predicted values"
        ):

            st.dataframe(
                pd.DataFrame(
                    {
                        "Index": np.arange(30),
                        "Predicted offset": y[:30],
                    }
                ),
                width="stretch",
                hide_index=True,
            )


# ============================================================
# RESULTS
# ============================================================

with tab_results:

    st.markdown(
        '<div class="section-label">Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Experimental results</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Results shown here come directly from the current "
        "surrogate-data experiment."
    )

    st.write("")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Neural Network MSE",
        f"{nn_mse:.8f}"
    )

    r2.metric(
        "Neural Network RMSE",
        f"{nn_rmse:.8f}"
    )

    r3.metric(
        "Neural Network MAE",
        f"{nn_mae:.8f}"
    )

    r4.metric(
        "Linear Regression MSE",
        f"{lr_mse:.8f}"
    )

    st.write("")

    c1, c2 = st.columns(2)

    with c1:

        if os.path.exists(
            "results/loss_curve.png"
        ):

            st.image(
                "results/loss_curve.png",
                caption="Training and validation loss",
            )

    with c2:

        if os.path.exists(
            "results/pca_variance.png"
        ):

            st.image(
                "results/pca_variance.png",
                caption="Cumulative PCA explained variance",
            )


    st.write("")

    st.markdown("### Baseline comparison")

    comparison = pd.DataFrame(
        {
            "Model": [
                "Neural Network",
                "Linear Regression",
            ],
            "Test MSE": [
                nn_mse,
                lr_mse,
            ],
        }
    )


    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison["Model"],
            y=comparison["Test MSE"],
            text=[
                f"{v:.6f}"
                for v in comparison["Test MSE"]
            ],
            textposition="auto",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=30,
        ),
        yaxis_title="Test MSE",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


    st.warning(
        "The linear baseline has a lower test MSE in this "
        "particular surrogate-data experiment. This is reported "
        "as an experimental finding rather than assuming that "
        "the neural network must be superior."
    )


# ============================================================
# METHODOLOGY
# ============================================================

with tab_method:

    st.markdown(
        '<div class="section-label">Architecture</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">How the system works</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "The system compresses the high-dimensional clothing "
        "representation with PCA and learns a pose-to-clothing "
        "mapping using a fully connected neural network."
    )

    st.write("")

    steps = [
        ("01", "Human pose", "56 features"),
        ("02", "Standardize", "Input scaling"),
        ("03", "PCA", "6,510 → 128"),
        ("04", "MLP", "56 → 128 → 256 → 128"),
        ("05", "Prediction", "PCA coefficients"),
        ("06", "Inverse PCA", "128 → 6,510"),
        ("07", "Evaluation", "MSE / RMSE / MAE"),
    ]

    pipeline_cols = st.columns(7)

    for col, (num, title, desc) in zip(
        pipeline_cols,
        steps
    ):

        with col:

            st.markdown(
                f"""
                <div class="pipeline-card">
                    <div class="pipeline-number">{num}</div>
                    <div class="pipeline-title">{title}</div>
                    <div class="pipeline-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


    st.write("")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("### Neural network")

        st.code(
            """
Input
56 pose features
        ↓
Dense(128) + ReLU
        ↓
Dense(256) + ReLU
        ↓
Output(128 PCA coefficients)
            """,
            language="text",
        )

    with c2:

        st.markdown("### Training configuration")

        st.code(
            """
Loss: Mean Squared Error
Optimizer: Adam
Batch size: 32
Early stopping: patience 15
Train / Val / Test: 1800 / 600 / 600
            """,
            language="text",
        )


    st.markdown("### Why PCA?")

    st.write(
        f"The original clothing representation contains 6,510 "
        f"values. PCA reduces this representation to "
        f"{pca.n_components_} components while retaining "
        f"approximately {pca_variance * 100:.4f}% of the variance "
        "in the training representation."
    )


# ============================================================
# DATASET & SCOPE
# ============================================================

with tab_dataset:

    st.markdown(
        '<div class="section-label">Dataset</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Dataset, assumptions and scope</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Reference-paper structure")

    st.write(
        "The reference paper describes a simulated clothing "
        "dataset using 56 pose inputs and 6,510 clothing-offset "
        "outputs."
    )


    st.markdown("### Dataset used in this implementation")

    st.info(
        "The raw physics-simulated dataset from the paper was "
        "not supplied with the assignment. Therefore, this "
        "implementation generates a synthetic surrogate dataset "
        "with the same dimensional structure so the complete "
        "machine-learning pipeline can be executed end-to-end."
    )


    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Samples",
        "3,000"
    )

    d2.metric(
        "Input",
        "56"
    )

    d3.metric(
        "Output",
        "6,510"
    )

    d4.metric(
        "Vertices",
        "2,170"
    )


    st.write("")

    st.markdown("### Data split")

    split_df = pd.DataFrame(
        {
            "Split": [
                "Training",
                "Validation",
                "Testing",
            ],
            "Samples": [
                1800,
                600,
                600,
            ],
        }
    )

    fig = go.Figure(
        go.Bar(
            x=split_df["Split"],
            y=split_df["Samples"],
            text=split_df["Samples"],
            textposition="auto",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        yaxis_title="Number of samples",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


    st.markdown("### Limitations")

    st.write(
        "The current implementation demonstrates the ML "
        "pipeline using a surrogate dataset. It does not "
        "reproduce the original physics-based cloth simulation "
        "or reconstruct the exact garment mesh topology."
    )


    st.markdown("### Future work")

    st.write(
        "Future work can use the original physics-simulated "
        "garment dataset, preserve its mesh topology, compare "
        "predicted and ground-truth garments visually, and "
        "evaluate different PCA dimensions and neural-network "
        "architectures."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ClothAI • Data-Driven Clothing Prediction •
        PCA + Fully Connected Neural Network
    </div>
    """,
    unsafe_allow_html=True,
)