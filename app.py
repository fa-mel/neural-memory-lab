"""
🧠 Neural Memory Lab — Hopfield Network Explorer
Streamlit app for interactively exploring associative memory,
Dale's Principle, and the standard-vs-inhibitory trade-off.

Mellini (2026) · Physical Methods of Biology · UNIBO
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import io, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hopfield import HopfieldNetwork
from utils import overlap, add_noise, sample_entropy, to_pil, pattern_to_image, frames_to_gif
from data import load_mnist_patterns

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
N_NEURONS = 784
CENTROID_STD = (0.834, 0.00749)
CENTROID_INH = (0.528, 0.00840)
BLUE, PURPLE, GREEN, YELLOW, RED, INDIGO = (
    "#60a5fa", "#c084fc", "#4ade80", "#facc15", "#f87171", "#818cf8"
)

def hex_alpha(hex_color, alpha=0.12):
    """Convert '#aabbcc' → 'rgba(r,g,b,alpha)' for plotly fills."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.set_page_config(page_title="Neural Memory Lab", page_icon="🧠", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
h1 { font-weight: 800; letter-spacing: -1px; }
h2, h3 { font-weight: 700; color: #94a3b8; }
code, .stCode { font-family: 'JetBrains Mono', monospace !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #0f172a; padding: 4px; border-radius: 10px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px; color: #64748b;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; padding: 6px 14px;
}
.stTabs [aria-selected="true"] { background: #1e293b !important; color: #e2e8f0 !important; }
.card {
    background: #0f172a; border: 1px solid #1e293b; border-radius: 12px;
    padding: 1rem; margin-bottom: 0.6rem; text-align: center;
}
.metric-big { font-size: 2.4rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }
.metric-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace; font-weight: 700; }
.badge-green  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-yellow { background: #1c1500; color: #facc15; border: 1px solid #713f12; }
.badge-red    { background: #1f0a0a; color: #f87171; border: 1px solid #7f1d1d; }
.explain-box {
    background: linear-gradient(135deg, #0f172a, #1a1035);
    border: 1px solid #6d28d9; border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.88rem; color: #c4b5fd; line-height: 1.65;
}
.warn-box {
    background: #1c1500; border: 1px solid #713f12; border-radius: 8px;
    padding: 0.5rem 0.8rem; font-size: 0.8rem; color: #facc15;
}
div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading MNIST…")
def get_patterns():
    return load_mnist_patterns()


def make_fig(**kw):
    """Create a plotly figure with consistent dark styling."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=11),
        margin=dict(l=50, r=25, t=45, b=40),
        **kw
    )
    return fig


def metric_card(col, label, value, color=INDIGO):
    col.markdown(
        f'<div class="card">'
        f'<div class="metric-big" style="color:{color}">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def interpret_result(ov, sampen, inhibitory, n_patterns):
    """Physics-grounded plain-language interpretation."""
    if ov < 0:
        return ("⚠️ **Anti-pattern convergence.** The network settled into the "
                "inverted attractor −ξ — a spurious minimum that always exists "
                "due to the ±symmetry of the Hebbian Hamiltonian. More likely "
                "with high noise or heavy memory load.")
    alpha = n_patterns / N_NEURONS
    parts = []
    if ov > 0.85:
        parts.append("✅ **Deep attractor convergence.** The recalled state is "
                      "well inside the target basin — consistent with low noise "
                      "and well-separated stored patterns.")
    elif ov > 0.5:
        parts.append("🟡 **Partial convergence.** The network moved toward the "
                      "target but stalled short of the prototype. This can "
                      "indicate proximity to the critical noise threshold "
                      f"(η_c ≈ 0.40) or cross-talk from similar patterns.")
    else:
        parts.append("🔴 **Spurious attractor.** Low similarity to any stored "
                      f"pattern. Likely causes: noise above η_c, or memory "
                      f"overload (current α = {alpha:.4f}).")
    if inhibitory:
        if sampen > 0.010:
            parts.append(f"🌀 **Ceaseless dynamics detected** (SampEn = {sampen:.5f}). "
                         "Dale's Principle broke the Lyapunov constraint — the energy "
                         "trajectory shows the irregular, non-monotonic regime "
                         "characteristic of frustrated networks.")
        else:
            parts.append(f"SampEn = {sampen:.5f}. Inhibitory network but the "
                         "trajectory is still relatively regular for this input.")
    else:
        parts.append(f"SampEn = {sampen:.5f}. Consistent with monotonic energy "
                     "descent toward a fixed-point attractor (paper centroid: "
                     "0.00749).")
    if alpha > 0.004:
        parts.append(f"⚠️ Load α = {alpha:.4f} ({n_patterns}/784). MNIST digits "
                     "share structure — the paper observes catastrophic forgetting "
                     "at M ≈ 3.")
    return "\n\n".join(parts)


def canvas_to_pattern(canvas_result):
    """Convert drawable-canvas output to a ±1 pattern."""
    r = canvas_result.image_data[:, :, 0].astype(np.uint8)
    gray = Image.fromarray(r, "L").resize((28, 28), Image.LANCZOS)
    arr = np.array(gray)
    return np.where(arr > 30, 1.0, -1.0).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
all_patterns = get_patterns()

with st.sidebar:
    st.markdown("## 🧠 Neural Memory Lab")
    st.caption("Hopfield Network Explorer")
    st.markdown("---")

    st.markdown("**Memory bank**")
    stored_digits = st.multiselect(
        "Digits to store", list(range(10)),
        default=[0, 1, 3, 5, 8], label_visibility="collapsed",
    )
    n_pat = len(stored_digits)
    alpha = n_pat / N_NEURONS
    alpha_c = RED if n_pat >= 4 else YELLOW if n_pat == 3 else GREEN
    st.markdown(
        f'<div class="card" style="padding:0.5rem">'
        f'<span style="font-family:JetBrains Mono;color:{alpha_c};font-size:1.1rem;font-weight:700">'
        f'α = {alpha:.4f}</span><br>'
        f'<span class="metric-label">{n_pat} patterns / {N_NEURONS} neurons</span></div>',
        unsafe_allow_html=True,
    )
    if n_pat >= 4:
        st.markdown('<div class="warn-box">⚠️ Catastrophic forgetting likely '
                    '(MNIST patterns are non-orthogonal)</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Input**")
    input_mode = st.radio("Mode", ["MNIST + noise", "Draw digit"],
                          label_visibility="collapsed")

    if input_mode == "MNIST + noise":
        test_digit = st.selectbox("Digit to recall",
                                  options=stored_digits if stored_digits else [0])
        noise_pct = st.slider("Noise η (%)", 0, 50, 25, 5, format="%d%%")
        noise_level = noise_pct / 100.0
    else:
        test_digit, noise_pct, noise_level = None, 0, 0.0

    st.markdown("---")
    st.markdown("**Network**")
    inhibitory = st.toggle("Inhibitory (20%) — Dale's Principle", value=False)
    inh_frac = 0.2 if inhibitory else 0.0
    max_steps = st.slider("Recall steps", 200, 3000, 1500, 100)

    st.markdown("---")

if not stored_digits:
    st.warning("Select at least one digit to store.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER + TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧠 Neural Memory Lab")
st.caption("Hopfield Networks · Hebbian Learning · Dale's Principle · MNIST")

tabs = st.tabs([
    "⚡ Recall",
    "⚖️ Comparison",
    "📡 Phase Space",
    "🌀 Trajectories",
    "📊 Experiments",
    "🔬 Weights",
])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — RECALL LAB
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_in, col_cor, col_out = st.columns(3)

    # -- input column --
    with col_in:
        st.markdown("### ① Input")
        if input_mode == "MNIST + noise":
            st.image(to_pil(all_patterns[test_digit]),
                     caption=f"MNIST prototype '{test_digit}'",
                     use_container_width=True)
            input_pattern = all_patterns[test_digit].copy()
        else:
            try:
                from streamlit_drawable_canvas import st_canvas
                st.caption("Draw a digit below, then press **Run Recall**.")
                cr = st_canvas(
                    stroke_width=20, stroke_color="#FFFFFF",
                    background_color="#000000", height=196, width=196,
                    drawing_mode="freedraw", key="canvas_recall",
                )
                if cr.image_data is not None and cr.image_data.sum() > 0:
                    input_pattern = canvas_to_pattern(cr)
                else:
                    input_pattern = None
            except ImportError:
                st.error("Install `streamlit-drawable-canvas` to enable drawing: "
                         "`pip install streamlit-drawable-canvas`")
                input_pattern = None

    run = st.button("▶  Run Recall", type="primary", use_container_width=True,
                    key="btn_recall")

    with col_cor:
        st.markdown("### ② Corrupted")
    with col_out:
        st.markdown("### ③ Recalled")

    if run:
        if input_mode == "Draw digit" and input_pattern is None:
            st.warning("Draw something on the canvas first.")
            st.stop()

        patterns = [all_patterns[d] for d in stored_digits]
        net = HopfieldNetwork(N_NEURONS, inhibitory_fraction=inh_frac)
        net.train(patterns)

        if input_mode == "MNIST + noise":
            noisy = add_noise(input_pattern, noise_level, seed=42)
        else:
            noisy = np.ascontiguousarray(input_pattern, dtype=np.float64).flatten()

        with col_cor:
            lbl = f"Noise η = {noise_pct}%" if input_mode == "MNIST + noise" else "Your drawing"
            st.image(to_pil(noisy), caption=lbl, use_container_width=True)

        # Single recall pass — snapshots + energy in one go
        with st.spinner("Recalling…"):
            res = net.recall(noisy, max_steps=max_steps,
                             snapshot_every=max(1, max_steps // 80), seed=42)

        final = res["state"]
        energy = res["energy"]
        snapshots = res["snapshots"]

        with col_out:
            st.image(to_pil(final), caption="Recalled attractor",
                     use_container_width=True)

        # -- Animation slider --
        if len(snapshots) > 1:
            st.markdown("#### 🎞️ Recall animation")
            col_slider, col_dl = st.columns([5, 1])
            with col_slider:
                frame_idx = st.slider(
                    "Scrub through recall", 0, len(snapshots) - 1,
                    len(snapshots) - 1, key="frame_slider",
                    help="Drag to watch the network converge step by step",
                )
            with col_dl:
                gif_bytes = frames_to_gif(
                    [to_pil(s, 112) for s in snapshots], duration_ms=50
                )
                st.download_button("⬇ GIF", gif_bytes, "recall.gif",
                                   mime="image/gif", use_container_width=True)

            fc1, fc2, fc3 = st.columns([1, 2, 1])
            with fc2:
                st.image(to_pil(snapshots[frame_idx], 224),
                         caption=f"Frame {frame_idx}/{len(snapshots)-1}",
                         use_container_width=True)

        # -- Energy plot with step marker --
        st.markdown("#### 📉 Energy landscape")
        step_mark = frame_idx * (max_steps // 80) if len(snapshots) > 1 else max_steps
        fig_e = make_fig(
            title="Energy during asynchronous recall",
            xaxis_title="Update step",
            yaxis_title="E(S)",
            height=280,
        )
        fig_e.add_trace(go.Scatter(
            y=energy, mode="lines", line=dict(color=INDIGO, width=1.5),
            name="Energy", hovertemplate="Step %{x}<br>E = %{y:.1f}",
        ))
        if len(snapshots) > 1:
            fig_e.add_vline(x=step_mark, line_dash="dot", line_color=YELLOW,
                            annotation_text=f"frame {frame_idx}")
        st.plotly_chart(fig_e, use_container_width=True)

        # -- Metrics --
        sampen = sample_entropy(energy)
        overlaps_all = {d: overlap(final, all_patterns[d]) for d in stored_digits}
        guessed = max(overlaps_all, key=overlaps_all.get)
        best_ov = overlaps_all[guessed]
        ov_target = overlap(final, input_pattern)
        recovery = ov_target - overlap(noisy, input_pattern)

        m1, m2, m3, m4, m5 = st.columns(5)
        metric_card(m1, "Stored", str(n_pat), INDIGO)
        metric_card(m2, "Guessed", "?" if best_ov < 0 else str(guessed), INDIGO)
        metric_card(m3, "Overlap",
                    f"{ov_target:.3f}",
                    GREEN if ov_target > 0.8 else YELLOW if ov_target > 0.4 else RED)
        metric_card(m4, "Recovery Δ",
                    f"{recovery:+.3f}",
                    GREEN if recovery > 0 else RED)
        metric_card(m5, "SampEn",
                    f"{sampen:.5f}",
                    RED if sampen > 0.010 else BLUE)

        # -- Draw-mode guess --
        if input_mode == "Draw digit":
            badge = ("badge-green" if best_ov > 0.7 else
                     "badge-yellow" if best_ov > 0.4 else "badge-red")
            conf = "High" if best_ov > 0.7 else "Medium" if best_ov > 0.4 else "Low"
            st.markdown(
                f'<div class="card" style="padding:1.2rem">'
                f'<div class="metric-label" style="margin-bottom:4px">Best match</div>'
                f'<div style="font-size:4rem;font-weight:800;font-family:JetBrains Mono;'
                f'color:{INDIGO};line-height:1">{"?" if best_ov < 0 else guessed}</div>'
                f'<span class="badge {badge}">{conf} — overlap {best_ov:.3f}</span></div>',
                unsafe_allow_html=True,
            )

        # -- Physics interpretation --
        st.markdown("#### 🔬 Interpretation")
        st.markdown(
            f'<div class="explain-box">{interpret_result(ov_target, sampen, inhibitory, n_pat)}</div>',
            unsafe_allow_html=True,
        )

        # -- Stored gallery --
        with st.expander("🗃️ Stored memory bank", expanded=False):
            gcols = st.columns(min(len(stored_digits), 10))
            for c, d in zip(gcols, stored_digits):
                ov_d = overlaps_all.get(d, 0)
                border = "3px solid #4ade80" if d == guessed else "1px solid #1e293b"
                c.image(to_pil(all_patterns[d], 84),
                        caption=f"{'→ ' if d==guessed else ''}{d}  ({ov_d:.2f})",
                        use_container_width=True)

        # Persist for other tabs
        st.session_state["last_recall"] = {
            "final": final, "noisy": noisy, "energy": energy,
            "sampen": sampen, "net": net, "input_pattern": input_pattern,
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SIDE-BY-SIDE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Standard vs Inhibitory — same noisy input, two architectures")
    st.caption("Reproduces the core comparison from §4.1 and §4.2 of the paper")

    if input_mode != "MNIST + noise":
        st.info("Switch to **MNIST + noise** in the sidebar to use this tab.")
    else:
        run_cmp = st.button("▶  Run Comparison", type="primary", key="btn_cmp")
        if run_cmp:
            patterns = [all_patterns[d] for d in stored_digits]
            noisy_cmp = add_noise(all_patterns[test_digit], noise_level, seed=42)

            col_s, col_i = st.columns(2)

            for col, frac, label, accent in [
                (col_s, 0.0, "Standard (Symmetric)", BLUE),
                (col_i, 0.2, "Inhibitory 20%", PURPLE),
            ]:
                with col:
                    st.markdown(f"#### {label}")
                    net_c = HopfieldNetwork(N_NEURONS, inhibitory_fraction=frac)
                    net_c.train(patterns)
                    rc = net_c.recall(noisy_cmp.copy(), max_steps=max_steps, seed=42)
                    fs, eh = rc["state"], rc["energy"]

                    st.image(to_pil(fs), caption="Recalled", use_container_width=True)

                    ov_c = overlap(fs, all_patterns[test_digit])
                    se_c = sample_entropy(eh)
                    rec_c = ov_c - overlap(noisy_cmp, all_patterns[test_digit])

                    metric_card(st, "Overlap", f"{ov_c:.3f}",
                                GREEN if ov_c > 0.8 else YELLOW if ov_c > 0.4 else RED)
                    m1c, m2c = st.columns(2)
                    metric_card(m1c, "Recovery Δ", f"{rec_c:+.3f}",
                                GREEN if rec_c > 0 else RED)
                    metric_card(m2c, "SampEn", f"{se_c:.5f}",
                                RED if se_c > 0.010 else accent)

                    # Energy
                    fig_c = make_fig(title="Energy", height=220,
                                     xaxis_title="step", yaxis_title="E")
                    fig_c.add_trace(go.Scatter(
                        y=eh, mode="lines", line=dict(color=accent, width=1.2),
                        hovertemplate="Step %{x}<br>E=%{y:.1f}",
                    ))
                    st.plotly_chart(fig_c, use_container_width=True)

            st.markdown("---")
            st.markdown(
                '<div class="explain-box">'
                '<b>Trade-off:</b> The standard network follows a monotonic gradient '
                'descent (low SampEn ≈ 0.007) and achieves higher recall fidelity. '
                'The inhibitory network breaks Lyapunov symmetry via Dale\'s Principle '
                '— accuracy drops but trajectory complexity increases ~12%, sustaining '
                'the <em>ceaseless dynamics</em> of living neural circuits '
                '(Larremore et al. 2014).</div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PHASE SPACE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Phase space: Overlap vs Sample Entropy")
    st.caption("Each dot = one trial.  ✕ = reference centroids from the paper (§4.2)")

    if input_mode != "MNIST + noise":
        st.info("Switch to **MNIST + noise** mode.")
    else:
        c_n, _ = st.columns([1, 3])
        with c_n:
            n_trials_ps = st.slider("Trials per network", 5, 30, 15, key="ps_trials")
        run_ps = st.button("▶  Run Phase Space Experiment", type="primary", key="btn_ps")

        if run_ps:
            patterns = [all_patterns[d] for d in stored_digits]
            data_ps = {0.0: [], 0.2: []}
            bar = st.progress(0, text="Running ensemble…")
            total = 2 * n_trials_ps
            done = 0

            for frac in [0.0, 0.2]:
                net_ps = HopfieldNetwork(N_NEURONS, inhibitory_fraction=frac)
                net_ps.train(patterns)
                for t in range(n_trials_ps):
                    n_t = add_noise(all_patterns[test_digit], noise_level, seed=t * 7)
                    r = net_ps.recall(n_t, max_steps=max_steps, seed=t * 7 + 1)
                    ov_t = overlap(r["state"], all_patterns[test_digit])
                    se_t = sample_entropy(r["energy"])
                    if ov_t > 0:
                        data_ps[frac].append((ov_t, se_t))
                    done += 1
                    bar.progress(done / total, text=f"Trial {done}/{total}")
            bar.empty()

            fig_ps = make_fig(
                title="Accuracy vs Dynamical Complexity",
                xaxis_title="Final overlap (accuracy)",
                yaxis_title="Sample Entropy (complexity)",
                height=480,
            )
            for frac, color, name in [(0.0, BLUE, "Standard"), (0.2, PURPLE, "Inhibitory (20%)")]:
                pts = data_ps[frac]
                if pts:
                    xs, ys = zip(*pts)
                    fig_ps.add_trace(go.Scatter(
                        x=xs, y=ys, mode="markers", name=name,
                        marker=dict(color=color, size=8, opacity=0.6),
                        hovertemplate="Overlap: %{x:.3f}<br>SampEn: %{y:.5f}",
                    ))

            # Paper centroids
            fig_ps.add_trace(go.Scatter(
                x=[CENTROID_STD[0]], y=[CENTROID_STD[1]], mode="markers",
                name="Centroid Standard (paper)",
                marker=dict(symbol="x", size=16, color="#93c5fd",
                            line=dict(width=2, color="white")),
            ))
            fig_ps.add_trace(go.Scatter(
                x=[CENTROID_INH[0]], y=[CENTROID_INH[1]], mode="markers",
                name="Centroid Inhibitory (paper)",
                marker=dict(symbol="x", size=16, color="#d8b4fe",
                            line=dict(width=2, color="white")),
            ))
            fig_ps.add_vline(x=0.7, line_dash="dash", line_color="#475569",
                             annotation_text="success threshold", annotation_font_color="#64748b")
            st.plotly_chart(fig_ps, use_container_width=True)

            st.markdown(
                '<div class="explain-box">'
                'Standard (blue) clusters in high-accuracy / low-entropy — '
                'predictable gradient descent to deep minima. '
                'Inhibitory (purple) shifts left and up — lower accuracy but '
                'higher complexity. The ✕ markers are the paper\'s reference centroids '
                'from a 20-trial ensemble.</div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — STATE-SPACE TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### State-space trajectory during recall")
    st.caption("2D projection: overlap with two competing stored patterns over time (cf. Fig. 4)")

    if input_mode != "MNIST + noise":
        st.info("Switch to **MNIST + noise** mode.")
    elif len(stored_digits) < 2:
        st.info("Store at least 2 patterns to see competing trajectories.")
    else:
        ca, cb = st.columns(2)
        with ca:
            digit_a = st.selectbox("Pattern A (x-axis)", stored_digits, index=0, key="traj_a")
        with cb:
            rest = [d for d in stored_digits if d != digit_a]
            digit_b = st.selectbox("Pattern B (y-axis)", rest, index=0, key="traj_b")

        run_traj = st.button("▶  Run Trajectory Analysis", type="primary", key="btn_traj")

        if run_traj:
            patterns = [all_patterns[d] for d in stored_digits]
            noisy_traj = add_noise(all_patterns[test_digit], noise_level, seed=42)
            pat_a = all_patterns[digit_a]
            pat_b = all_patterns[digit_b]

            fig_traj = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Standard (Symmetric)", "Inhibitory 20%"],
                horizontal_spacing=0.08,
            )
            fig_traj.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.8)",
                font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=11),
                margin=dict(l=50, r=25, t=55, b=50),
                height=480,
                showlegend=False,
            )

            for idx_plot, (frac, accent) in enumerate([(0.0, BLUE), (0.2, PURPLE)]):
                net_t = HopfieldNetwork(N_NEURONS, inhibitory_fraction=frac)
                net_t.train(patterns)
                rt = net_t.recall(
                    noisy_traj.copy(), max_steps=max_steps,
                    trajectory_patterns=(pat_a, pat_b), seed=42,
                )
                ta, tb = rt["traj_a"], rt["traj_b"]
                n_pts = len(ta)
                col_idx = idx_plot + 1

                # Color by time: build segments
                cmap = px.colors.sample_colorscale("Plasma",
                                                   np.linspace(0.1, 0.9, n_pts))

                for i in range(n_pts - 1):
                    fig_traj.add_trace(go.Scatter(
                        x=ta[i:i+2], y=tb[i:i+2], mode="lines",
                        line=dict(color=cmap[i], width=2.5),
                        showlegend=False,
                        hovertemplate=f"Step ~{i*20}<br>O_a=%{{x:.3f}}<br>O_b=%{{y:.3f}}",
                    ), row=1, col=col_idx)

                # Start / End / Prototypes
                fig_traj.add_trace(go.Scatter(
                    x=[ta[0]], y=[tb[0]], mode="markers",
                    marker=dict(size=12, color=RED, symbol="circle"),
                    name="Start",
                ), row=1, col=col_idx)
                fig_traj.add_trace(go.Scatter(
                    x=[ta[-1]], y=[tb[-1]], mode="markers",
                    marker=dict(size=14, color="white", symbol="star"),
                    name="End",
                ), row=1, col=col_idx)
                ov_ab = overlap(pat_a, pat_b)
                fig_traj.add_trace(go.Scatter(
                    x=[1.0], y=[ov_ab], mode="markers",
                    marker=dict(size=10, color=GREEN),
                    name=f"Perfect '{digit_a}'",
                ), row=1, col=col_idx)
                fig_traj.add_trace(go.Scatter(
                    x=[ov_ab], y=[1.0], mode="markers",
                    marker=dict(size=10, color=YELLOW),
                    name=f"Perfect '{digit_b}'",
                ), row=1, col=col_idx)

                fig_traj.update_xaxes(title_text=f"Overlap '{digit_a}'", row=1, col=col_idx)
                fig_traj.update_yaxes(title_text=f"Overlap '{digit_b}'", row=1, col=col_idx)

            st.plotly_chart(fig_traj, use_container_width=True)

            st.markdown(
                '<div class="explain-box">'
                'Path colour goes from dark (start) to bright (end). '
                '<b>Standard:</b> sustained trajectory driving straight toward the target attractor. '
                '<b>Inhibitory:</b> constrained, wandering path — structural frustration from '
                'Dale\'s Principle prevents the full descent into the basin. '
                'Red dot = noisy start, white star = final state.</div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — EXPERIMENTS (NOISE SWEEP + CAPACITY)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    exp_a, exp_b = st.tabs(["📈 Noise Robustness", "📉 Memory Capacity"])

    # ── 5A: Noise sweep ──────────────────────────────────────────────────
    with exp_a:
        st.markdown("### Recall accuracy vs noise level")
        st.caption("Reproduces Fig. 2 — phase transition from memory to paramagnetic phase")

        if input_mode != "MNIST + noise":
            st.info("Switch to **MNIST + noise** mode.")
        else:
            c1e, c2e = st.columns(2)
            with c1e:
                n_trials_sw = st.slider("Trials per point", 3, 20, 5, key="sw_trials")
            with c2e:
                n_levels = st.slider("Noise levels", 5, 20, 11, key="sw_levels")

            run_sw = st.button("▶  Run Noise Sweep", type="primary", key="btn_sw")
            if run_sw:
                patterns = [all_patterns[d] for d in stored_digits]
                noise_vals = np.linspace(0.0, 0.5, n_levels)
                sweep_data = {0.0: [], 0.2: []}
                total = 2 * n_levels * n_trials_sw
                bar = st.progress(0, text="Sweeping…")
                done = 0

                for frac in [0.0, 0.2]:
                    net_sw = HopfieldNetwork(N_NEURONS, inhibitory_fraction=frac)
                    net_sw.train(patterns)
                    for eta in noise_vals:
                        ovs = []
                        for t in range(n_trials_sw):
                            nt = add_noise(all_patterns[test_digit], eta, seed=t * 13)
                            r = net_sw.recall(nt, max_steps=max_steps,
                                              seed=t * 13 + 1)
                            ovs.append(overlap(r["state"], all_patterns[test_digit]))
                            done += 1
                            bar.progress(done / total, text=f"{done}/{total}")
                        sweep_data[frac].append((eta, np.mean(ovs), np.std(ovs)))
                bar.empty()

                fig_sw = make_fig(
                    title="Experiment 1: Robustness to Noise",
                    xaxis_title="Noise η (fraction of flipped bits)",
                    yaxis_title="Recall accuracy (overlap)",
                    height=430,
                    yaxis=dict(range=[-1.1, 1.1]),
                )
                for frac, color, name in [(0.0, BLUE, "Standard"),
                                          (0.2, PURPLE, "Inhibitory (20%)")]:
                    etas = [r[0] for r in sweep_data[frac]]
                    means = np.array([r[1] for r in sweep_data[frac]])
                    stds = np.array([r[2] for r in sweep_data[frac]])
                    fig_sw.add_trace(go.Scatter(
                        x=etas, y=means, mode="lines+markers", name=name,
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                        hovertemplate="η=%{x:.2f}<br>O=%{y:.3f}",
                    ))
                    fig_sw.add_trace(go.Scatter(
                        x=list(etas) + list(etas)[::-1],
                        y=list(means + stds) + list(means - stds)[::-1],
                        fill="toself", fillcolor=hex_alpha(color),
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))
                fig_sw.add_vline(x=0.40, line_dash="dash", line_color=RED,
                                 annotation_text="ηc ≈ 0.40")
                fig_sw.add_hline(y=0, line_dash="dot", line_color="#475569")
                st.plotly_chart(fig_sw, use_container_width=True)

                st.markdown(
                    '<div class="explain-box">'
                    'Phase transition at η_c ≈ 0.40: below this threshold the '
                    'network corrects errors; above it, recall collapses. '
                    'Standard network shows a sharper transition and higher peak accuracy. '
                    'Inhibitory network degrades more gradually — biological asymmetry '
                    'provides flexibility at the cost of absolute fidelity.</div>',
                    unsafe_allow_html=True,
                )

    # ── 5B: Capacity ─────────────────────────────────────────────────────
    with exp_b:
        st.markdown("### Memory capacity — recall accuracy vs stored patterns")
        st.caption("Reproduces Fig. 5 — catastrophic forgetting with correlated MNIST patterns")

        c1cap, c2cap = st.columns(2)
        with c1cap:
            n_trials_cap = st.slider("Trials per point", 3, 15, 5, key="cap_trials")
        with c2cap:
            fixed_noise = st.slider("Fixed noise η", 0.0, 0.3, 0.10, 0.05,
                                    key="cap_noise", format="%.2f")
        max_M = min(10, len(stored_digits))
        if max_M < 2:
            st.info("Store at least 2 digits to test capacity.")
        else:
            run_cap = st.button("▶  Run Capacity Experiment", type="primary", key="btn_cap")
            if run_cap:
                cap_data = {0.0: [], 0.2: []}
                total = 2 * max_M * n_trials_cap
                bar = st.progress(0, text="Testing capacity…")
                done = 0

                for frac in [0.0, 0.2]:
                    for M in range(1, max_M + 1):
                        pats = [all_patterns[stored_digits[i]] for i in range(M)]
                        net_cap = HopfieldNetwork(N_NEURONS, inhibitory_fraction=frac)
                        net_cap.train(pats)
                        ovs = []
                        for t in range(n_trials_cap):
                            tgt_idx = t % M
                            tgt = pats[tgt_idx]
                            nt = add_noise(tgt, fixed_noise, seed=t * 11 + M)
                            r = net_cap.recall(nt, max_steps=max_steps,
                                               seed=t * 11 + M + 1)
                            ovs.append(overlap(r["state"], tgt))
                            done += 1
                            bar.progress(done / total, text=f"{done}/{total}")
                        cap_data[frac].append((M, np.mean(ovs), np.std(ovs)))
                bar.empty()

                fig_cap = make_fig(
                    title="Experiment 3: Memory Capacity",
                    xaxis_title="Number of stored patterns M",
                    yaxis_title="Average recall accuracy",
                    height=430,
                    xaxis=dict(dtick=1),
                )
                for frac, color, name in [(0.0, BLUE, "Standard"),
                                          (0.2, PURPLE, "Inhibitory (20%)")]:
                    Ms = [r[0] for r in cap_data[frac]]
                    means = np.array([r[1] for r in cap_data[frac]])
                    stds = np.array([r[2] for r in cap_data[frac]])
                    fig_cap.add_trace(go.Scatter(
                        x=Ms, y=means, mode="lines+markers", name=name,
                        line=dict(color=color, width=2), marker=dict(size=7),
                        hovertemplate="M=%{x}<br>Accuracy=%{y:.3f}",
                    ))
                    fig_cap.add_trace(go.Scatter(
                        x=list(Ms) + list(Ms)[::-1],
                        y=list(means + stds) + list(means - stds)[::-1],
                        fill="toself",
                        fillcolor=hex_alpha(color),
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))

                # Theoretical limit annotation
                fig_cap.add_annotation(
                    x=max_M, y=0.05, text=f"Theoretical limit: M ≈ 0.138N = 108",
                    showarrow=False, font=dict(size=10, color="#64748b"),
                )
                st.plotly_chart(fig_cap, use_container_width=True)

                st.markdown(
                    '<div class="explain-box">'
                    'The theoretical Hopfield capacity is M ≈ 0.138N ≈ 108 patterns '
                    'for N=784, but this assumes random, orthogonal patterns. '
                    'MNIST digits share significant spatial structure, so catastrophic '
                    'forgetting begins at M ≈ 3. The symmetric network maintains deeper '
                    'basins (higher capacity), while inhibitory asymmetry reduces basin '
                    'depth and increases susceptibility to spurious states.</div>',
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — WEIGHT MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### Synaptic weight matrix W")
    st.caption("Structural comparison: symmetric vs asymmetric architecture (cf. Fig. 1)")

    run_wm = st.button("▶  Visualise Weight Matrices", type="primary", key="btn_wm")

    if run_wm:
        patterns = [all_patterns[d] for d in stored_digits]

        fig_wm = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Standard (Symmetric)", "Inhibitory 20% (Asymmetric)"],
            horizontal_spacing=0.06,
        )
        fig_wm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.8)",
            font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=11),
            margin=dict(l=50, r=25, t=55, b=50),
            height=500,
        )

        for idx_wm, (frac, label) in enumerate([(0.0, "Standard"), (0.2, "Inhibitory")]):
            net_wm = HopfieldNetwork(N_NEURONS, inhibitory_fraction=frac)
            net_wm.train(patterns)

            # Downsample for performance: show every 8th neuron
            step = 8
            W_sub = net_wm.weights[::step, ::step]
            fig_wm.add_trace(go.Heatmap(
                z=W_sub, colorscale="RdYlGn", zmin=-1, zmax=1,
                showscale=(idx_wm == 1),
                hovertemplate=f"{label}<br>i=%{{y}}, j=%{{x}}<br>w=%{{z:.4f}}",
            ), row=1, col=idx_wm + 1)
            fig_wm.update_xaxes(title_text="Neuron j", row=1, col=idx_wm + 1)
            fig_wm.update_yaxes(title_text="Neuron i", row=1, col=idx_wm + 1)

        st.plotly_chart(fig_wm, use_container_width=True)

        st.markdown(
            '<div class="explain-box">'
            '<b>Left (Standard):</b> perfectly symmetric across the diagonal '
            '(w<sub>ij</sub> = w<sub>ji</sub>) — an undirected graph. '
            'W is a valid Lyapunov function.<br><br>'
            '<b>Right (Inhibitory):</b> columns corresponding to inhibitory neurons '
            'are flipped to all-negative (visible as vertical stripes). '
            'This breaks w<sub>ij</sub> = w<sub>ji</sub>, converting the network '
            'into a directed graph and removing the convergence guarantee.<br><br>'
            '<i>Showing every 8th neuron for performance. Full matrix is 784×784.</i></div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div class="metric-label" style="text-align:center">'
    'Hopfield Networks · Hebbian Learning · Dale\'s Principle · '
    'Mellini (2026) — Physical Methods of Biology, UNIBO</div>',
    unsafe_allow_html=True,
)
