import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import io
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hopfield import HopfieldNetwork
from utils import calculate_overlap, add_noise, pattern_to_image, calculate_sampen
from data import load_mnist_patterns

# ── Reference centroids (20-trial ensemble on MNIST digits 0,1,2,8) ─────────
CENTROID_STD  = (0.834, 0.00749)
CENTROID_INH  = (0.528, 0.00840)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neural Memory Lab", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { font-family: 'Syne', sans-serif; font-weight: 800; color: #e2e8f0; letter-spacing: -1px; }
    h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; color: #94a3b8; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #0f172a; padding: 4px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px; color: #64748b;
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; padding: 6px 14px;
    }
    .stTabs [aria-selected="true"] { background: #1e293b !important; color: #e2e8f0 !important; }
    .card {
        background: #0f172a; border: 1px solid #1e293b; border-radius: 12px;
        padding: 1.2rem; margin-bottom: 0.8rem;
    }
    .metric-big { font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
    .metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; font-weight: 700;
    }
    .badge-green  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
    .badge-yellow { background: #1c1500; color: #facc15; border: 1px solid #713f12; }
    .badge-red    { background: #1f0a0a; color: #f87171; border: 1px solid #7f1d1d; }
    .badge-blue   { background: #0a1628; color: #60a5fa; border: 1px solid #1d4ed8; }
    .explain-box {
        background: linear-gradient(135deg, #0f172a, #1a1035);
        border: 1px solid #6d28d9; border-radius: 10px; padding: 1rem 1.2rem;
        font-size: 0.88rem; color: #c4b5fd; line-height: 1.6;
    }
    .warn-box {
        background: #1c1500; border: 1px solid #713f12; border-radius: 8px;
        padding: 0.6rem 1rem; font-size: 0.82rem; color: #facc15;
    }
    div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def get_patterns():
    return load_mnist_patterns()

def to_pil(pattern, size=196):
    arr = pattern_to_image(pattern)
    return Image.fromarray(arr, mode="L").resize((size, size), Image.NEAREST)

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def dark_fig(w=6, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#475569")
    for sp in ax.spines.values(): sp.set_edgecolor("#1e293b")
    return fig, ax

def recall_frames(net, noisy, steps, every=8):
    state = np.ascontiguousarray(noisy, dtype=np.float64).flatten()
    W = np.ascontiguousarray(net.weights, dtype=np.float64)
    N = W.shape[0]
    frames = [to_pil(state)]
    for s in range(steps):
        idx = int(np.random.randint(0, N))
        lf  = float(W[idx].dot(state))
        state[idx] = 1.0 if lf >= 0.0 else -1.0
        if s % every == 0:
            frames.append(to_pil(state))
    return frames, state

def to_gif(frames, duration=60):
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:], loop=0, duration=duration)
    buf.seek(0)
    return buf.read()

def canvas_to_pattern(canvas_result):
    r = canvas_result.image_data[:, :, 0].astype(np.uint8)
    gray_28 = Image.fromarray(r, mode="L").resize((28, 28), Image.LANCZOS)
    arr = np.array(gray_28)
    return np.where(arr > 30, 1.0, -1.0).astype(np.float64)

def metric_card(col, label, value, color="#60a5fa"):
    col.markdown(
        f'<div class="card" style="text-align:center">'
        f'<div class="metric-big" style="color:{color}">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True)

def interpret_result(overlap, sampen, inhibitory, n_patterns):
    """Generate physics-grounded explanation of the recall result."""
    if overlap < 0:
        return ("⚠️ **Anti-pattern convergence.** The network settled into the inverted attractor "
                "−ξ, a spurious minimum that always exists due to the even symmetry of the Hebbian "
                "Hamiltonian. This is more likely with high noise or many stored patterns.")
    alpha = n_patterns / 784
    lines = []
    if overlap > 0.85:
        lines.append("✅ **Deep attractor convergence.** The recall settled cleanly inside the target "
                     "basin of attraction — consistent with low-noise, well-separated patterns.")
    elif overlap > 0.5:
        lines.append("🟡 **Partial convergence.** The network moved toward the target but did not reach "
                     "the prototype. This can indicate proximity to the phase transition threshold "
                     f"(η_c ≈ 0.40) or interference from competing patterns.")
    else:
        lines.append("🔴 **Spurious attractor.** The recalled state has low similarity to any stored "
                     "pattern. Likely causes: noise above critical threshold, or memory overload "
                     f"(current α = {alpha:.4f}).")

    if inhibitory:
        if sampen > 0.010:
            lines.append(f"🌀 **Ceaseless dynamics detected** (SampEn = {sampen:.5f} > 0.010). "
                         "Dale's Principle is breaking the Lyapunov constraint — the network is "
                         "exhibiting complex trajectory behaviour with no fixed-point guarantee.")
        else:
            lines.append(f"SampEn = {sampen:.5f}. Inhibitory network but trajectory is still "
                         "relatively regular — frustration is present but not dominant for this input.")
    else:
        lines.append(f"SampEn = {sampen:.5f} — monotonic energy descent toward a fixed-point attractor, "
                     f"consistent with symmetric Hebbian dynamics.")

    if alpha > 0.004:
        lines.append(f"⚠️ Memory load α = {alpha:.4f} ({n_patterns} patterns / 784 neurons). "
                     "MNIST digits share structural overlap — interference between stored patterns "
                     "increases significantly beyond M = 3.")
    return "\n\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Neural Memory Lab")
    st.markdown("<div class='metric-label'>Hopfield Network Explorer</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Memory patterns**")
    stored_digits = st.multiselect("Digits to store", options=list(range(10)),
                                   default=[0, 1, 3, 5, 8], label_visibility="collapsed")

    n_pat = len(stored_digits)
    alpha = n_pat / 784
    alpha_color = "#f87171" if n_pat >= 4 else "#facc15" if n_pat == 3 else "#4ade80"
    st.markdown(
        f'<div class="card" style="text-align:center; padding:0.6rem">'
        f'<span style="font-family:JetBrains Mono; color:{alpha_color}; font-size:1.2rem; font-weight:700">'
        f'α = {alpha:.4f}</span><br>'
        f'<span class="metric-label">load = {n_pat} / 784 neurons</span></div>',
        unsafe_allow_html=True)
    if n_pat >= 4:
        st.markdown('<div class="warn-box">⚠️ MNIST patterns share structure — catastrophic forgetting expected at M ≥ 3</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Input**")
    input_mode = st.radio("Mode", ["MNIST + noise", "Draw digit"], label_visibility="collapsed")

    if input_mode == "MNIST + noise":
        test_digit = st.selectbox("Digit to recall", options=stored_digits if stored_digits else [0])
        noise_pct  = st.slider("Noise level", 0, 50, 25, 5, format="%d%%")
        noise_level = noise_pct / 100.0
    else:
        test_digit  = None
        noise_pct   = 0
        noise_level = 0.0

    st.markdown("---")
    st.markdown("**Network**")
    inhibitory = st.checkbox("Inhibitory neurons (20% — Dale's Principle)", value=False)
    inhibitory_fraction = 0.2 if inhibitory else 0.0
    max_steps = st.slider("Recall steps", 200, 2000, 1000, 100)
    animate   = st.checkbox("Animate recall (GIF)", value=True)

    st.markdown("---")
    run_btn = st.button("▶  Run Recall", type="primary", use_container_width=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading MNIST…"):
    all_patterns = get_patterns()

if not stored_digits:
    st.warning("Select at least one digit to store.")
    st.stop()

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Neural Memory Lab")
st.markdown("<div class='metric-label' style='margin-bottom:1rem'>Hopfield Networks · Hebbian Learning · Dale's Principle</div>",
            unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_recall, tab_sidebyside, tab_phasespace, tab_statespace, tab_weights, tab_sweep = st.tabs([
    "⚡ Recall", "⚖️ Side-by-Side", "📡 Phase Space", "🌀 State Space", "🔬 Weight Matrix", "📈 Noise Sweep"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECALL
# ══════════════════════════════════════════════════════════════════════════════
with tab_recall:
    c1, c2, c3 = st.columns(3)

    # Input
    with c1:
        st.markdown("### ① Input")
        if input_mode == "MNIST + noise":
            st.image(to_pil(all_patterns[test_digit]),
                     caption=f"MNIST prototype '{test_digit}'", use_container_width=True)
            input_pattern = all_patterns[test_digit].copy()
        else:
            try:
                from streamlit_drawable_canvas import st_canvas
                cr = st_canvas(stroke_width=18, stroke_color="#FFFFFF",
                               background_color="#000000", height=196, width=196,
                               drawing_mode="freedraw", key="canvas_t1")
                input_pattern = canvas_to_pattern(cr) if cr.image_data is not None else None
                if input_pattern is None:
                    st.info("Draw above, then press Run Recall.")
            except ImportError:
                st.error("streamlit-drawable-canvas not installed.")
                input_pattern = None

    with c2:
        st.markdown("### ② Corrupted")
        if not run_btn:
            st.info("Press ▶ Run Recall")
    with c3:
        st.markdown("### ③ Recalled")
        if not run_btn:
            st.info("Results will appear here")

    if run_btn:
        if input_mode == "Draw digit" and input_pattern is None:
            st.warning("Draw something first!")
            st.stop()

        patterns = [all_patterns[d] for d in stored_digits]
        with st.spinner("Training network…"):
            net = HopfieldNetwork(784, inhibitory_fraction=inhibitory_fraction)
            net.train(patterns)

        noisy = (add_noise(input_pattern, noise_level, seed=42)
                 if input_mode == "MNIST + noise"
                 else np.ascontiguousarray(input_pattern, dtype=np.float64).flatten())

        with c2:
            st.image(to_pil(noisy),
                     caption=f"{'Noise: ' + str(noise_pct) + '%' if input_mode == 'MNIST + noise' else 'Hand-drawn input'}",
                     use_container_width=True)

        if animate:
            with st.spinner("Generating recall animation…"):
                frames, final_state = recall_frames(net, noisy, max_steps)
            with c3:
                st.image(to_gif(frames), caption="Asynchronous recall", use_container_width=True)
        else:
            with st.spinner("Running recall…"):
                final_state, energy_history = net.recall(noisy, max_steps=max_steps, record_energy=True)
            with c3:
                st.image(to_pil(final_state), caption="Recalled attractor", use_container_width=True)

        # Energy + SampEn
        if not animate:
            _, energy_history = net.recall(noisy, max_steps=max_steps, record_energy=True)
        else:
            _, energy_history = net.recall(noisy, max_steps=max_steps, record_energy=True)

        sampen = calculate_sampen(energy_history)

        # Guess
        overlaps_all = {d: calculate_overlap(final_state, all_patterns[d]) for d in stored_digits}
        guessed = max(overlaps_all, key=overlaps_all.get)
        best_ov  = overlaps_all[guessed]

        # Anti-pattern check
        is_anti = best_ov < 0

        st.markdown("---")

        if input_mode == "Draw digit":
            conf_badge = ("badge-green" if best_ov > 0.7 else
                          "badge-yellow" if best_ov > 0.4 else "badge-red")
            conf_text  = ("High" if best_ov > 0.7 else
                          "Medium" if best_ov > 0.4 else "Low")
            st.markdown(
                f'<div class="card" style="text-align:center; padding:1.5rem">'
                f'<div class="metric-label" style="margin-bottom:6px">Network guess</div>'
                f'<div style="font-size:5rem; font-weight:800; font-family:JetBrains Mono; '
                f'color:#818cf8; line-height:1">'
                f'{"?" if is_anti else guessed}</div>'
                f'<div style="margin-top:8px">'
                f'<span class="badge {conf_badge}">{conf_text} confidence</span>'
                f'</div>'
                f'<div class="metric-label" style="margin-top:6px">overlap = {best_ov:.3f}</div>'
                f'</div>', unsafe_allow_html=True)

        # Metric cards
        overlap_v = calculate_overlap(final_state, input_pattern)
        recovery  = overlap_v - calculate_overlap(noisy, input_pattern)

        m1, m2, m3, m4, m5 = st.columns(5)
        metric_card(m1, "Stored patterns",   str(n_pat),             "#818cf8")
        metric_card(m2, "Guessed digit",     "?" if is_anti else str(guessed), "#818cf8")
        metric_card(m3, "Overlap",           f"{overlap_v:.3f}",
                    "#4ade80" if overlap_v > 0.8 else "#facc15" if overlap_v > 0.4 else "#f87171")
        metric_card(m4, "Recovery Δ",        f"{recovery:+.3f}",
                    "#4ade80" if recovery > 0 else "#f87171")
        metric_card(m5, "SampEn",            f"{sampen:.5f}",
                    "#f87171" if sampen > 0.010 else "#60a5fa")

        # Energy plot
        if not animate:
            st.markdown("#### 📉 Energy landscape")
            fig, ax = dark_fig(8, 2.8)
            ax.plot(energy_history, color="#818cf8", lw=1.0)
            ax.set_xlabel("Update step", color="#64748b", fontsize=9)
            ax.set_ylabel("E", color="#64748b", fontsize=9)
            ax.set_title("Energy during recall", color="#94a3b8", fontsize=10)
            st.image(fig_to_pil(fig), use_container_width=True)

        # Explain
        st.markdown("#### 🔬 Physics interpretation")
        explanation = interpret_result(overlap_v, sampen, inhibitory, n_pat)
        st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

        # Gallery
        st.markdown("---")
        st.markdown("#### 🗃️ Stored patterns")
        gcols = st.columns(len(stored_digits))
        for col, d in zip(gcols, stored_digits):
            col.image(to_pil(all_patterns[d], 90), caption=f"Digit {d}", use_container_width=True)

        st.session_state["last_result"] = {
            "final_state": final_state, "noisy": noisy,
            "overlap": overlap_v, "sampen": sampen,
            "net": net, "input_pattern": input_pattern,
            "inhibitory": inhibitory, "energy": energy_history
        }

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIDE BY SIDE
# ══════════════════════════════════════════════════════════════════════════════
with tab_sidebyside:
    st.markdown("### Standard vs Inhibitory — same input, two networks")
    st.markdown("<div class='metric-label'>Same noisy input run through both architectures simultaneously</div>",
                unsafe_allow_html=True)
    st.markdown("")

    if input_mode != "MNIST + noise":
        st.info("Switch to **MNIST + noise** mode in the sidebar to use this tab.")
    else:
        run_sbs = st.button("▶  Run Side-by-Side Comparison", type="primary")

        if run_sbs:
            patterns = [all_patterns[d] for d in stored_digits]
            noisy_sbs = add_noise(all_patterns[test_digit], noise_level, seed=42)

            col_std, col_inh = st.columns(2)

            for col, frac, label, accent in [
                (col_std, 0.0, "Standard (Symmetric)", "#60a5fa"),
                (col_inh, 0.2, "Inhibitory 20% — Dale's Principle", "#c084fc"),
            ]:
                with col:
                    st.markdown(f"#### {label}")
                    net_i = HopfieldNetwork(784, inhibitory_fraction=frac)
                    net_i.train(patterns)

                    with st.spinner(f"Recalling ({label})…"):
                        fs, eh = net_i.recall(noisy_sbs.copy(), max_steps=max_steps, record_energy=True)

                    ov  = calculate_overlap(fs, all_patterns[test_digit])
                    se  = calculate_sampen(eh)
                    rec = ov - calculate_overlap(noisy_sbs, all_patterns[test_digit])

                    st.image(to_pil(fs), caption="Recalled pattern", use_container_width=True)

                    st.markdown(
                        f'<div class="card" style="text-align:center">'
                        f'<span class="metric-big" style="color:{accent}">{ov:.3f}</span><br>'
                        f'<span class="metric-label">Final overlap</span></div>',
                        unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    metric_card(m1, "Recovery Δ", f"{rec:+.3f}",
                                "#4ade80" if rec > 0 else "#f87171")
                    metric_card(m2, "SampEn", f"{se:.5f}",
                                "#f87171" if se > 0.010 else accent)

                    # Energy
                    fig, ax = dark_fig(5, 2.5)
                    ax.plot(eh, color=accent, lw=1.0)
                    ax.set_title("Energy", color="#94a3b8", fontsize=9)
                    ax.set_xlabel("step", color="#64748b", fontsize=8)
                    st.image(fig_to_pil(fig), use_container_width=True)

            # Summary comparison
            st.markdown("---")
            st.markdown("#### Key trade-off")
            st.markdown(
                '<div class="explain-box">'
                'Standard networks achieve higher retrieval accuracy by following a monotonic '
                'gradient descent (low SampEn ≈ 0.007). Inhibitory networks break the Lyapunov '
                'symmetry via Dale\'s Principle, reducing accuracy but increasing trajectory '
                'complexity — sustaining the <em>ceaseless dynamics</em> characteristic '
                'of living neural circuits (Larremore et al. 2014).'
                '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PHASE SPACE
# ══════════════════════════════════════════════════════════════════════════════
with tab_phasespace:
    st.markdown("### Phase space: Overlap vs Sample Entropy")
    st.markdown("<div class='metric-label'>Each point = one recall trial. ✕ markers = reference centroids from a 20-trial ensemble</div>",
                unsafe_allow_html=True)
    st.markdown("")

    run_ps = st.button("▶  Run Phase Space Experiment (20 trials)", type="primary")

    if run_ps:
        if input_mode != "MNIST + noise":
            st.info("Switch to MNIST + noise mode.")
        else:
            patterns = [all_patterns[d] for d in stored_digits]
            results = {0.0: [], 0.2: []}

            progress = st.progress(0)
            total = 40
            done  = 0

            for frac in [0.0, 0.2]:
                net_ps = HopfieldNetwork(784, inhibitory_fraction=frac)
                net_ps.train(patterns)
                for trial in range(20):
                    seed = trial * 7
                    noisy_t = add_noise(all_patterns[test_digit], noise_level, seed=seed)
                    fs, eh  = net_ps.recall(noisy_t, max_steps=max_steps, record_energy=True)
                    ov = calculate_overlap(fs, all_patterns[test_digit])
                    se = calculate_sampen(eh)
                    if ov > 0:
                        results[frac].append((ov, se))
                    done += 1
                    progress.progress(done / total)

            fig, ax = dark_fig(8, 5)

            colors = {0.0: "#60a5fa", 0.2: "#c084fc"}
            labels = {0.0: "Standard", 0.2: "Inhibitory (20%)"}

            for frac, pts in results.items():
                if pts:
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, c=colors[frac], alpha=0.5, s=40,
                               label=labels[frac], edgecolors="none")

            # Reference centroids
            ax.scatter(*CENTROID_STD, marker="X", s=250, color="#93c5fd",
                       edgecolors="white", lw=1.5, zorder=5, label="Centroid — Standard")
            ax.scatter(*CENTROID_INH, marker="X", s=250, color="#d8b4fe",
                       edgecolors="white", lw=1.5, zorder=5, label="Centroid — Inhibitory")

            ax.axvline(0.7, color="#475569", lw=0.8, ls="--", alpha=0.6)
            ax.set_xlabel("Final overlap (recall accuracy)", color="#94a3b8", fontsize=10)
            ax.set_ylabel("Sample Entropy (trajectory complexity)", color="#94a3b8", fontsize=10)
            ax.set_title("Phase Space: Accuracy vs Dynamical Complexity", color="#e2e8f0", fontsize=11)
            ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=9)

            st.image(fig_to_pil(fig), use_container_width=True)

            st.markdown(
                '<div class="explain-box">'
                'Standard network (blue): clusters in high-accuracy / low-entropy region — '
                'predictable gradient descent toward fixed-point attractors. '
                'Inhibitory network (purple): shifts left and up — lower accuracy but higher '
                'dynamical complexity. ✕ markers show expected centroids from a 20-trial ensemble.'
                '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — STATE SPACE TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_statespace:
    st.markdown("### State space trajectory during recall")
    st.markdown("<div class='metric-label'>2D projection of S(t) onto two competing memory patterns — color = time (purple → yellow)</div>",
                unsafe_allow_html=True)
    st.markdown("")

    if input_mode != "MNIST + noise":
        st.info("Switch to MNIST + noise mode.")
    elif len(stored_digits) < 2:
        st.info("Store at least 2 patterns to see competing trajectories.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            digit_a = st.selectbox("Pattern A (x-axis)", stored_digits, index=0, key="ss_a")
        with c2:
            remaining = [d for d in stored_digits if d != digit_a]
            digit_b = st.selectbox("Pattern B (y-axis)", remaining, index=0, key="ss_b")

        run_ss = st.button("▶  Run Trajectory Analysis", type="primary")

        if run_ss:
            noisy_ss = add_noise(all_patterns[test_digit], noise_level, seed=42)
            patterns  = [all_patterns[d] for d in stored_digits]
            pat_a = all_patterns[digit_a]
            pat_b = all_patterns[digit_b]

            # Cross-overlaps for placing prototype markers correctly
            ov_aa = 1.0
            ov_bb = 1.0
            ov_ab = float(np.dot(pat_a, pat_b) / 784)  # overlap of pat_a projected onto pat_b axis
            ov_ba = ov_ab  # symmetric

            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="#0f172a")
            fig.subplots_adjust(wspace=0.35, bottom=0.22)

            for ax_i, (frac, title, accent) in enumerate([
                (0.0, "Standard  (Symmetric)", "#60a5fa"),
                (0.2, "Inhibitory  (20% Dale's Principle)", "#c084fc"),
            ]):
                ax = axes[ax_i]
                ax.set_facecolor("#0a0f1e")

                net_ss = HopfieldNetwork(784, inhibitory_fraction=frac)
                net_ss.train(patterns)
                fs, eh, traj_a, traj_b = net_ss.recall_with_trajectory(
                    noisy_ss.copy(), pat_a, pat_b, max_steps=max_steps, sample_every=10)

                n_pts = len(traj_a)

                # ── Trajectory colored by time ────────────────────────────
                for i in range(n_pts - 1):
                    t_norm = i / max(n_pts - 1, 1)
                    c_seg  = cm.plasma(0.1 + 0.8 * t_norm)
                    ax.plot(traj_a[i:i+2], traj_b[i:i+2],
                            color=c_seg, lw=1.8, alpha=0.85, solid_capstyle="round")

                # ── Directional arrows every ~15% of trajectory ───────────
                arrow_steps = max(1, n_pts // 7)
                for i in range(arrow_steps, n_pts - 1, arrow_steps):
                    dx = traj_a[i] - traj_a[i-1]
                    dy = traj_b[i] - traj_b[i-1]
                    if abs(dx) + abs(dy) > 1e-4:
                        t_norm = i / max(n_pts - 1, 1)
                        ax.annotate("", xy=(traj_a[i], traj_b[i]),
                                    xytext=(traj_a[i] - dx*0.4, traj_b[i] - dy*0.4),
                                    arrowprops=dict(arrowstyle="-|>", color=cm.plasma(0.1 + 0.8*t_norm),
                                                    lw=1.2, mutation_scale=10))

                # ── Start & End points ────────────────────────────────────
                ax.scatter(traj_a[0], traj_b[0], s=120, color="#f87171",
                           zorder=6, edgecolors="white", lw=0.8)
                ax.scatter(traj_a[-1], traj_b[-1], s=180, color="white",
                           zorder=6, marker="*")

                # ── Prototype attractors — annotated, not in legend ───────
                ax.scatter(ov_aa, ov_ab, s=90, color="#4ade80",
                           zorder=5, marker="D", edgecolors="none")
                ax.annotate(f"  ξ{digit_a}", xy=(ov_aa, ov_ab),
                            color="#4ade80", fontsize=8, va="center",
                            fontfamily="monospace")

                ax.scatter(ov_ba, ov_bb, s=90, color="#facc15",
                           zorder=5, marker="D", edgecolors="none")
                ax.annotate(f"  ξ{digit_b}", xy=(ov_ba, ov_bb),
                            color="#facc15", fontsize=8, va="center",
                            fontfamily="monospace")

                # ── Final overlap labels ──────────────────────────────────
                final_ov_a = traj_a[-1]
                final_ov_b = traj_b[-1]
                ax.annotate(f"end ({final_ov_a:.2f}, {final_ov_b:.2f})",
                            xy=(traj_a[-1], traj_b[-1]),
                            xytext=(traj_a[-1] - 0.08, traj_b[-1] + 0.04),
                            color="#94a3b8", fontsize=7, fontfamily="monospace",
                            arrowprops=dict(arrowstyle="-", color="#475569", lw=0.6))

                ax.set_xlabel(f"Overlap with  ξ{digit_a}  (digit {digit_a})",
                              color="#94a3b8", fontsize=9)
                ax.set_ylabel(f"Overlap with  ξ{digit_b}  (digit {digit_b})",
                              color="#94a3b8", fontsize=9)
                ax.set_title(title, color=accent, fontsize=10, fontweight="bold", pad=10)
                ax.tick_params(colors="#475569", labelsize=8)
                for sp in ax.spines.values(): sp.set_edgecolor("#1e293b")

                ax.set_xlim(-0.05, 1.1)
                ax.set_ylim(-0.05, 1.1)

            # ── Shared legend below both plots ────────────────────────────
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_elements = [
                Line2D([0], [0], color=cm.plasma(0.1), lw=2, label="Start of recall"),
                Line2D([0], [0], color=cm.plasma(0.9), lw=2, label="End of recall"),
                plt.scatter([], [], s=120, c="#f87171", edgecolors="white", lw=0.8,
                            label="Start point (noisy input)"),
                plt.scatter([], [], s=180, c="white", marker="*",
                            label="End point (recalled state)"),
                plt.scatter([], [], s=90, c="#4ade80", marker="D",
                            label=f"Prototype  ξ{digit_a}"),
                plt.scatter([], [], s=90, c="#facc15", marker="D",
                            label=f"Prototype  ξ{digit_b}"),
            ]
            fig.legend(handles=legend_elements, loc="lower center", ncol=3,
                       facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1",
                       fontsize=8, bbox_to_anchor=(0.5, -0.02))

            st.image(fig_to_pil(fig), use_container_width=True)

            st.markdown(
                '<div class="explain-box">'
                'Each panel shows how the high-dimensional state S(t) moves through the phase plane '
                'defined by its overlap with two competing attractors. '
                '<b>Standard network:</b> directed, sustained trajectory descending into one basin — '
                'guaranteed by the Lyapunov energy function. '
                '<b>Inhibitory network:</b> more constrained path — structural frustration from '
                'Dale\'s Principle prevents full descent into the attractor, stalling at lower overlap. '
                'Arrows indicate the direction of time.'
                '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WEIGHT MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_weights:
    st.markdown("### Synaptic weight matrix W")
    st.markdown("<div class='metric-label'>Structural comparison of symmetric vs asymmetric architecture</div>",
                unsafe_allow_html=True)
    st.markdown("")

    run_wm = st.button("▶  Visualise Weight Matrices", type="primary")

    if run_wm:
        patterns = [all_patterns[d] for d in stored_digits]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#0f172a")

        for ax_i, (frac, label) in enumerate([(0.0, "Standard (Symmetric)"), (0.2, "Inhibitory (Asymmetric)")]):
            net_wm = HopfieldNetwork(784, inhibitory_fraction=frac)
            net_wm.train(patterns)
            W = net_wm.weights

            ax = axes[ax_i]
            ax.set_facecolor("#0f172a")
            im = ax.imshow(W, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax.set_title(label, color="#e2e8f0", fontsize=10, fontweight="bold")
            ax.set_xlabel("Neuron j", color="#64748b", fontsize=8)
            ax.set_ylabel("Neuron i", color="#64748b", fontsize=8)
            ax.tick_params(colors="#475569", labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor("#1e293b")
            plt.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(colors="#94a3b8", labelsize=7)

        fig.suptitle("Visualization of Synaptic Weight Matrix W",
                     color="#e2e8f0", fontsize=12, y=1.02)
        plt.tight_layout()
        st.image(fig_to_pil(fig), use_container_width=True)

        st.markdown(
            '<div class="explain-box">'
            '<b>Left (Standard):</b> perfectly symmetric across the diagonal (w_ij = w_ji) — '
            'an undirected graph. The weight matrix is a valid Lyapunov function.<br><br>'
            '<b>Right (Inhibitory):</b> specific columns are flipped to all-negative values '
            '(striped pattern). These are the inhibitory neurons. The asymmetry breaks '
            'w_ij = w_ji, converting the network into a weighted directed graph — and '
            'removing the guarantee of convergence to a fixed point.'
            '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — NOISE SWEEP
# ══════════════════════════════════════════════════════════════════════════════
with tab_sweep:
    st.markdown("### Noise sweep — recall accuracy vs η")
    st.markdown("<div class='metric-label'>Recall accuracy vs η across 10 noise levels, averaged over multiple trials</div>",
                unsafe_allow_html=True)
    st.markdown("")

    if input_mode != "MNIST + noise":
        st.info("Switch to MNIST + noise mode.")
    else:
        n_trials = st.slider("Trials per noise level", 3, 20, 5)
        run_sweep = st.button("▶  Run Noise Sweep", type="primary")

        if run_sweep:
            patterns   = [all_patterns[d] for d in stored_digits]
            noise_vals = np.linspace(0.0, 0.5, 11)
            results    = {0.0: [], 0.2: []}
            progress   = st.progress(0)
            total_runs = len(noise_vals) * 2 * n_trials
            done       = 0

            for frac in [0.0, 0.2]:
                net_sw = HopfieldNetwork(784, inhibitory_fraction=frac)
                net_sw.train(patterns)
                for eta in noise_vals:
                    run_ovs = []
                    for t in range(n_trials):
                        noisy_t = add_noise(all_patterns[test_digit], eta, seed=t*13)
                        fs, _   = net_sw.recall(noisy_t, max_steps=max_steps, record_energy=False)
                        run_ovs.append(calculate_overlap(fs, all_patterns[test_digit]))
                        done += 1
                        progress.progress(done / total_runs)
                    results[frac].append((eta, np.mean(run_ovs), np.std(run_ovs)))

            fig, ax = dark_fig(8, 4.5)

            for frac, color, label in [(0.0, "#60a5fa", "Standard"), (0.2, "#c084fc", "Inhibitory (20%)")]:
                etas   = [r[0] for r in results[frac]]
                means  = np.array([r[1] for r in results[frac]])
                stds   = np.array([r[2] for r in results[frac]])
                ax.plot(etas, means, color=color, lw=2, marker="o", ms=5, label=label)
                ax.fill_between(etas, means - stds, means + stds, color=color, alpha=0.15)

            ax.axvline(0.40, color="#f87171", lw=1, ls="--", alpha=0.7, label="ηc ≈ 0.40")
            ax.axhline(0.0,  color="#475569", lw=0.6, ls=":")
            ax.set_xlabel("Noise level η (fraction of flipped bits)", color="#94a3b8", fontsize=10)
            ax.set_ylabel("Recall accuracy (overlap)", color="#94a3b8", fontsize=10)
            ax.set_title("Robustness to Noise", color="#e2e8f0", fontsize=11)
            ax.set_ylim(-1.1, 1.1)
            ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=9)

            st.image(fig_to_pil(fig), use_container_width=True)

            st.markdown(
                '<div class="explain-box">'
                'Phase transition from ordered (memory) to disordered (paramagnetic) phase at '
                'η_c ≈ 0.40. Standard network: sharp transition, higher peak accuracy. '
                'Inhibitory network: more gradual degradation — biological asymmetry provides '
                'flexibility but at the cost of absolute retrieval fidelity.'
                '</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="metric-label" style="text-align:center">Hopfield Networks · Hebbian Learning · '
    'Dale\'s Principle · MNIST · Built with Streamlit</div>',
    unsafe_allow_html=True)
