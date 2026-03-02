import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from core.hopfield import HopfieldNetwork
from core.utils import calculate_overlap, add_noise, pattern_to_image
from core.data import load_mnist_patterns

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Memory Lab",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    h1 { color: #a78bfa; }
    h2, h3 { color: #c4b5fd; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #3b3b5c;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #a78bfa; }
    .metric-label { font-size: 0.85rem; color: #888; }
    .stSlider > div > div { color: #a78bfa; }
</style>
""", unsafe_allow_html=True)

# ── Load MNIST (cached) ───────────────────────────────────────────────────────
@st.cache_data
def get_patterns():
    return load_mnist_patterns()

# ── Helper: render pattern as PIL image ──────────────────────────────────────
def pattern_to_pil(pattern, size=196):
    img_array = pattern_to_image(pattern)
    img = Image.fromarray(img_array, mode="L").resize((size, size), Image.NEAREST)
    return img

# ── Helper: energy plot ───────────────────────────────────────────────────────
def plot_energy(energy_history, max_steps):
    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    steps = np.arange(max_steps)
    ax.plot(steps, energy_history, color="#a78bfa", linewidth=1.2)
    ax.set_xlabel("Update step", color="#888", fontsize=9)
    ax.set_ylabel("Energy  E", color="#888", fontsize=9)
    ax.set_title("Energy landscape during recall", color="#c4b5fd", fontsize=10)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ── Helper: animated recall frames ───────────────────────────────────────────
def generate_recall_frames(net, noisy_pattern, animation_steps=600, sample_every=10):
    state = noisy_pattern.copy()
    frames = [pattern_to_pil(state)]
    for step in range(animation_steps):
        neuron = np.random.randint(net.N)
        local_field = np.dot(net.weights[neuron, :], state)
        state[neuron] = 1.0 if local_field >= 0 else -1.0
        if step % sample_every == 0:
            frames.append(pattern_to_pil(state))
    return frames, state

# ── Helper: create GIF ────────────────────────────────────────────────────────
def frames_to_gif(frames, duration=80):
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration,
    )
    buf.seek(0)
    return buf.read()

# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════

st.title("🧠 Neural Memory Lab")
st.markdown("*Associative memory via Hopfield Networks — interactive recall demo*")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Network configuration")

    stored_digits = st.multiselect(
        "Patterns to store in memory",
        options=list(range(10)),
        default=[0, 1, 3, 5, 8],
        help="Which MNIST digit prototypes the network will memorise.",
    )

    test_digit = st.selectbox(
        "Digit to recall",
        options=stored_digits if stored_digits else [0],
        index=0,
        help="Starting point for the recall process.",
    )

    noise_level = st.slider(
        "Noise level",
        min_value=0.0,
        max_value=0.5,
        value=0.25,
        step=0.05,
        format="%.0f%%",
        help="Fraction of bits randomly flipped in the input.",
    )
    noise_level_display = noise_level  # keep as fraction

    inhibitory = st.checkbox(
        "Enable inhibitory neurons (20%)",
        value=False,
        help="Applies Dale's Principle: 20% of neurons become purely inhibitory.",
    )
    inhibitory_fraction = 0.2 if inhibitory else 0.0

    max_steps = st.slider(
        "Max recall steps",
        min_value=200,
        max_value=2000,
        value=1000,
        step=100,
    )

    animate = st.checkbox("Generate recall animation (slower)", value=True)

    run_btn = st.button("▶  Run Recall", type="primary", use_container_width=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading MNIST prototypes…"):
    all_patterns = get_patterns()

if not stored_digits:
    st.warning("Select at least one digit to store.")
    st.stop()

# ── Main columns ──────────────────────────────────────────────────────────────
col_orig, col_noisy, col_recalled = st.columns(3)

with col_orig:
    st.subheader("① Original pattern")
    st.image(
        pattern_to_pil(all_patterns[test_digit]),
        caption=f"Stored prototype  '{test_digit}'",
        use_container_width=True,
    )

# ── Run on button press ───────────────────────────────────────────────────────
if run_btn:
    patterns = [all_patterns[d] for d in stored_digits]

    with st.spinner("Training network…"):
        net = HopfieldNetwork(784, inhibitory_fraction=inhibitory_fraction)
        net.train(patterns)

    noisy = add_noise(all_patterns[test_digit], noise_level_display, seed=42)

    with col_noisy:
        st.subheader("② Corrupted input")
        st.image(
            pattern_to_pil(noisy),
            caption=f"Noise: {int(noise_level_display*100)}% flipped bits",
            use_container_width=True,
        )

    # ── Recall ────────────────────────────────────────────────────────────────
    if animate:
        with st.spinner("Running animated recall…"):
            frames, final_state = generate_recall_frames(
                net, noisy, animation_steps=max_steps, sample_every=8
            )
        with col_recalled:
            st.subheader("③ Recall in progress")
            gif_bytes = frames_to_gif(frames, duration=60)
            st.image(gif_bytes, caption="Asynchronous update animation", use_container_width=True)
    else:
        with st.spinner("Running recall…"):
            final_state, energy_history = net.recall(
                noisy, max_steps=max_steps, record_energy=True
            )
        with col_recalled:
            st.subheader("③ Recalled pattern")
            st.image(
                pattern_to_pil(final_state),
                caption="Network attractor",
                use_container_width=True,
            )

    # ── Metrics ───────────────────────────────────────────────────────────────
    overlap = calculate_overlap(final_state, all_patterns[test_digit])
    noisy_overlap = calculate_overlap(noisy, all_patterns[test_digit])
    recovery = overlap - noisy_overlap

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)

    def metric_card(col, label, value, color="#a78bfa"):
        col.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    metric_card(m1, "Stored patterns", len(stored_digits))
    metric_card(
        m2,
        "Overlap with original",
        f"{overlap:.3f}",
        "#34d399" if overlap > 0.8 else "#f87171",
    )
    metric_card(m3, "Input overlap (noisy)", f"{noisy_overlap:.3f}", "#facc15")
    metric_card(
        m4,
        "Recovery Δ",
        f"{recovery:+.3f}",
        "#34d399" if recovery > 0 else "#f87171",
    )

    # ── Energy landscape (non-animated path) ─────────────────────────────────
    if not animate:
        st.markdown("---")
        st.subheader("📉 Energy landscape")
        energy_img = plot_energy(energy_history, max_steps)
        st.image(energy_img, use_container_width=True)

    # ── Stored patterns gallery ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🗃️ Stored memory patterns")
    gallery_cols = st.columns(len(stored_digits))
    for col, d in zip(gallery_cols, stored_digits):
        col.image(
            pattern_to_pil(all_patterns[d], size=100),
            caption=f"Digit {d}",
            use_container_width=True,
        )

else:
    with col_noisy:
        st.subheader("② Corrupted input")
        st.info("Press **▶ Run Recall** to generate the noisy input and start the network.")

    with col_recalled:
        st.subheader("③ Recalled pattern")
        st.info("Results will appear here after recall.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Hopfield Network · Hebbian learning · MNIST dataset · "
    "Built with Streamlit & Numba"
)
