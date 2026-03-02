import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image, ImageOps
import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from hopfield import HopfieldNetwork
from utils import calculate_overlap, add_noise, pattern_to_image
from data import load_mnist_patterns

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neural Memory Lab", page_icon="🧠", layout="wide")

st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

# ── Load MNIST (cached) ───────────────────────────────────────────────────────
@st.cache_data
def get_patterns():
    return load_mnist_patterns()

# ── Helpers ───────────────────────────────────────────────────────────────────
def pattern_to_pil(pattern, size=196):
    img_array = pattern_to_image(pattern)
    return Image.fromarray(img_array, mode="L").resize((size, size), Image.NEAREST)

def plot_energy(energy_history, max_steps):
    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    ax.plot(np.arange(max_steps), energy_history, color="#a78bfa", linewidth=1.2)
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

def generate_recall_frames(net, noisy_pattern, animation_steps=600, sample_every=8):
    state = np.ascontiguousarray(noisy_pattern, dtype=np.float64).flatten()
    W = np.ascontiguousarray(net.weights, dtype=np.float64)
    N = W.shape[0]
    frames = [pattern_to_pil(state)]
    for step in range(animation_steps):
        neuron = int(np.random.randint(0, N))
        local_field = float(W[neuron].dot(state))
        state[neuron] = 1.0 if local_field >= 0.0 else -1.0
        if step % sample_every == 0:
            frames.append(pattern_to_pil(state))
    return frames, state

def frames_to_gif(frames, duration=60):
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:], loop=0, duration=duration)
    buf.seek(0)
    return buf.read()

def canvas_to_pattern(canvas_result):
    """Convert drawable canvas image to a 784-element bipolar pattern."""
    img_data = canvas_result.image_data  # RGBA numpy array (H, W, 4)
    # Use alpha channel: drawn pixels have alpha > 0
    alpha = img_data[:, :, 3]
    gray = Image.fromarray(alpha.astype(np.uint8), mode="L")
    gray_28 = gray.resize((28, 28), Image.LANCZOS)
    arr = np.array(gray_28)
    return np.where(arr > 30, 1, -1).astype(np.float64)

# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════
st.title("🧠 Neural Memory Lab")
st.markdown("*Associative memory via Hopfield Networks — interactive recall demo*")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    stored_digits = st.multiselect(
        "Patterns to store in memory",
        options=list(range(10)),
        default=[0, 1, 3, 5, 8],
    )

    st.markdown("---")
    st.subheader("Input mode")
    input_mode = st.radio(
        "Choose input",
        ["MNIST prototype + noise", "Draw your own digit"],
        index=0,
    )

    if input_mode == "MNIST prototype + noise":
        test_digit = st.selectbox(
            "Digit to recall",
            options=stored_digits if stored_digits else [0],
        )
        # ✅ FIX: slider as integer 0–50, then divide by 100
        noise_pct = st.slider("Noise level", min_value=0, max_value=50,
                               value=25, step=5, format="%d%%")
        noise_level = noise_pct / 100.0
    else:
        test_digit = None
        noise_level = 0.0

    st.markdown("---")
    inhibitory = st.checkbox("Enable inhibitory neurons (20%)", value=False)
    inhibitory_fraction = 0.2 if inhibitory else 0.0

    max_steps = st.slider("Max recall steps", 200, 2000, 1000, step=100)
    animate = st.checkbox("Animate recall (GIF)", value=True)

    run_btn = st.button("▶  Run Recall", type="primary", use_container_width=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading MNIST prototypes…"):
    all_patterns = get_patterns()

if not stored_digits:
    st.warning("Select at least one digit to store.")
    st.stop()

# ── Main layout ───────────────────────────────────────────────────────────────
col_input, col_noisy, col_recalled = st.columns(3)

# ── Column 1: input ───────────────────────────────────────────────────────────
with col_input:
    if input_mode == "MNIST prototype + noise":
        st.subheader("① Original pattern")
        st.image(pattern_to_pil(all_patterns[test_digit]),
                 caption=f"MNIST prototype '{test_digit}'",
                 use_container_width=True)
        input_pattern = all_patterns[test_digit]
    else:
        st.subheader("① Draw a digit")
        try:
            from streamlit_drawable_canvas import st_canvas
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=196,
                width=196,
                drawing_mode="freedraw",
                key="canvas",
            )
            if canvas_result.image_data is not None:
                input_pattern = canvas_to_pattern(canvas_result)
                st.caption("Drawn digit (28×28 after binarization)")
            else:
                input_pattern = None
                st.info("Draw a digit in the box above, then press Run Recall.")
        except ImportError:
            st.error("streamlit-drawable-canvas not installed.")
            input_pattern = None

# ── Column 2 & 3: placeholder ────────────────────────────────────────────────
with col_noisy:
    st.subheader("② Corrupted input")
    if not run_btn:
        st.info("Press **▶ Run Recall** to start.")

with col_recalled:
    st.subheader("③ Recalled pattern")
    if not run_btn:
        st.info("Results will appear here.")

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    if input_mode == "Draw your own digit" and input_pattern is None:
        st.warning("Draw something first!")
        st.stop()

    patterns = [all_patterns[d] for d in stored_digits]

    with st.spinner("Training network…"):
        net = HopfieldNetwork(784, inhibitory_fraction=inhibitory_fraction)
        net.train(patterns)

    # Generate noisy input
    if input_mode == "MNIST prototype + noise":
        noisy = add_noise(input_pattern, noise_level, seed=42)
        noise_caption = f"Noise: {noise_pct}% flipped bits"
    else:
        # For drawn input: add a small random noise to help convergence
        noisy = input_pattern.copy()
        noise_caption = "Your hand-drawn input"

    with col_noisy:
        st.image(pattern_to_pil(noisy), caption=noise_caption,
                 use_container_width=True)

    # Recall
    if animate:
        with st.spinner("Generating animation…"):
            frames, final_state = generate_recall_frames(
                net, noisy, animation_steps=max_steps, sample_every=8)
        with col_recalled:
            gif_bytes = frames_to_gif(frames)
            st.image(gif_bytes, caption="Asynchronous recall animation",
                     use_container_width=True)
    else:
        with st.spinner("Running recall…"):
            final_state, energy_history = net.recall(
                noisy, max_steps=max_steps, record_energy=True)
        with col_recalled:
            st.image(pattern_to_pil(final_state), caption="Network attractor",
                     use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    overlap = calculate_overlap(final_state, input_pattern)
    noisy_overlap = calculate_overlap(noisy, input_pattern)
    recovery = overlap - noisy_overlap

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)

    def metric_card(col, label, value, color="#a78bfa"):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color}">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True)

    metric_card(m1, "Stored patterns", len(stored_digits))
    metric_card(m2, "Overlap with original", f"{overlap:.3f}",
                "#34d399" if overlap > 0.8 else "#f87171")
    metric_card(m3, "Input overlap (noisy)", f"{noisy_overlap:.3f}", "#facc15")
    metric_card(m4, "Recovery Δ", f"{recovery:+.3f}",
                "#34d399" if recovery > 0 else "#f87171")

    if not animate:
        st.markdown("---")
        st.subheader("📉 Energy landscape")
        st.image(plot_energy(energy_history, max_steps), use_container_width=True)

    st.markdown("---")
    st.subheader("🗃️ Stored memory patterns")
    gcols = st.columns(len(stored_digits))
    for col, d in zip(gcols, stored_digits):
        col.image(pattern_to_pil(all_patterns[d], size=100),
                  caption=f"Digit {d}", use_container_width=True)

st.markdown("---")
st.caption("Hopfield Network · Hebbian learning · MNIST · Built with Streamlit & Numba")
