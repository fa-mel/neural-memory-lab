# 🧠 Neural Memory Lab

An interactive web application to explore **Hopfield Networks** and associative memory — built from a computational physics simulation.

---

## What is this?

A Hopfield Network is a recurrent neural network that works as a **content-addressable memory**: you store patterns in it, corrupt them with noise, and the network autonomously reconstructs the original. It is a physical model — the dynamics are equivalent to a spin glass system minimizing a Hamiltonian energy function.

This app lets you interact with two architectures:

- **Standard (Symmetric)** — classical Hebbian network, guaranteed convergence via Lyapunov function
- **Inhibitory (20%)** — Dale's Principle enforced: 20% of neurons are purely inhibitory, breaking symmetry and generating complex dynamics

---

## Features

| Tab | Description |
|-----|-------------|
| ⚡ **Recall** | Run a single recall. Draw a digit or use MNIST + noise. See animated convergence, energy landscape, SampEn, and a physics interpretation of the result |
| ⚖️ **Side-by-Side** | Same input through both architectures simultaneously — compare recalled patterns, energy trajectories, and Sample Entropy |
| 📡 **Phase Space** | 20-trial ensemble scatter of overlap vs Sample Entropy — visualises the trade-off between retrieval fidelity and dynamical complexity |
| 🌀 **State Space** | 2D projection of S(t) onto two competing attractors, colored by time with directional arrows — shows how the network navigates the energy landscape |
| 🔬 **Weight Matrix** | Heatmap of W for both architectures — shows the structural symmetry breaking introduced by Dale's Principle |
| 📈 **Noise Sweep** | Automated sweep of η from 0 to 50%, averaged over multiple trials — maps the phase transition from ordered to paramagnetic regime |

---

## Physics background

The network stores M patterns as attractors of the energy function:

```
H(S) = -½ Σ wᵢⱼ sᵢ sⱼ
```

Weights are set by the Hebbian rule: `W = (1/M) Σ ξᵐ ⊗ ξᵐ`

The system evolves by asynchronous update: each neuron flips to match the sign of its local field. For a symmetric W, energy is guaranteed to decrease monotonically → convergence to a fixed-point attractor.

Introducing **inhibitory neurons** (Dale's Principle) breaks W's symmetry → no Lyapunov guarantee → the system can exhibit *ceaseless dynamics*, quantified here by **Sample Entropy** on the energy time series.

**Key observable — Storage load:** `α = M/N`. For MNIST digits (non-orthogonal), retrieval degrades significantly beyond M ≈ 3 due to pattern interference.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free

### Streamlit Community Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select the repo, set main file to `app.py`
4. Deploy — public URL in ~2 minutes

### Hugging Face Spaces
1. Create a new Space → SDK: Streamlit, Hardware: CPU Basic (free)
2. Upload all files or push via `huggingface_hub`

---

## Project structure

```
├── app.py              # Streamlit UI — all 6 tabs
├── requirements.txt
└── core/ (or root)
    ├── hopfield.py     # HopfieldNetwork class — train, recall, trajectory
    ├── utils.py        # overlap, noise, binarization, Sample Entropy
    └── data.py         # MNIST loader and binarization
```

---

## Dependencies

- `streamlit` — UI framework
- `numpy` — all linear algebra
- `matplotlib` — plots and heatmaps
- `Pillow` — image rendering and GIF generation
- `tensorflow` — MNIST dataset download only
- `streamlit-drawable-canvas` — freehand digit drawing
