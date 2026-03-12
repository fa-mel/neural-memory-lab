# 🧠 Neural Memory Lab

Interactive Streamlit app for exploring Hopfield Network associative memory — built from a physics simulation of inhibitory dynamics and Dale's Principle.

**Mellini (2026) · Physical Methods of Biology · UNIBO**

## Features

- **Recall Lab** — choose MNIST digits or draw your own, control noise, animate the recall with a frame-by-frame slider, see the energy landscape in real time
- **Standard vs Inhibitory** — side-by-side comparison on the same noisy input (§4.1)
- **Phase Space** — ensemble scatter of Overlap vs Sample Entropy with paper centroids (§4.2)
- **State-Space Trajectories** — 2D overlap projection with time-colored paths (§4.3)
- **Noise Sweep & Capacity** — reproduce the phase transition (Fig. 2) and catastrophic forgetting (Fig. 5)
- **Weight Matrix** — interactive heatmap comparing symmetric vs asymmetric W (Fig. 1)

All charts are interactive (Plotly) — hover, zoom, pan.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

### Streamlit Community Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `app.py` → Deploy

### Hugging Face Spaces
1. Create Space → SDK: Streamlit
2. Upload all files
3. Public URL generated automatically

## Project structure

```
neural_memory_lab/
├── app.py               # Streamlit UI (plotly charts, all experiments)
├── requirements.txt
├── README.md
└── core/
    ├── __init__.py
    ├── hopfield.py      # HopfieldNetwork — unified recall with incremental energy
    ├── utils.py         # overlap, noise, SampEn, image helpers
    └── data.py          # MNIST loader
```
