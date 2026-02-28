# Explainable AI for Healthcare Diagnostics

A dual-task clinical decision support system that combines deep learning with explainability methods to assist clinicians in diagnosing **pneumonia from chest X-rays** and predicting **cardiac risk from structured patient data**.

Built on the **DA-SPL architecture** (arXiv:2510.10037), augmented with Grad-CAM, SHAP, and Gemini-powered structured report generation.

---

## Architecture Overview

Both models share the DA-SPL pipeline:

```
Input
  └── Feature Extractor (ResNet-18 CNN / Token Embeddings)
       └── Dual-Attention Module (DAM)
            ├── Multi-head Self-Attention
            ├── Learnable Head Weights  (wₐ)
            └── Cosine Dual-Weighting  (w_dwa)
                 └── FFN
                      ├── Classifier 1  ─┐
                      ├── Classifier 2  ─┤── Combined Loss
                      └── LEM (Label Enhancement Module) ─┘
```

**Training objective:** `loss = loss₁ + λ·loss₂ + α·loss_t`  (DA-SPL Eq. 15–16, α=5)

---

## Models

### 1. ChestXrayCNN — Pneumonia Detection
| Property | Value |
|---|---|
| Backbone | ResNet-18 (pretrained, ImageNet) |
| Task | Binary: NORMAL / PNEUMONIA |
| Input | Chest X-ray image (224×224 RGB) |
| LEM labels | lung_opacity, consolidation, air_bronchograms, pleural_effusion |
| Best Val Accuracy | 100% |
| Saved checkpoint | `models/cnn.pth` |

### 2. TabularNN — Heart Disease Prediction
| Property | Value |
|---|---|
| Backbone | Token-attention MLP (13 feature tokens) |
| Task | Binary: Healthy / Heart Disease |
| Input | 13 clinical features (UCI Heart Disease dataset) |
| LEM labels | typical_angina, elevated_heart_rate, exercise_induced_angina, ST_depression, vessel_narrowing |
| Best Val Accuracy | 89.61% |
| Saved checkpoint | `models/tabular.pth` |

---

## Explainability Methods

| Method | Model | Description |
|---|---|---|
| **Grad-CAM** | X-ray CNN | Class-discriminative spatial heatmap over lung regions |
| **Dual-Attention Heatmap** | X-ray CNN | DA head-weighted 7×7 token attention map, upsampled to 224×224 |
| **SHAP GradientExplainer** | Tabular | Per-feature contribution scores, signed for risk direction |
| **DAM Feature Importance** | Tabular | Per-feature attention scores from dual-weighted attention heads |
| **LEM Indicators** | Both | Probabilistic clinical sub-label predictions (tiered HIGH/MOD/LOW) |
| **Gemini Report** | Both | Clinically structured narrative report grounded in all model outputs |

---

## Project Structure

```
explainable_ai_health/
├── app.py                  # Streamlit UI (tab-based, dual-model)
├── model.py                # DA-SPL model definitions (DAM, LEM, ChestXrayCNN, TabularNN)
├── train.py                # Training loops for both models
├── interpret.py            # Grad-CAM, DA heatmap, SHAP, feature attention
├── data_loader.py          # Tabular data pipeline (StandardScaler, train/val split)
├── report_generator.py     # Structured Gemini prompt builder
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages for Streamlit Cloud (libgl1-mesa-glx)
├── .env                    # API key (not committed)
├── .gitignore
├── data/
│   ├── chest_pneumonia/    # X-ray dataset (train/val/test folders)
│   └── tabular/
│       └── heart.csv       # UCI Heart Disease dataset
└── models/
    ├── cnn.pth             # Trained ChestXrayCNN weights
    └── tabular.pth         # Trained TabularNN weights
```

---

## Setup

### 1. Clone and enter the project
```bash
git clone https://github.com/sambhukalaga27/explainable_ai_health.git
cd explainable_ai_health
```

### 2. Create and activate virtual environment
```powershell
python -m venv explainai_env
.\explainai_env\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Gemini API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```
Get a free key at [https://aistudio.google.com](https://aistudio.google.com).

---

## Training

Trained checkpoints (`models/cnn.pth`, `models/tabular.pth`) are included. To retrain:

```bash
python train.py
```

Training configuration:
- **CNN**: AdamW, backbone LR 5e-5 / head LR 1e-4, CosineAnnealingLR, 50 epochs
- **Tabular**: AdamW LR 1e-3, CosineAnnealingLR, 300 epochs

---

## Running the App

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

### Tab 1 — Chest X-ray Diagnosis
1. Upload a chest X-ray (JPG/PNG)
2. View prediction + confidence
3. Inspect three-panel visualisation: Original / Grad-CAM / Dual-Attention Heatmap
4. Review LEM clinical sub-findings (tiered by probability)
5. Expand dual-attention head weight chart
6. Click **Generate Structured Diagnostic Report** for a Gemini-powered radiology report

### Tab 2 — Heart Disease Prediction
1. Enter the 13 patient biomarkers
2. Click **Predict Cardiac Risk**
3. View prediction + confidence + risk tier
4. Inspect SHAP feature contributions and Dual-Attention feature importance side by side
5. Review LEM cardiac risk indicators
6. Click **Generate Structured Cardiac Report** for a Gemini-powered cardiology report

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Model training and inference |
| `streamlit` | Web UI |
| `shap` | SHAP GradientExplainer |
| `opencv-python` | Heatmap overlays |
| `google-genai` | Gemini 2.5 Flash API |
| `python-dotenv` | `.env` API key loading |
| `scikit-learn` | StandardScaler, metrics |
| `pandas`, `numpy` | Data handling |

---

## Deployment (Streamlit Cloud)

- `packages.txt` lists `libgl1-mesa-glx` — required by OpenCV on Linux (Streamlit Cloud)
- Set `GEMINI_API_KEY` as a **Secret** in the Streamlit Cloud dashboard
- Point deployment to `app.py`

---

## Reference

DA-SPL architecture adapted from:
> *"Dual-Attention Based Semi-supervised Prototype Learning for Glaucoma Grading Report Generation"*  
> arXiv:2510.10037

---

## License

© 2026 Sai Sambhu Prasad Kalaga. All rights reserved.
