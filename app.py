# app.py  =  DA-SPL Enhanced Explainable AI for Healthcare
# ==================================================================
import streamlit as st
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv
import os
from google import genai


from model import ChestXrayCNN, TabularNN, XRAY_CLINICAL_LABELS, HEART_CLINICAL_LABELS
from interpret import (
    GradCAM,
    get_shap_values,
    get_dual_attention_heatmap,
    attention_to_rgb_overlay,
    get_feature_attention_scores,
)
from data_loader import load_tabular_data
from report_generator import build_xray_report_data, build_heart_report_data

load_dotenv()
# Support both local .env and Streamlit Cloud st.secrets
try:
    _api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    _api_key = os.getenv("GEMINI_API_KEY")
_genai_client = genai.Client(api_key=_api_key)

def explain_with_gemini(prompt: str) -> str:
    response = _genai_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()

# -- Page config 
st.set_page_config(
    page_title="Explainable AI – Healthcare Diagnostics",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    "<h1 style='text-align:center;'>🧠 Explainable AI for Healthcare Diagnostics</h1>"
    "<p style='text-align:center; color:grey;'>Powered by Dual-Attention (DA-SPL) &middot; "
    "Grad-CAM &middot; SHAP &middot; Label Enhancement &middot; Gemini</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Model loading (cached) ----------------------------------------
@st.cache_resource
def load_models():
    cnn = ChestXrayCNN().to(device)
    cnn.load_state_dict(
        torch.load("models/cnn.pth", map_location=device, weights_only=True))
    cnn.eval()

    tab = TabularNN(input_dim=13).to(device)
    tab.load_state_dict(
        torch.load("models/tabular.pth", map_location=device, weights_only=True))
    tab.eval()
    return cnn, tab

cnn_model, tabular_model = load_models()


# TAB LAYOUT
# ==================================================================
tab_xray, tab_heart = st.tabs(["📸 Chest X-ray Diagnosis", "❤️ Heart Disease Prediction"])


# TAB 1  =  CHEST X-RAY
# ==================================================================
with tab_xray:
    st.subheader("Pneumonia Detection via DA-SPL Dual-Attention CNN")
    st.caption(
        "Upload a chest X-ray. The model uses a Dual-Attention Mechanism (DAM) to focus on "
        "pathologically significant lung regions, then a Label Enhancement Module (LEM) predicts "
        "specific radiological sub-findings to produce a richer diagnostic report."
    )

    xray_file = st.file_uploader("Upload X-ray (JPG / PNG)", type=["jpg","jpeg","png"],
                                  key="xray_upload")

    if xray_file:
        image = Image.open(xray_file).convert("RGB")
        original_np = np.array(image.resize((224, 224)).convert("RGB"))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)

        with st.spinner("Running DA-SPL inference..."):
            # -- forward pass --
            with torch.no_grad():
                model_out = cnn_model(input_tensor)

            logits1     = model_out["logits1"]
            pred_class  = logits1.argmax(1).item()
            confidence  = torch.softmax(logits1, dim=1)[0, pred_class].item()
            pred_label  = "PNEUMONIA" if pred_class == 1 else "NORMAL"

            # -- Grad-CAM (requires grad so re-run outside no_grad) --
            gradcam = GradCAM(
                model=cnn_model,
                target_layer=cnn_model.feature_extractor[-1],   # layer4
            )
            cam       = gradcam.generate(input_tensor, class_idx=pred_class)
            gradcam.remove_hooks()
            cam_heat  = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
            cam_heat  = cv2.cvtColor(cam_heat, cv2.COLOR_BGR2RGB).astype(np.uint8)
            if cam_heat.shape != original_np.shape:
                cam_heat = cv2.resize(cam_heat,
                                      (original_np.shape[1], original_np.shape[0]))
            gradcam_overlay = cv2.addWeighted(original_np, 0.6, cam_heat, 0.4, 0)

            # -- Dual-Attention heatmap --
            with torch.no_grad():
                da_heat   = get_dual_attention_heatmap(cnn_model, input_tensor)
            da_overlay = attention_to_rgb_overlay(original_np, da_heat, alpha=0.45)

        # -- Result banner --
        color = "🔴" if pred_class == 1 else "🟢"
        st.markdown(
            f"<h3 style='text-align:center;'>{color} {pred_label} — "
            f"{confidence:.1%} confidence</h3>",
            unsafe_allow_html=True,
        )

        # -- Three-column visual panel --
        col1, col2, col3 = st.columns(3)
        col1.image(image.resize((224,224)), caption="Original X-ray", use_container_width=True)
        col2.image(gradcam_overlay, caption="Grad-CAM (class-discriminative)",
                   use_container_width=True)
        col3.image(da_overlay,      caption="Dual-Attention Heatmap (DA-SPL)",
                   use_container_width=True)

        # -- LEM clinical sub-findings --
        st.markdown("#### 🔬 Label Enhancement — Clinical Sub-Findings")
        lem_probs = torch.sigmoid(model_out["lem_logits"]).detach().cpu().numpy().flatten()
        lem_cols  = st.columns(len(XRAY_CLINICAL_LABELS))
        for col, label, prob in zip(lem_cols, XRAY_CLINICAL_LABELS, lem_probs):
            bar_color = "#e74c3c" if prob > 0.5 else "#2ecc71"
            col.markdown(
                f"<div style='text-align:center;'>"
                f"<b>{label.replace('_',' ').title()}</b><br>"
                f"<span style='color:{bar_color}; font-size:1.4em;'><b>{prob:.0%}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # -- Dual-Attention head weights --
        with st.expander("🔍 Dual-Attention Head Weights (DAM)"):
            da_w  = model_out["da_weights"].cpu().numpy()
            chart = pd.DataFrame({"Head Weight": da_w},
                                  index=[f"Head {i}" for i in range(len(da_w))])
            st.bar_chart(chart)
            st.caption(
                "Higher weight = this attention head focuses more on "
                "pathologically significant regions (cosine dual-weighting, DA-SPL §III-B)."
            )

        # -- Structured Gemini report --
        st.markdown("---")
        if st.button("📝 Generate Structured Diagnostic Report (Gemini)", key="xray_report"):
            report_data = build_xray_report_data(model_out, pred_label, confidence)
            with st.spinner("Generating structured report..."):
                report_text = explain_with_gemini(report_data["gemini_prompt"])
            st.markdown("#### 🧠 AI Diagnostic Report")
            st.info(report_text)


# ==================================================================
# TAB 2  --  HEART DISEASE
# ==================================================================
with tab_heart:
    st.subheader("Cardiac Risk Prediction via DA-SPL Tabular Dual-Attention")
    st.caption(
        "Each of the 13 clinical features is treated as an attention token. "
        "The Dual-Attention Module (DAM) re-weights features based on pathological relevance, "
        "and the Label Enhancement Module (LEM) predicts specific cardiac risk indicators."
    )

    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ]
    feature_hints = {
        "age": "years", "sex": "1=M/0=F", "cp": "0-3",
        "trestbps": "mmHg", "chol": "mg/dL", "fbs": "0/1",
        "restecg": "0-2", "thalach": "bpm", "exang": "0/1",
        "oldpeak": "0-6", "slope": "0-2", "ca": "0-4", "thal": "1-3",
    }

    input_cols   = st.columns(4)
    input_values = []
    for i, feat in enumerate(features):
        val = input_cols[i % 4].number_input(
            f"{feat}  ({feature_hints.get(feat,'')})",
            value=0.0, format="%.2f", step=1.0, key=f"feat_{feat}")
        input_values.append(val)

    if st.button("🩺 Predict Cardiac Risk", key="heart_predict"):
        if all(v == 0.0 for v in input_values):
            st.warning("Please enter valid patient data before predicting.")
        else:
            with st.spinner("Running DA-SPL tabular inference..."):
                # -- scale input --
                _, _, _, feature_order, scaler = load_tabular_data(
                    "data/tabular/heart.csv")
                inp_dict    = dict(zip(features, input_values))
                inp_ordered = [inp_dict[f] for f in feature_order]
                inp_scaled  = scaler.transform([inp_ordered])
                inp_tensor  = torch.tensor(inp_scaled,
                                           dtype=torch.float32).to(device)

                # -- DA-SPL forward --
                with torch.no_grad():
                    model_out = tabular_model(inp_tensor)

                logits1    = model_out["logits1"]
                pred       = logits1.argmax(1).item()
                prob       = torch.softmax(logits1, dim=1)[0, pred].item()
                pred_label = "Heart Disease" if pred == 1 else "Healthy"

                # -- SHAP --
                _, val_loader, _, _, _ = load_tabular_data("data/tabular/heart.csv")
                background = next(iter(val_loader))[0][:100]
                shap_vals  = get_shap_values(
                    tabular_model, background, inp_tensor.cpu(), pred)

                # -- Feature attention scores from DAM --
                feat_attn = get_feature_attention_scores(tabular_model, inp_tensor)

            # -- Save to session state --
            st.session_state.heart_result = {
                "pred_label":  pred_label,
                "prob":        prob,
                "pred":        pred,
                "shap_vals":   shap_vals.tolist()
                               if hasattr(shap_vals, "tolist") else shap_vals,
                "feat_attn":   feat_attn.tolist(),
                "inp_ordered": inp_ordered,
                "model_out":   {k: v.detach().cpu() if isinstance(v, torch.Tensor)
                                else v for k, v in model_out.items()},
            }

    # -- Display results --
    if "heart_result" in st.session_state:
        r          = st.session_state.heart_result
        pred_label = r["pred_label"]
        prob       = r["prob"]
        shap_vals  = r["shap_vals"]
        feat_attn  = r["feat_attn"]
        inp_ordered= r["inp_ordered"]
        model_out  = r["model_out"]

        color = "🔴" if r["pred"] == 1 else "🟢"
        st.markdown(
            f"<h3 style='text-align:center;'>{color} {pred_label} — "
            f"{prob:.1%} confidence</h3>",
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)

        # -- SHAP values --
        with col_a:
            st.markdown("#### 🔍 SHAP Feature Contributions")
            sv = [v[0] if isinstance(v, list) else v for v in shap_vals]
            shap_df = pd.DataFrame({
                "Feature":     features,
                "SHAP Value":  sv,
                "Input Value": inp_ordered,
            }).sort_values("SHAP Value", key=abs, ascending=False)
            for _, row in shap_df.iterrows():
                c = "#e74c3c" if row["SHAP Value"] > 0 else "#3498db"
                st.markdown(
                    f"<span style='color:{c};'>&#9632;</span> "
                    f"<b>{row['Feature']}</b> = {row['Input Value']:.1f} "
                    f"&nbsp;&nbsp; SHAP: <code>{row['SHAP Value']:+.4f}</code>",
                    unsafe_allow_html=True,
                )

        # -- Dual-Attention feature importance --
        with col_b:
            st.markdown("#### 🎯 Dual-Attention Feature Importance (DAM)")
            fa_df = pd.DataFrame({
                "Attention Score": feat_attn,
            }, index=features).sort_values("Attention Score", ascending=False)
            st.bar_chart(fa_df)
            st.caption(
                "Higher score = the dual-attention head re-weighting marked this "
                "feature as more clinically relevant."
            )

        # -- LEM cardiac sub-indicators --
        st.markdown("#### 🏥 Label Enhancement — Cardiac Risk Indicators")
        lem_logits = model_out["lem_logits"]
        lem_probs  = torch.sigmoid(lem_logits).numpy().flatten()
        lem_cols   = st.columns(len(HEART_CLINICAL_LABELS))
        for col, label, p in zip(lem_cols, HEART_CLINICAL_LABELS, lem_probs):
            bar_color = "#e74c3c" if p > 0.5 else "#2ecc71"
            col.markdown(
                f"<div style='text-align:center;'>"
                f"<b>{label.replace('_',' ').title()}</b><br>"
                f"<span style='color:{bar_color}; font-size:1.3em;'><b>{p:.0%}</b></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # -- Dual-Attention head weights --
        with st.expander("🔍 Dual-Attention Head Weights (DAM)"):
            da_w  = model_out["da_weights"].numpy()
            chart = pd.DataFrame({"Head Weight": da_w},
                                  index=[f"Head {i}" for i in range(len(da_w))])
            st.bar_chart(chart)

        # -- Structured Gemini report --
        st.markdown("---")
        if st.button("📝 Generate Structured Cardiac Report (Gemini)", key="heart_report"):
            # Rebuild model_out with tensors on device for report
            mo_tensor = {
                k: torch.tensor(v) if isinstance(v, list) else v
                for k, v in model_out.items()
            }
            report_data = build_heart_report_data(
                mo_tensor, features, inp_ordered, shap_vals, pred_label, prob)
            with st.spinner("Generating structured cardiac report..."):
                report_text = explain_with_gemini(report_data["gemini_prompt"])
            st.markdown("#### 🧠 AI Cardiac Risk Report")
            st.info(report_text)


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("© 2026 Sai Sambhu Prasad Kalaga. All rights reserved.")
