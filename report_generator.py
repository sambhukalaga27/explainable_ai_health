# report_generator.py
# Structured diagnostic report builder — DA-SPL's LEM output (§III-C)
# drives clinically-grounded Gemini prompts (Label Enhancement analog).
# ——————————————————————————————————————————————————————————————————
import torch
import torch.nn.functional as F
import numpy as np
from model import XRAY_CLINICAL_LABELS, HEART_CLINICAL_LABELS


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _sigmoid_probs(logits: torch.Tensor) -> np.ndarray:
    """Convert raw LEM logits → probabilities (0-1)."""
    return torch.sigmoid(logits).detach().cpu().numpy().flatten()


def _head_weight_summary(da_weights: torch.Tensor) -> str:
    """Format dual-attention head weights for the prompt."""
    w = da_weights.cpu().numpy()
    lines = [f"  Head {i}: {v:.3f}" for i, v in enumerate(w)]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 1.  X-RAY STRUCTURED REPORT
# ══════════════════════════════════════════════════════════════════

def build_xray_report_data(model_out: dict, pred_label: str,
                            confidence: float) -> dict:
    """
    Assembles a structured data dict from the ChestXrayCNN forward pass.

    Returns keys:
      pred_label, confidence, clinical_findings, da_head_summary,
      high_attention_region (qualitative), gemini_prompt
    """
    lem_probs = _sigmoid_probs(model_out["lem_logits"])   # (4,)

    # Map LEM probabilities to clinical findings
    clinical_findings = {
        label: float(prob)
        for label, prob in zip(XRAY_CLINICAL_LABELS, lem_probs)
    }

    # Determine high-attention region qualitatively from the attn_map
    attn_map  = model_out["attn_map"]           # B, H, N(49), N(49)
    da_w      = model_out["da_weights"]          # H,
    weighted  = (attn_map * da_w.view(1, -1, 1, 1)).sum(dim=1)  # B,49,49
    tok_score = weighted[0].mean(dim=0).cpu().numpy()            # 49,
    grid      = tok_score.reshape(7, 7)
    cy, cx    = divmod(int(grid.argmax()), 7)

    # Map 7×7 grid position to quadrant description
    v_pos = "upper" if cy < 3 else ("lower" if cy > 4 else "mid")
    h_pos = "left"  if cx < 3 else ("right" if cx > 4 else "central")
    high_attention_region = f"{v_pos}-{h_pos} lung field"

    da_head_summary = _head_weight_summary(model_out["da_weights"])

    # Classify LEM findings by severity tier
    high   = {k: v for k, v in clinical_findings.items() if v > 0.65}
    mod    = {k: v for k, v in clinical_findings.items() if 0.40 < v <= 0.65}
    low    = {k: v for k, v in clinical_findings.items() if v <= 0.40}

    def _fmt_findings(d):
        return "\n".join(f"    - {k.replace('_',' ').title()}: {v:.1%}" for k, v in d.items()) or "    (none)"

    # Anatomical interpretation of the attention region
    anatomy_note = (
        "The upper lung fields correspond to the apical segments — "
        "a common site for early consolidation, atelectasis, and TB-related changes."
        if "upper" in high_attention_region else
        "The lower lung fields are common sites for aspiration pneumonia, "
        "basal consolidation, and pleural effusion."
        if "lower" in high_attention_region else
        "The mid-lung fields are often affected in lobar pneumonia "
        "and perihilar infiltrative processes."
    )

    gemini_prompt = f"""You are a senior radiologist reviewing an AI-assisted chest X-ray interpretation report.
A ResNet-18 CNN augmented with a Dual-Attention Module (DAM) and Label Enhancement Module (LEM), \
inspired by the DA-SPL architecture (arXiv:2510.10037), has analysed this X-ray.

Your role is to produce a structured, detailed radiological report grounded strictly in the model \
outputs below. Do NOT speculate beyond what the data supports.

════════════════════════════════════════
 MODEL OUTPUT SUMMARY
════════════════════════════════════════
Primary Diagnosis  : {pred_label}
Model Confidence   : {confidence:.1%}  {'(HIGH — strong evidence)' if confidence > 0.85 else '(MODERATE — interpret with caution)' if confidence > 0.65 else '(LOW — elevated clinical correlation required)'}

Region of Peak Dual-Attention Activity : {high_attention_region}
Anatomical Context : {anatomy_note}

Dual-Attention Head Weights (pathological focus per head):
{da_head_summary}

Label Enhancement Module (LEM) — Radiological Sub-Findings:
  HIGH PROBABILITY (>65%):
{_fmt_findings(high)}
  MODERATE PROBABILITY (40–65%):
{_fmt_findings(mod)}
  LOW PROBABILITY (<40%):
{_fmt_findings(low)}
════════════════════════════════════════

════════════════════════════════════════
 YOUR TASK — STRUCTURED RADIOLOGY REPORT
════════════════════════════════════════
Write a detailed, structured radiology report using ONLY the model data above. \
Each section must be substantive — do not leave sections as one-liners.

**1. RADIOLOGICAL IMPRESSION**
State the primary diagnosis, confidence tier, and the involved lung region. \
Note whether findings are unilateral or bilateral if determinable from attention data.

**2. DETAILED FINDINGS**
For each LEM sub-finding (regardless of threshold), provide:
  - Its clinical significance (what it indicates radiologically)
  - What its probability level means in practice
  - How it relates to the primary {pred_label} diagnosis
Specifically address:
  - Lung Opacity: Is the opacity focal, patchy, or diffuse? What lobe/region?
  - Consolidation: Does it suggest lobar, segmental, or bronchopneumonia pattern?
  - Air Bronchograms: Do they confirm alveolar filling? Significance for pneumonia vs other pathology?
  - Pleural Effusion: Is it likely a parapneumonic effusion or independent process?

**3. ATTENTION ANALYSIS**
Interpret the dual-attention focus on the {high_attention_region}. \
Explain what pathological processes are anatomically expected in this region \
and whether the attention pattern is consistent with the predicted diagnosis.

**4. DIFFERENTIAL DIAGNOSIS**
List 2–3 differential diagnoses to consider given these findings, ordered by likelihood. \
For each, state what feature or absence of feature makes it more or less probable.

**5. SEVERITY & URGENCY ASSESSMENT**
Based on confidence ({confidence:.1%}) and active LEM findings, classify as:
  - URGENT / SEMI-URGENT / ROUTINE
Justify the classification in 1–2 sentences.

**6. CLINICAL RECOMMENDATIONS**
Provide specific, actionable recommendations:
  - Immediate: What should the attending physician do now?
  - Imaging: Is follow-up X-ray, CT chest, or ultrasound warranted?
  - Laboratory: Which labs would help confirm (CBC, CRP, sputum culture, procalcitonin)?
  - Treatment: Any empirical management considerations?

End with the statement: \
"This AI-assisted report is based on deep learning analysis and must be correlated with \
clinical history, physical examination, and laboratory findings before clinical decisions are made."

Tone: Formal radiology report style. Address a clinician. No ML terminology.
Do NOT include any patient demographics header (e.g. Patient Name, Date of Study, Modality, Clinical Indication). Begin directly with section 1.
"""
    return {
        "pred_label":            pred_label,
        "confidence":            confidence,
        "clinical_findings":     clinical_findings,
        "high_attention_region": high_attention_region,
        "da_head_summary":       da_head_summary,
        "gemini_prompt":         gemini_prompt,
    }


# ══════════════════════════════════════════════════════════════════
# 2.  HEART DISEASE STRUCTURED REPORT
# ══════════════════════════════════════════════════════════════════

_FEATURE_DESCRIPTIONS = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol":     "Serum Cholesterol (mg/dL)",
    "fbs":      "Fasting Blood Sugar >120 mg/dL (1=yes)",
    "restecg":  "Resting ECG (0=normal, 1=ST-T wave abnormality, 2=LVH)",
    "thalach":  "Max Heart Rate Achieved (bpm)",
    "exang":    "Exercise-Induced Angina (1=yes)",
    "oldpeak":  "ST Depression (exercise vs rest)",
    "slope":    "Slope of Peak ST Segment (0=upsloping, 1=flat, 2=downsloping)",
    "ca":       "Major Vessels Coloured by Fluoroscopy (0–4)",
    "thal":     "Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)",
}

_SHAP_DIRECTION = {True: "↑ increases risk", False: "↓ reduces risk"}


def build_heart_report_data(model_out: dict, features: list,
                             input_values: list, shap_vals: list,
                             pred_label: str, prob: float) -> dict:
    """
    Assembles a structured data dict from TabularNN forward pass + SHAP values.

    Returns keys:
      pred_label, prob, clinical_findings, feature_contributions,
      da_head_summary, gemini_prompt
    """
    lem_probs = _sigmoid_probs(model_out["lem_logits"])   # (5,)

    clinical_findings = {
        label: float(prob_)
        for label, prob_ in zip(HEART_CLINICAL_LABELS, lem_probs)
    }

    da_head_summary = _head_weight_summary(model_out["da_weights"])

    # Top-5 SHAP features
    sv = [v[0] if isinstance(v, list) else v for v in shap_vals]
    pairs = sorted(zip(features, sv, input_values),
                   key=lambda t: abs(t[1]), reverse=True)
    top5 = pairs[:5]

    feature_contributions = {
        feat: {"shap": float(shap), "value": float(val)}
        for feat, shap, val in top5
    }

    # Build complete patient profile for all 13 features
    all_features_dict = dict(zip(features, input_values))
    all_feat_lines = "\n".join(
        f"  {_FEATURE_DESCRIPTIONS.get(f, f):<55} = {all_features_dict.get(f, 0.0):.1f}"
        for f in features
    )

    # Clinical thresholds for contextual notes
    def _clinical_flag(feat, val):
        flags = {
            "trestbps": ("elevated (Stage 1 HTN)" if val >= 130 else "normal"   if val < 120 else "elevated-normal"),
            "chol":     ("high risk" if val >= 240 else "borderline" if val >= 200 else "optimal"),
            "thalach":  ("below expected max" if val < 100 else "within expected range"),
            "oldpeak":  ("significant ST depression" if val >= 2.0 else "mild ST change" if val > 0 else "no ST depression"),
            "ca":       (f"{int(val)} vessel(s) with stenosis" if val > 0 else "no stenotic vessels"),
            "fbs":      ("fasting glucose >120 mg/dL (diabetic range)" if val == 1 else "fasting glucose normal"),
            "exang":    ("exercise-induced angina present" if val == 1 else "no exercise-induced angina"),
        }
        return flags.get(feat, "")

    clinical_context_lines = []
    for f, s, v in top5:
        flag = _clinical_flag(f, v)
        desc = _FEATURE_DESCRIPTIONS.get(f, f)
        direction = _SHAP_DIRECTION[s > 0]
        note = f" [{flag}]" if flag else ""
        clinical_context_lines.append(
            f"  • {desc} = {v:.1f}{note}\n"
            f"    SHAP: {s:+.4f} → {direction}"
        )
    feat_lines = "\n".join(clinical_context_lines)

    # LEM with severity tiers
    lem_high = {k: v for k, v in clinical_findings.items() if v > 0.65}
    lem_mod  = {k: v for k, v in clinical_findings.items() if 0.40 < v <= 0.65}
    lem_low  = {k: v for k, v in clinical_findings.items() if v <= 0.40}

    def _fmt_lem(d):
        return "\n".join(f"    - {k.replace('_',' ').title()}: {v:.1%}" for k, v in d.items()) or "    (none)"

    active_lem = [k for k, v in clinical_findings.items() if v > 0.5]
    active_str = ", ".join(active_lem) if active_lem else "none detected above threshold"

    gemini_prompt = f"""You are a senior cardiologist reviewing an AI-assisted cardiac risk assessment. \
A DA-SPL–inspired Tabular Dual-Attention neural network with Label Enhancement Module (LEM) \
has analysed structured patient data from 13 validated cardiac biomarkers (UCI Heart Disease dataset).

Your role is to produce a detailed, clinically grounded cardiac risk report strictly based on the \
data below. Do NOT speculate beyond what the evidence supports.

════════════════════════════════════════
 PATIENT DATA — ALL 13 BIOMARKERS
════════════════════════════════════════
{all_feat_lines}
════════════════════════════════════════
 MODEL PREDICTION
════════════════════════════════════════
Prediction  : {pred_label}
Confidence  : {prob:.1%}  {'(HIGH confidence)' if prob > 0.85 else '(MODERATE — interpret carefully)' if prob > 0.65 else '(LOW — elevated clinical scrutiny required)'}

Risk Tier   : {'HIGH RISK' if prob > 0.75 and pred_label == 'Heart Disease' else 'MODERATE RISK' if prob > 0.50 and pred_label == 'Heart Disease' else 'LOW-MODERATE RISK' if pred_label == 'Heart Disease' else 'LOW RISK'}

Dual-Attention Head Weights (feature group emphasis):
{da_head_summary}

════════════════════════════════════════
 TOP-5 INFLUENTIAL FEATURES (SHAP)
════════════════════════════════════════
{feat_lines}

════════════════════════════════════════
 LABEL ENHANCEMENT MODULE — CARDIAC INDICATORS
════════════════════════════════════════
  HIGH SIGNAL (>65%):
{_fmt_lem(lem_high)}
  MODERATE SIGNAL (40–65%):
{_fmt_lem(lem_mod)}
  LOW SIGNAL (<40%):
{_fmt_lem(lem_low)}
Active indicators (>50%): {active_str}
════════════════════════════════════════

════════════════════════════════════════
 YOUR TASK — STRUCTURED CARDIAC REPORT
════════════════════════════════════════
Write a thorough, structured cardiac risk report. Each section must be substantive.

**1. OVERALL RISK ASSESSMENT**
State the prediction, confidence level, and risk tier. Classify as LOW / MODERATE / HIGH / VERY HIGH risk. \
Explain what the confidence level means for clinical decision-making.

**2. KEY RISK DRIVERS — CLINICAL INTERPRETATION**
For each of the top-5 SHAP features:
  - State the actual measured value and whether it is within/outside normal clinical ranges
  - Explain the pathophysiological significance of this value
  - Describe how it contributes to (or protects against) cardiovascular disease
Be specific: "A resting BP of X mmHg represents Stage Y hypertension, which..."

**3. CARDIAC RISK INDICATORS (LEM ANALYSIS)**
For each LEM indicator, regardless of probability level:
  - Typical Angina: Describe the symptom pattern and its classic CAD implication
  - Elevated Heart Rate: Discuss chronotropic implications and autonomic tone
  - Exercise-Induced Angina: Address myocardial ischaemia under stress
  - ST Depression: Discuss subendocardial ischaemia and its ECG significance
  - Vessel Narrowing: Discuss coronary artery stenosis burden and its prognostic weight
For each, relate the probability to what it means clinically (high/mod/low signal).

**4. DUAL-ATTENTION ANALYSIS**
Interpret which feature groups the model weighted most heavily. Explain the clinical \
coherence — do the high-attention features align with known CAD risk factor clustering? \
Are there any surprising head weight patterns worth flagging?

**5. DIFFERENTIAL CONSIDERATIONS**
Given this profile, list 2–3 conditions the cardiologist should consider:
  - Stable Angina / Unstable Angina / ACS
  - Non-cardiac chest pain
  - Silent ischaemia in diabetic/elderly patients
For each, state which features support or argue against it.

**6. SEVERITY & URGENCY**
Classify: URGENT (admit/immediate workup) / SEMI-URGENT (within 24–72 hrs) / OUTPATIENT
Justify clearly based on the data.

**7. CLINICAL RECOMMENDATIONS**
Provide specific, actionable next steps:
  - Immediate investigations (ECG, troponin, echocardiogram, stress test, coronary angiography)
  - Laboratory workup (lipid panel, HbA1c, CRP, BNP if indicated)
  - Risk modification (hypertension management, statin therapy, antiplatelet if appropriate)
  - Follow-up timeline and specialist referral recommendation

Tone: Formal cardiology consultation note. Address a clinician. No ML terminology.
Do NOT include any patient demographics header (e.g. Patient Name, Date, MRN, Referring Physician). Begin directly with section 1.

End with: "This AI-assisted cardiac risk assessment is intended to support — not replace — \
clinical judgement. All findings must be correlated with the patient's full medical history, \
physical examination, symptoms, and validated diagnostic investigations."
"""
    return {
        "pred_label":           pred_label,
        "prob":                 prob,
        "clinical_findings":    clinical_findings,
        "feature_contributions":feature_contributions,
        "da_head_summary":      da_head_summary,
        "gemini_prompt":        gemini_prompt,
    }
