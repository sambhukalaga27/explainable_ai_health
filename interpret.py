# interpret.py  —  Grad-CAM + Dual-Attention heatmap + SHAP
# ——————————————————————————————————————————————————————————————————
import torch
import torch.nn.functional as F
import numpy as np
import shap
import cv2


# ══════════════════════════════════════════════════════════════════
# 1.  GRAD-CAM  (unchanged, works on any CNN)
# ══════════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        # DA-SPL forward returns a dict; Grad-CAM needs the primary logits
        output_raw = self.model(input_tensor)
        if isinstance(output_raw, dict):
            output = output_raw["logits1"]
        else:
            output = output_raw

        if class_idx is None:
            class_idx = output.argmax().item()

        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()

        weights   = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_map   = (weights * self.activations).sum(dim=1, keepdim=True)
        cam       = F.relu(cam_map)
        cam       = cam - cam.min()
        cam       = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()


# ══════════════════════════════════════════════════════════════════
# 2.  DUAL-ATTENTION HEATMAP
#     Visualises which spatial tokens (7×7 grid) the DAM attended to.
#     Based on the mean attention weight across heads / tokens.
# ══════════════════════════════════════════════════════════════════
def get_dual_attention_heatmap(model, input_tensor,
                                target_size: int = 224) -> np.ndarray:
    """
    Returns an (H, W) float32 ndarray in [0,1] representing the
    aggregated dual-attention weights over the 7×7 spatial token grid.

    Steps (paper §III-B):
      1. Run forward pass to get attn_map  (B, H, N, N) and da_weights (H,)
      2. Weight attention heads by da_weights (the dual-weight output)
      3. Average over heads and query tokens  →  (N,)  = 49 scores
      4. Reshape to (7, 7), up-sample to target_size
    """
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)

    attn_map  = out["attn_map"]     # B, H, N, N
    da_w      = out["da_weights"]   # H,

    # Weighted sum over heads
    # attn_map: B,H,N,N  ×  da_w: H  →  B,N,N
    weighted_attn = (attn_map * da_w.view(1, -1, 1, 1)).sum(dim=1)  # B,N,N

    # Average over query positions  →  B, N
    token_importance = weighted_attn[0].mean(dim=0).cpu().numpy()   # N=49

    # Reshape to spatial grid (7×7)
    grid_size = int(token_importance.shape[0] ** 0.5)
    heat      = token_importance.reshape(grid_size, grid_size)
    heat      = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # Up-sample to image resolution
    heat_up = cv2.resize(heat, (target_size, target_size),
                         interpolation=cv2.INTER_CUBIC)
    heat_up = np.clip(heat_up, 0, 1)
    return heat_up.astype(np.float32)


def attention_to_rgb_overlay(original_rgb: np.ndarray,
                              heat: np.ndarray,
                              alpha: float = 0.45) -> np.ndarray:
    """
    Overlays the dual-attention heatmap on the original RGB image.

    original_rgb : H, W, 3  uint8
    heat         : H, W     float32  in [0,1]
    Returns      : H, W, 3  uint8   blended overlay
    """
    colormap = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_PLASMA)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    if original_rgb.dtype != np.uint8:
        original_rgb = original_rgb.astype(np.uint8)
    if colormap.shape != original_rgb.shape:
        colormap = cv2.resize(colormap,
                              (original_rgb.shape[1], original_rgb.shape[0]))
    return cv2.addWeighted(original_rgb, 1 - alpha, colormap, alpha, 0)


# ══════════════════════════════════════════════════════════════════
# 3.  FEATURE ATTENTION BAR  (Tabular)
#     Returns per-feature attention scores from the DAM for
#     visualization in the app (no SHAP needed for this view).
# ══════════════════════════════════════════════════════════════════
def get_feature_attention_scores(model, input_tensor) -> np.ndarray:
    """
    Returns (num_features,) ndarray of attention importance scores.
    Uses the dual-weighted attention map from TabularNN.
    """
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)

    attn_map = out["attn_map"]    # B, H, F, F   (F = num features = 13)
    da_w     = out["da_weights"]  # H,

    weighted = (attn_map * da_w.view(1, -1, 1, 1)).sum(dim=1)  # B, F, F
    scores   = weighted[0].mean(dim=0).cpu().numpy()             # F,
    scores   = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


# ══════════════════════════════════════════════════════════════════
# 4.  SHAP  (GradientExplainer for TabularNN)
# ══════════════════════════════════════════════════════════════════
def get_shap_values(model, background_data, input_data, pred_class):
    """Returns SHAP values for the primary output stream (logits1)."""

    # Wrapper to extract logits1 from the dict output
    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)["logits1"]

    wrapped = _LogitsWrapper(model).to('cpu').eval()
    background_data = background_data.detach().cpu()
    input_data      = input_data.detach().cpu()

    explainer  = shap.GradientExplainer(wrapped, background_data)
    shap_vals  = explainer.shap_values(input_data)
    return shap_vals[0]
