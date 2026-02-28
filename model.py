# model.py  —  DA-SPL inspired architecture
# Implements: Dual-Attention Module (DAM) + Parallel Classifiers (PLN analog)
#             + Label Enhancement Module (LEM), adapted from:
# "Automated Glaucoma Report Generation via DA-SPL" (arXiv:2510.10037)
# ——————————————————————————————————————————————————————————————————————————
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


# ══════════════════════════════════════════════════════════════════
# 1.  DUAL-ATTENTION MODULE  (DAM)
#     Paper §III-B, Eq.(1)–(8)
#
#     Two-step head re-weighting:
#       Step-1  Learned weights  wₐ  (Eq.3-4)   ← trained end-to-end
#       Step-2  Cosine dual-weights  w_dwa (Eq.5-8) ← emphasise heads
#               aligned with the "best" (highest-norm) head
# ══════════════════════════════════════════════════════════════════
class DualAttentionModule(nn.Module):
    """
    Multi-head self-attention with dual-weight head re-weighting (DAM).

    Input shape : (B, N, C)   B=batch, N=tokens, C=feature dim
    Output      : (B, N, C), attention_map (B, H, N, N), head_weights (H,)
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.H = num_heads
        self.D = embed_dim // num_heads          # per-head dim
        self.scale = self.D ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Eq.(3)  —  learnable per-head weights  wₐ  (H,)
        self.head_weights = nn.Parameter(torch.ones(num_heads))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        H, D = self.H, self.D

        # ── project & reshape to (B, H, N, D) ──
        def proj_split(lin):
            return lin(x).reshape(B, N, H, D).permute(0, 2, 1, 3)

        Q, K, V = proj_split(self.q), proj_split(self.k), proj_split(self.v)

        # ── Eq.(1)-(2)  standard multi-head attention ──
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # B,H,N,N
        attn_map    = torch.softmax(attn_scores, dim=-1)
        attn_map    = self.attn_drop(attn_map)
        head_out    = torch.matmul(attn_map, V)                            # B,H,N,D

        # ── Eq.(3)-(4)  Step-1: learned head weights ──
        wa = torch.softmax(self.head_weights, dim=0)                       # H,

        # ── Eq.(5)-(8)  Step-2: cosine dual-weight  w_dwa ──
        # Flatten each head to a 1-D vector for cosine comparison
        ho_flat  = head_out.permute(1, 0, 2, 3).reshape(H, -1)            # H, B*N*D
        norms    = ho_flat.norm(dim=1)                                     # H,
        best_idx = norms.argmax()
        best_vec = ho_flat[best_idx].unsqueeze(0).expand(H, -1)           # H, B*N*D

        w_cos = F.cosine_similarity(ho_flat, best_vec, dim=1).clamp(min=0) # H,
        beta  = w_cos.mean()                                               # scalar  Eq.(7)
        w_dwa = F.relu(beta - w_cos)                                       # Eq.(8)  H,

        # Combined final per-head weight
        final_w = (wa + w_dwa)                                             # H,
        final_w = final_w / (final_w.sum() + 1e-8)

        # Apply weights and merge heads  →  B, N, C
        weighted = head_out * final_w.view(1, H, 1, 1)                    # B,H,N,D
        merged   = weighted.permute(0, 2, 1, 3).reshape(B, N, H * D)      # B,N,C
        out      = self.out_proj(merged)

        return out, attn_map, final_w.detach()


# ══════════════════════════════════════════════════════════════════
# 2.  LABEL ENHANCEMENT MODULE  (LEM)
#     Paper §III-C, Eq.(14)-(16)
#
#     Predicts disorder-specific clinical labels from the pooled
#     feature vector and adds  losst = Σ BCE(pred, gt_label)
#     to the training objective, forcing the representation to
#     encode medically meaningful attributes.
# ══════════════════════════════════════════════════════════════════
class LabelEnhancementModule(nn.Module):
    """
    Lightweight MLP classifier that predicts multi-label clinical
    findings from the pooled feature vector.

    num_labels : number of clinical sub-labels to predict
    """

    def __init__(self, in_features: int, num_labels: int, hidden: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),                          # FFN uses GeLU (paper §III-A)
            nn.Dropout(0.2),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, num_labels). Use sigmoid for probs."""
        return self.fc(feat)


# ══════════════════════════════════════════════════════════════════
# 3.  CHEST X-RAY MODEL  —  DA-SPL adapted
#     Backbone : ResNet-18 (frozen conv layers, fine-tune from layer3)
#     Tokens   : 7×7 spatial positions from layer4  →  49 tokens × 512 dim
#     Two classification streams (PLN analog, Eq.10–13)
#     LEM      : 4 pneumonia-relevant clinical sub-labels
# ══════════════════════════════════════════════════════════════════

# Clinical sub-label names for Chest X-ray LEM
XRAY_CLINICAL_LABELS = [
    "lung_opacity",       # diffuse opacity — hallmark of pneumonia
    "consolidation",      # lobar/segmental consolidation
    "air_bronchograms",   # air in bronchi surrounded by consolidation
    "pleural_effusion",   # fluid in pleural space
]

def _make_xray_lem_target(labels: torch.Tensor) -> torch.Tensor:
    """
    Soft multi-label targets derived from the binary PNEUMONIA(1)/NORMAL(0) label.
    Evidence-based approximate co-occurrence rates from radiology literature.
    """
    B = labels.size(0)
    targets = torch.zeros(B, 4, device=labels.device)
    pneum = (labels == 1).float()
    targets[:, 0] = pneum * 0.95          # lung_opacity
    targets[:, 1] = pneum * 0.80          # consolidation
    targets[:, 2] = pneum * 0.65          # air_bronchograms
    targets[:, 3] = pneum * 0.35          # pleural_effusion
    return targets


class ChestXrayCNN(nn.Module):
    """
    DA-SPL-inspired chest X-ray classifier.

    Forward returns a dict with:
      'logits1' : primary classification logits  (B, 2)
      'logits2' : secondary stream logits        (B, 2)
      'lem_logits': clinical sub-label logits    (B, 4)
      'da_weights': final dual-attention head weights (H,)
      'attn_map'  : attention map from DAM       (B, H, N, N)
    """

    NUM_XRAY_LABELS = len(XRAY_CLINICAL_LABELS)

    def __init__(self, num_classes: int = 2, num_heads: int = 8):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Feature extractor (up to layer4 inclusive)
        self.feature_extractor = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        # Freeze early layers (conv1 → layer2), fine-tune layer3/layer4
        for name, module in self.feature_extractor.named_children():
            if name in ("0", "1", "2", "3", "4", "5"):   # conv1..layer2
                for p in module.parameters():
                    p.requires_grad = False

        token_dim = 512   # layer4 output channels

        # Project tokens to attention embed_dim
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.LayerNorm(256),
        )

        # Dual-Attention Module (DAM)
        self.dam = DualAttentionModule(embed_dim=256, num_heads=num_heads)

        # Feed-Forward Network after attention (paper FFN with GeLU)
        self.ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        )
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)

        # ── Stream 1: primary classifier  (loss1) ──
        self.classifier1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

        # ── Stream 2: secondary classifier  (loss2) ──
        self.classifier2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

        # ── Label Enhancement Module  (LEM, losst) ──
        self.lem = LabelEnhancementModule(256, self.NUM_XRAY_LABELS)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        # ── backbone → spatial feature map (B, 512, 7, 7) ──
        feat_map = self.feature_extractor(x)                  # B,512,7,7
        B, C, H, W = feat_map.shape
        tokens = feat_map.flatten(2).permute(0, 2, 1)         # B,49,512

        # ── project to attention dim ──
        tokens = self.token_proj(tokens)                       # B,49,256

        # ── DAM with residual connection ──
        da_out, attn_map, da_weights = self.dam(tokens)
        tokens = self.norm1(tokens + da_out)                   # residual

        # ── FFN with residual connection ──
        tokens = self.norm2(tokens + self.ffn(tokens))

        # ── global average pool over spatial tokens ──
        pooled = tokens.mean(dim=1)                            # B,256

        # ── two classifier streams ──
        logits1 = self.classifier1(pooled)                     # B,2
        logits2 = self.classifier2(pooled)                     # B,2

        # ── LEM ──
        lem_logits = self.lem(pooled)                          # B,4

        return {
            "logits1":    logits1,
            "logits2":    logits2,
            "lem_logits": lem_logits,
            "da_weights": da_weights,
            "attn_map":   attn_map,
            "pooled":     pooled,
        }


# ══════════════════════════════════════════════════════════════════
# 4.  TABULAR HEART-DISEASE MODEL  —  DA-SPL adapted
#     Treats each of the 13 clinical features as an attention "token"
#     Two classification streams + LEM for cardiac-risk sub-labels
# ══════════════════════════════════════════════════════════════════

# Clinical sub-label names for Tabular LEM
HEART_CLINICAL_LABELS = [
    "typical_angina",          # cp=0 → classic angina
    "elevated_heart_rate",     # thalach > 150
    "exercise_induced_angina", # exang=1
    "ST_depression",           # oldpeak > 1.0
    "vessel_narrowing",        # ca > 0 → major vessels coloured by fluoroscopy
]

def _make_heart_lem_target(xb: torch.Tensor) -> torch.Tensor:
    """
    Rule-based soft labels derived from the raw (unscaled) input features.
    NOTE: called before scaling, so feature indices are:
      [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
           0   1   2    3      4   5      6       7     8      9     10  11   12
    """
    B = xb.size(0)
    t = torch.zeros(B, 5, device=xb.device)
    t[:, 0] = (xb[:, 2] == 0).float()                  # cp==0 → typical angina
    t[:, 1] = (xb[:, 7] > 150).float()                 # thalach > 150
    t[:, 2] = xb[:, 8].float()                         # exang
    t[:, 3] = (xb[:, 9] > 1.0).float()                 # oldpeak > 1
    t[:, 4] = (xb[:, 11] > 0).float()                  # ca > 0
    return t


class TabularNN(nn.Module):
    """
    DA-SPL-inspired tabular classifier.

    Each clinical feature becomes an attention token.
    Forward returns a dict with:
      'logits1', 'logits2', 'lem_logits', 'da_weights', 'attn_map', 'pooled'
    """

    NUM_HEART_LABELS = len(HEART_CLINICAL_LABELS)

    def __init__(self, input_dim: int = 13, num_classes: int = 2,
                 embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        # Project each scalar feature to embed_dim  →  B,13,64
        self.feature_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Dual-Attention Module (DAM)
        self.dam = DualAttentionModule(embed_dim=embed_dim, num_heads=num_heads)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ── Stream 1: primary classifier  (loss1) ──
        self.classifier1 = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

        # ── Stream 2: secondary classifier  (loss2) ──
        self.classifier2 = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

        # ── Label Enhancement Module  (LEM, losst) ──
        self.lem = LabelEnhancementModule(embed_dim, self.NUM_HEART_LABELS)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        B, F = x.shape                                         # B, 13

        # ── tokenise: each feature → embed vector ──
        tokens = self.feature_embed(x.unsqueeze(-1))           # B,13,64

        # ── DAM ──
        da_out, attn_map, da_weights = self.dam(tokens)
        tokens = self.norm1(tokens + da_out)

        # ── FFN ──
        tokens = self.norm2(tokens + self.ffn(tokens))

        # ── pool over feature tokens ──
        pooled = tokens.mean(dim=1)                            # B,64

        # ── two streams ──
        logits1 = self.classifier1(pooled)                     # B,2
        logits2 = self.classifier2(pooled)                     # B,2

        # ── LEM ──
        lem_logits = self.lem(pooled)                          # B,5

        return {
            "logits1":    logits1,
            "logits2":    logits2,
            "lem_logits": lem_logits,
            "da_weights": da_weights,
            "attn_map":   attn_map,
            "pooled":     pooled,
        }


# ══════════════════════════════════════════════════════════════════
# 5.  COMBINED LOSS  (Eq.15-16 from paper)
#     loss = loss1 + λ·loss2 + α·loss_t
# ══════════════════════════════════════════════════════════════════

def da_spl_loss(out: dict, labels: torch.Tensor,
                lem_targets: torch.Tensor,
                lam: float = 0.5, alpha: float = 5.0) -> torch.Tensor:
    """
    Combined DA-SPL training objective.

    out        : forward() dict with 'logits1', 'logits2', 'lem_logits'
    labels     : hard class labels  (B,)
    lem_targets: soft multi-label targets  (B, num_labels)  in [0,1]
    lam        : weight for secondary stream loss  (λ)
    alpha      : weight for label-enhancement loss  (α=5 per paper ablation)
    """
    ce = nn.CrossEntropyLoss()
    loss1 = ce(out["logits1"], labels)
    loss2 = ce(out["logits2"], labels)

    # losst  — binary cross-entropy per clinical sub-label  (Eq.16)
    loss_t = F.binary_cross_entropy_with_logits(
        out["lem_logits"], lem_targets
    )

    return loss1 + lam * loss2 + alpha * loss_t