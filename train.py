# train.py  —  DA-SPL training loop
# Trains both models with the combined objective:
#   loss = loss1 + λ·loss2 + α·loss_t   (Eq.15-16, arXiv:2510.10037)
# ——————————————————————————————————————————————————————————————————
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import (
    ChestXrayCNN, TabularNN,
    da_spl_loss, _make_xray_lem_target, _make_heart_lem_target
)
from data_loader import load_tabular_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("models", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════
# 1.  TRAIN CHEST X-RAY MODEL  (DA-SPL)
# ══════════════════════════════════════════════════════════════════
def train_cnn(epochs: int = 50, lam: float = 0.5, alpha: float = 5.0):
    img_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ]),
    }

    train_dataset = datasets.ImageFolder(
        'data/chest_pneumonia/train', transform=img_transforms['train'])
    val_dataset   = datasets.ImageFolder(
        'data/chest_pneumonia/val',   transform=img_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = ChestXrayCNN().to(device)
    # Warm-start from ResNet weights already loaded; use separate LR for backbone
    optimizer = optim.AdamW([
        {"params": model.feature_extractor.parameters(), "lr": 5e-5},
        {"params": model.token_proj.parameters()},
        {"params": model.dam.parameters()},
        {"params": model.ffn.parameters()},
        {"params": model.norm1.parameters()},
        {"params": model.norm2.parameters()},
        {"params": model.classifier1.parameters()},
        {"params": model.classifier2.parameters()},
        {"params": model.lem.parameters()},
    ], lr=1e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── training ──
        model.train()
        total_loss = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            lem_targets = _make_xray_lem_target(labels)

            optimizer.zero_grad()
            out  = model(images)
            loss = da_spl_loss(out, labels, lem_targets, lam=lam, alpha=alpha)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (out["logits1"].argmax(1) == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # ── validation ──
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                val_correct += (out["logits1"].argmax(1) == labels).sum().item()
                val_total   += labels.size(0)
        val_acc = val_correct / val_total

        print(f"[CNN] Epoch {epoch:3d}/{epochs}  "
              f"Loss: {total_loss/total:.4f}  "
              f"Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        # ── save best checkpoint ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/cnn.pth')
            print(f"           ✅ Best model saved (val_acc={val_acc:.4f})")

    print(f"\n[CNN] Training complete. Best Val Acc: {best_val_acc:.4f}")


# ══════════════════════════════════════════════════════════════════
# 2.  TRAIN TABULAR MODEL  (DA-SPL)
# ══════════════════════════════════════════════════════════════════
def train_tabular(epochs: int = 100, lam: float = 0.5, alpha: float = 5.0):
    train_loader, val_loader, test_loader, feature_cols, scaler = \
        load_tabular_data("data/tabular/heart.csv")

    input_dim = next(iter(train_loader))[0].shape[1]
    model     = TabularNN(input_dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0

    # Keep a copy of the raw (unscaled) batches for LEM target generation
    raw_train_loader, _, _, _, _ = load_tabular_data(
        "data/tabular/heart.csv", scale=False)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = correct = total = 0

        for (xb, yb), (raw_xb, _) in zip(train_loader, raw_train_loader):
            xb, yb          = xb.to(device), yb.to(device)
            raw_xb          = raw_xb.to(device)
            lem_targets     = _make_heart_lem_target(raw_xb)

            optimizer.zero_grad()
            out  = model(xb)
            loss = da_spl_loss(out, yb, lem_targets, lam=lam, alpha=alpha)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct    += (out["logits1"].argmax(1) == yb).sum().item()
            total      += yb.size(0)

        scheduler.step()

        # ── validation ──
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_correct += (out["logits1"].argmax(1) == yb).sum().item()
                val_total   += yb.size(0)
        val_acc = val_correct / val_total

        if epoch % 10 == 0 or epoch == epochs:
            print(f"[Tabular] Epoch {epoch:3d}/{epochs}  "
                  f"Loss: {total_loss/total:.4f}  "
                  f"Train Acc: {correct/total:.4f}  Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/tabular.pth')

    # ── full evaluation on validation set ──
    print("\n📊 Model Evaluation on Validation Set:")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            all_preds.extend(out["logits1"].argmax(1).cpu().numpy())
            all_labels.extend(yb.numpy())

    print("Accuracy Score:", accuracy_score(all_labels, all_preds))
    print("\nClassification Report:\n",
          classification_report(all_labels, all_preds, digits=4))
    print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print(f"\n[Tabular] Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    # train_cnn()
    train_tabular(epochs=300)

