# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ChestXrayCNN, TabularNN
from data_loader import load_tabular_data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train Chest X-ray Model
def train_cnn():
    img_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = datasets.ImageFolder('data/chest_pneumonia/train', transform=img_transforms['train'])
    val_dataset   = datasets.ImageFolder('data/chest_pneumonia/val',   transform=img_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

    model = ChestXrayCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, 50):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(f"[CNN] Epoch {epoch}, Loss: {total_loss / total:.4f}, Acc: {correct / total:.4f}")

    torch.save(model.state_dict(), 'models/cnn.pth')

# Train Tabular MLP Model
def train_tabular():
    train_loader, val_loader, _, _, _ = load_tabular_data("data/tabular/heart.csv")
    input_dim = next(iter(train_loader))[0].shape[1]
    model = TabularNN(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 100):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)

        print(f"[Tabular] Epoch {epoch}, Loss: {total_loss / total:.4f}, Acc: {correct / total:.4f}")

    torch.save(model.state_dict(), 'models/tabular.pth')
    # --------- Evaluation on Validation Set ----------
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    print("\n📊 Model Evaluation on Validation Set:")
    print("Accuracy Score:", accuracy_score(all_labels, all_preds))
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, digits=4))
    print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    train_cnn()
    train_tabular()
