# data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_tabular_data(csv_path, batch_size=32):
    df = pd.read_csv(csv_path)

    X = df.drop(columns='target')  # Change 'target' to whatever the target column is
    y = df['target']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def to_tensor(dataX, datay):
        return TensorDataset(torch.tensor(dataX, dtype=torch.float32),
                             torch.tensor(datay.values, dtype=torch.long))

    train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(to_tensor(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, X.columns.tolist(), scaler
