# model.py
import torch
import torch.nn as nn
import torchvision.models as models

# CNN Model for Chest X-rays
class ChestXrayCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ChestXrayCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# MLP Model for Tabular Heart Data
class TabularNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(TabularNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# class MultimodalNet(nn.Module):
#     def __init__(self, tabular_input_dim, num_classes=2):
#         super().__init__()
#         # CNN branch
#         self.cnn = models.resnet18(pretrained=True)
#         for param in self.cnn.parameters():
#             param.requires_grad = False
#         self.cnn.fc = nn.Identity()  # remove classification layer
#         cnn_output_dim = 512

#         # Tabular branch
#         self.tabular = nn.Sequential(
#             nn.Linear(tabular_input_dim, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#         )
#         tab_output_dim = 32

#         # Fusion layer
#         self.classifier = nn.Sequential(
#             nn.Linear(cnn_output_dim + tab_output_dim, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, image, tabular_data):
#         image_feat = self.cnn(image)
#         tabular_feat = self.tabular(tabular_data)
#         combined = torch.cat((image_feat, tabular_feat), dim=1)
#         return self.classifier(combined)