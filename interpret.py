# interpret.py
import torch
import torch.nn.functional as F
import numpy as np
import shap

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

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax().item()

        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam_map = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(grad_cam_map)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()

# SHAP explainer for TabularNN
def get_shap_values(model, background_data, input_data, pred_class):
    # Move model to CPU
    model_cpu = model.to('cpu').eval()

    # Move data to CPU and detach
    background_data = background_data.detach().cpu()
    input_data = input_data.detach().cpu()

    # Use GradientExplainer for PyTorch models
    explainer = shap.GradientExplainer(model_cpu, background_data)

    # Get SHAP values
    shap_values = explainer.shap_values(input_data)

    return shap_values[0]
