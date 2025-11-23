# src/models/gradcam.py
import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks(target_layer)

    def _register_hooks(self, layer):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(layer.register_forward_hook(forward_hook))
        self.hook_handles.append(layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        loss = out[0, class_idx]
        loss.backward(retain_graph=True)
        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()
        weights = np.mean(grads, axis=(1,2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i,w in enumerate(weights):
            cam += w * acts[i]
        cam = np.maximum(cam,0)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = (cam - cam.min()) / (cam.max()-cam.min() + 1e-8)
        return cam
