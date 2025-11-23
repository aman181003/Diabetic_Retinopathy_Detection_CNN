# src/inference/predict.py
import os
import time
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import cv2
import torch
from torchvision import transforms

from src.models.model import get_resnet50, CustomCNN
from src.models.gradcam import GradCAM

# project-local tmp dir (portable)
TMP_DIR = os.path.join(os.getcwd(), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


def build_model(model_path, backbone="resnet50", device="cpu"):
    """
    Load a model (resnet50 or custom) from the .pth file. Handles 'module.' prefixes.
    """
    device = torch.device(device)
    if backbone == "resnet50":
        model = get_resnet50(n_classes=5, pretrained=False)
    else:
        model = CustomCNN(n_classes=5)

    # Load state dict, automatically handling 'module.' prefix from DataParallel models
    state = torch.load(model_path, map_location=device)
    if list(state.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove `module.`
            new_state[name] = v
        model.load_state_dict(new_state)
    else:
        model.load_state_dict(state)
        
    model.to(device).eval()
    return model


def preprocess_pil(pil_img, size=224):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return tf(pil_img).unsqueeze(0)


def predict_and_explain(model, pil_img, device="cpu", backbone="resnet50", img_size=224, return_heatmap=True):
    """
    Run inference and Grad-CAM explanation, including bounding box highlighting.
    """
    device = torch.device(device)
    x = preprocess_pil(pil_img, size=img_size).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    # Choose target layer for Grad-CAM
    target_layer = None
    if hasattr(model, "layer4"):
        try:
            target_layer = model.layer4[-1].conv3
        except Exception:
            target_layer = model.layer4[-1] 
    
    if target_layer is None:
        for m in reversed(list(model.modules())):
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
                break

    heatmap = None
    overlay_path = None
    orig_path = None
    
    if return_heatmap and target_layer is not None:
        cam = GradCAM(model, target_layer)
        cam_map = cam.generate(x, class_idx=pred_idx) 
        heatmap = cam_map.squeeze() if cam_map.ndim > 2 else cam_map

        # --- HIGHLIGHTING LOGIC ---

        orig_arr = np.array(pil_img.convert("RGB"))
        H, W = orig_arr.shape[:2]
        
        heatmap_u8 = (heatmap * 255).astype("uint8")
        heatmap_resized = cv2.resize(heatmap_u8, (W, H))
        heat_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        overlay_base = cv2.addWeighted(orig_arr, 0.6, heat_color, 0.4, 0)

        # Find the most intense region to highlight
        threshold_value = int(heatmap_u8.max() * 0.75) 
        _, thresh = cv2.threshold(heatmap_u8, threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        overlay_with_highlight = overlay_base.copy() 
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the bounding box coordinates for this contour, scaled to the original image size
            x_h, y_h, w_h, h_h = cv2.boundingRect(largest_contour)
            x1 = int(x_h * (W / heatmap_u8.shape[1]))
            y1 = int(y_h * (H / heatmap_u8.shape[0]))
            x2 = int((x_h + w_h) * (W / heatmap_u8.shape[1]))
            y2 = int((y_h + h_h) * (H / heatmap_u8.shape[0]))


            # Draw a highly visible rectangle and text
            cv2.rectangle(overlay_with_highlight, (x1, y1), (x2, y2), (0, 255, 255), 3)
            label_text = "Area of Interest"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(overlay_with_highlight, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0,0,0), -1)
            cv2.putText(overlay_with_highlight, label_text, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Save original and the NEW overlay with the highlight
        timestamp = int(time.time())
        orig_path = os.path.join(TMP_DIR, f"orig_{timestamp}.png")
        overlay_path = os.path.join(TMP_DIR, f"overlay_{timestamp}.png")
        Image.fromarray(orig_arr).save(orig_path)
        Image.fromarray(overlay_with_highlight).save(overlay_path) 
    else:
        # Fallback if no heatmap is generated
        orig_path = os.path.join(TMP_DIR, f"orig_{int(time.time())}.png")
        pil_img.convert("RGB").save(orig_path)

    result = {
        "pred_idx": pred_idx,
        "pred_label": LABELS[pred_idx],
        "probs": {LABELS[i]: float(probs[i]) for i in range(len(probs))},
        "orig_path": orig_path,
        "overlay_path": overlay_path,
        "heatmap": heatmap
    }
    return result, heatmap


# The `make_pdf_report` function is deprecated and kept as a placeholder if needed.
def make_pdf_report(*args, **kwargs):
    print("Legacy PDF report function called, but it is deprecated.")
    return None