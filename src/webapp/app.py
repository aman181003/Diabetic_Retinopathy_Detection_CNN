# src/webapp/app.py
import os
import sys
import io
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path to enable imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from flask import Flask, render_template, request, jsonify, send_file, url_for
from PIL import Image
import numpy as np
import torch

from src.inference.predict import build_model, predict_and_explain
from src.inference.report_generator import map_to_icdr

# PDF Generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ---- Config / paths ----
APP_ROOT = Path.cwd()
TMP_DIR = APP_ROOT / "tmp"
TMP_DIR.mkdir(exist_ok=True)
MODEL_PATH = os.environ.get("MODEL_PATH", str(APP_ROOT / "outputs" / "best_model.pth"))
BACKBONE = os.environ.get("BACKBONE", "resnet50")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Flask app ----
app = Flask(__name__, static_folder="static", template_folder="templates")

# ---- Load model (once) ----
print(f"[app] Loading model: {MODEL_PATH} backbone={BACKBONE} device={DEVICE}")
try:
    model = build_model(MODEL_PATH, backbone=BACKBONE, device=DEVICE)
    print("[app] Model loaded successfully.")
except Exception as e:
    print(f"[app] FATAL: Failed to load model at startup: {e}")
    model = None

# ---- Helpers ----
def numpy_to_pil(hm):
    import cv2
    hm_arr = np.array(hm)
    if hm_arr.max() <= 1.0: hm_arr = (hm_arr * 255).astype("uint8")
    else: hm_arr = hm_arr.astype("uint8")
    heat_color = cv2.applyColorMap(hm_arr, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB))

# ---- Professional PDF Generation ----
def create_professional_report(pdf_path, pil_img, overlay_path, report_json):
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=30, bottomMargin=50, leftMargin=40, rightMargin=40)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Diabetic Retinopathy Screening Report</b>", styles['h1']))
    story.append(Spacer(1, 10))

    p_info = report_json
    patient_data = [
        [Paragraph("<b>Patient Name:</b>", styles['Normal']), p_info.get("patient_name") or "N/A", Paragraph("<b>Patient ID:</b>", styles['Normal']), p_info.get("patient_id") or "N/A"],
        [Paragraph("<b>Age / Sex:</b>", styles['Normal']), f"{p_info.get('patient_age', 'N/A')} / {p_info.get('patient_sex', 'N/A')}", Paragraph("<b>Report Date:</b>", styles['Normal']), datetime.utcnow().strftime('%d-%b-%Y %H:%M UTC')],
        [Paragraph("<b>Visual Acuity:</b>", styles['Normal']), p_info.get("visual_acuity") or "N/A", Paragraph("<b>Affected Eye:</b>", styles['Normal']), p_info.get("laterality") or "N/A"],
    ]
    patient_table = Table(patient_data, colWidths=[90, 160, 90, 160])
    patient_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#F0F4FF")),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor("#F0F4FF")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>AI Screening Analysis</b>", styles['h2']))
    story.append(Spacer(1, 10))

    triage_color = {"emergency": "red", "urgent": "orange", "expedited": "blue", "routine": "green"}.get(p_info.get('triage', {}).get('level', 'routine'), 'black')
    result_data = [
        [Paragraph("<b>Predicted Condition:</b>", styles['Normal']), Paragraph(f"<b>{p_info.get('predicted_label')}</b>", styles['Normal'])],
        [Paragraph("<b>AI Confidence:</b>", styles['Normal']), f"{p_info.get('confidence', 'N/A')}% (Adjusted for image quality)"],
        [Paragraph("<b>Image Quality Score:</b>", styles['Normal']), f"{p_info.get('image_quality', {}).get('score', 'N/A')}/100"],
        [Paragraph("<b>Triage Recommendation:</b>", styles['Normal']), Paragraph(f"<font color='{triage_color}'><b>{p_info.get('triage', {}).get('level', 'N/A').upper()}</b></font> ({p_info.get('triage', {}).get('timeframe', 'N/A')})", styles['Normal'])],
    ]
    result_table = Table(result_data, colWidths=[150, 350])
    result_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0), (-1,-1), 0)]))
    story.append(result_table)
    story.append(Spacer(1, 20))
    
    tmp_orig = TMP_DIR / f"_pdf_orig_{int(time.time())}.png"
    pil_img.convert("RGB").save(tmp_orig)
    orig_img_report = PlatypusImage(str(tmp_orig), width=240, height=240, kind='proportional')
    
    img_data = [[orig_img_report]]
    captions = [[Paragraph("Original Fundus Image", styles['Normal'])]]
    if overlay_path and os.path.exists(overlay_path):
        overlay_img_report = PlatypusImage(overlay_path, width=240, height=240, kind='proportional')
        img_data[0].append(overlay_img_report)
        captions[0].append(Paragraph("Highlighted Area of Interest", styles['Normal']))

    img_table = Table(img_data, colWidths=[250, 250])
    img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    captions_table = Table(captions, colWidths=[250, 250])
    captions_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(img_table); story.append(captions_table)
    story.append(Spacer(1, 20))

    # Add Clinical Correlates to PDF
    correlates = p_info.get('clinical_correlates', {})
    story.append(Paragraph(f"<b>{correlates.get('title', 'Associated Clinical Signs')}</b>", styles['h2']))
    story.append(Paragraph(f"<i><font size=8>{correlates.get('disclaimer')}</font></i>", styles['Normal']))
    story.append(Spacer(1, 10))
    for point in correlates.get("points", []):
        story.append(Paragraph(f"• {point}", styles['Normal'])); story.append(Spacer(1, 5))
    story.append(Spacer(1, 20))
    
    recs = p_info.get('general_recommendations', {})
    story.append(Paragraph(f"<b>{recs.get('title', 'Recommendations')}</b>", styles['h2']))
    story.append(Paragraph(f"<i><font size=8>{recs.get('disclaimer')}</font></i>", styles['Normal']))
    story.append(Spacer(1, 10))
    for point in recs.get("points", []):
        story.append(Paragraph(f"• {point}", styles['Normal'])); story.append(Spacer(1, 5))
    story.append(Spacer(1, 20))


    disclaimer = "<b>DISCLAIMER:</b> This report is generated by an AI assistant (DR-Assist) and is intended for clinical decision support only. It is not a substitute for a comprehensive examination by a qualified ophthalmologist. All findings must be clinically correlated."
    story.append(Paragraph(disclaimer, styles['Italic']))

    doc.build(story)
    
    if os.path.exists(tmp_orig): os.remove(tmp_orig)


# ---- Routes ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None: return "Model not available on server.", 500
    if "file" not in request.files: return jsonify({"error": "no file provided"}), 400

    file = request.files["file"]
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"cannot open image: {e}"}), 400

    metadata = {k: v for k, v in request.form.items()}

    try:
        result, heatmap = predict_and_explain(model, pil_img, device=DEVICE, backbone=BACKBONE, return_heatmap=True)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    report_bundle = map_to_icdr(result["pred_idx"], result["probs"], heatmap=heatmap, pil_img=pil_img, metadata=metadata)
    machine_json = report_bundle.get("json", {})

    pdf_name = f"DR_Report_{metadata.get('patient_id') or int(time.time())}.pdf"
    pdf_path = TMP_DIR / pdf_name
    try:
        # Pass the overlay path to the PDF generator
        create_professional_report(pdf_path, pil_img, result.get("overlay_path"), machine_json)
    except Exception as e:
        print(f"Error creating PDF: {e}")

    # The full machine_json is now what we send to the frontend
    response = {
        "json": machine_json,
        "patient": report_bundle.get("patient"),
        "clinician": report_bundle.get("clinician"),
        "overlay_url": url_for("tmp_file", filename=os.path.basename(result["overlay_path"])) if result.get("overlay_path") else None,
        "orig_url": url_for("tmp_file", filename=os.path.basename(result["orig_path"])),
        "report_url": url_for("download_report", filename=pdf_name)
    }
    return jsonify(response)


@app.route("/tmp/<path:filename>")
def tmp_file(filename):
    return send_file(TMP_DIR / filename)

@app.route("/download/<path:filename>")
def download_report(filename):
    return send_file(TMP_DIR / filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)