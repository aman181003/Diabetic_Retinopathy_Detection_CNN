# src/inference/report_generator.py
"""
Final version with detailed analysis, heatmap interpretation, and robust data handling.
"""
import os
import math
import numpy as np
from datetime import datetime
import cv2
import time

LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

def score_image_quality(pil_img=None, heatmap=None):
    """
    Simple heuristics for image quality score (0-100) and flags.
    """
    flags = []
    score = 100
    if heatmap is None or np.array(heatmap).max() < 0.1:
        flags.append("low_signal_from_heatmap")
        score -= 30
    
    try:
        if pil_img is not None:
            # Use a copy for analysis to avoid modifying the original
            arr = np.asarray(pil_img.copy().convert("L")).astype(np.float32)
            mean_brightness = arr.mean()
            if mean_brightness < 40:
                flags.append("underexposed")
                score -= 20
            if mean_brightness > 215:
                flags.append("overexposed")
                score -= 20
            
            # Use a resized version for faster blur detection
            arr_resized = cv2.resize(arr, (224, 224))
            lap_var = cv2.Laplacian(arr_resized, cv2.CV_64F).var()
            if lap_var < 50:
                flags.append("blur_detected")
                score -= 25
    except Exception as e:
        print(f"Could not perform image quality checks: {e}")
        flags.append("quality_check_failed")
        score -= 10
    
    score = max(0, min(100, score))
    return {"score": int(score), "flags": flags if flags else ["ok"]}

def get_clinical_correlates(pred_idx):
    """
    Provides a list of common clinical signs associated with the predicted DR grade.
    """
    correlates = {
        "title": "Associated Clinical Signs",
        "disclaimer": "The following are common findings for this grade and may not all be present. For educational and contextual purposes only.",
        "points": []
    }
    if pred_idx == 0: # No DR
        correlates["points"] = ["No visible microaneurysms, hemorrhages, or other signs of retinopathy."]
    elif pred_idx == 1: # Mild
        correlates["points"] = ["Presence of microaneurysms (small red dots).", "Occasional dot and blot hemorrhages may be seen."]
    elif pred_idx == 2: # Moderate
        correlates["points"] = [
            "More extensive microaneurysms and hemorrhages.",
            "Cotton wool spots (fluffy white patches) may be present.",
            "Venous beading or looping might be observed.",
            "Risk of Clinically Significant Macular Edema (CSME) increases."
        ]
    elif pred_idx == 3: # Severe
        correlates["points"] = [
            "Significant hemorrhages in all four retinal quadrants.",
            "Definite venous beading in two or more quadrants.",
            "Intraretinal Microvascular Abnormalities (IRMA) are prominent.",
            "High risk of progression to Proliferative DR."
        ]
    else: # Proliferative
        correlates["points"] = [
            "Neovascularization (growth of new, abnormal blood vessels).",
            "Vitreous or pre-retinal hemorrhage is common.",
            "Fibrous tissue proliferation may be visible.",
            "High risk of tractional retinal detachment and severe vision loss."
        ]
    return correlates

def get_general_recommendations(pred_idx):
    """
    Provides safe, non-prescriptive general recommendations based on DR grade.
    """
    recs = {
        "title": "General Recommendations",
        "disclaimer": "The following are general management guidelines and not a direct medical prescription. All treatment plans must be determined by a qualified ophthalmologist.",
        "points": []
    }
    if pred_idx == 0: # No DR
        recs["title"] = "Preventative Care Recommendations"
        recs["points"] = [
            "Strict glycemic control (maintain target HbA1c levels as advised by your physician).",
            "Annual or biennial diabetic eye screening is recommended.",
            "Regular blood pressure and cholesterol management."
        ]
    elif pred_idx == 1: # Mild
        recs["points"] = [
            "Optimize glycemic, blood pressure, and lipid control to reduce progression risk.",
            "Follow-up with an eye care professional in 6-12 months for repeat imaging and examination.",
            "Patient should be educated on the importance of regular monitoring and recognizing symptoms."
        ]
    elif pred_idx == 2: # Moderate
        recs["points"] = [
            "Consultation with an ophthalmologist is recommended to assess for macular edema.",
            "If clinically significant macular edema (CSME) is present, treatment may include intravitreal anti-VEGF injections or focal laser photocoagulation.",
            "More frequent monitoring (e.g., every 3-6 months) is typically advised."
        ]
    else: # Severe / PDR
        recs["title"] = "Urgent Management Recommendations"
        recs["points"] = [
            "Urgent referral to a retina specialist is critical to prevent vision loss.",
            "Pan-retinal photocoagulation (PRP) is a standard treatment to manage Proliferative DR.",
            "Intravitreal anti-VEGF injections are often used, especially with macular edema or vitreous hemorrhage.",
            "Surgical intervention (vitrectomy) may be required for advanced complications like non-clearing hemorrhage."
        ]
    return recs

def analyze_heatmap_region(heatmap_np, original_image_shape, laterality="OD", img_size=224):
    """
    Analyzes the 'hottest' region of a heatmap to extract location, extent, and intensity.
    """
    if heatmap_np is None or heatmap_np.max() < 0.15:
        return None

    H, W = original_image_shape[:2]
    heatmap_u8 = (heatmap_np * 255).astype("uint8")
    threshold_value = int(heatmap_u8.max() * 0.70)
    _, thresh = cv2.threshold(heatmap_u8, threshold_value, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return None

    largest_contour = max(contours, key=cv2.contourArea)
    x_h, y_h, w_h, h_h = cv2.boundingRect(largest_contour)
    center_x, center_y = x_h + w_h / 2, y_h + h_h / 2

    vertical = "Superior" if center_y < img_size / 3 else "Inferior" if center_y > img_size * 2 / 3 else "Central/Macular"
    
    location_desc = vertical
    if "Central" not in vertical:
        is_right_eye = "OS" not in laterality.upper()
        if (center_x > img_size / 2 and is_right_eye) or (center_x < img_size / 2 and not is_right_eye):
            horizontal = "Nasal"
        else:
            horizontal = "Temporal"
        location_desc = f"{vertical}-{horizontal} Quadrant"

    area_ratio = cv2.contourArea(largest_contour) / (img_size * img_size)
    extent_desc = "Diffuse" if area_ratio > 0.1 else "Moderate" if area_ratio > 0.02 else "Focal"

    mask = np.zeros_like(heatmap_np, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)
    mean_activation = heatmap_np[mask > 0].mean()

    return {
        "location": location_desc,
        "extent": extent_desc,
        "relative_intensity": f"{mean_activation:.2f} (avg activation)",
        "bounding_box_scaled": [int(x_h * (W / img_size)), int(y_h * (H / img_size)), int(w_h * (W / img_size)), int(h_h * (H / img_size))]
    }

def map_to_icdr(pred_idx, probs, heatmap=None, pil_img=None, metadata=None):
    """
    Builds the complete JSON report with detailed analysis.
    """
    if metadata is None: metadata = {}
    
    label = LABELS.get(pred_idx, "Ungradable")
    # Fix: robustly access probability by label if probs is dict, fallback to index if probs is list or tuple.
    if isinstance(probs, dict):
        prob = float(probs.get(label, 0))
    elif isinstance(probs, (list, tuple, np.ndarray)):
        try: prob = float(probs[pred_idx])
        except Exception: prob = 0
    else:
        prob = 0

    iq = score_image_quality(pil_img=pil_img, heatmap=heatmap)
    confidence = int(round(prob * 100))
    
    # Adjust confidence based on image quality
    if iq["score"] < 50:
        confidence = int(max(0, confidence - 25))
    elif iq["score"] < 75:
        confidence = int(max(0, confidence - 10))


    heatmap_analysis = analyze_heatmap_region(heatmap, np.array(pil_img).shape, metadata.get("laterality", "OD"))
    # Top-3 predictions sorted by probability
    if isinstance(probs, dict):
        top3_preds_list = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_preds_dict = {label: float(prob) for label, prob in top3_preds_list}
    else:
        # Ensure probs is a list/tuple of numbers before sorting
        prob_list = list(probs) if isinstance(probs, (np.ndarray)) else probs
        top3_preds_list = sorted(list(enumerate(prob_list)), key=lambda x: x[1], reverse=True)[:3]
        top3_preds_dict = {LABELS[i]: float(p) for i, p in top3_preds_list}

    interpretation_text = f"The AI model's primary prediction is <b>{label}</b> with {confidence}% confidence. "
    if len(top3_preds_list) > 1:
        # Handle both dict and list/tuple formats for probs
        if isinstance(probs, dict):
            next_pred_label = top3_preds_list[1][0]  # Already a label string
            next_pred_prob = int(top3_preds_list[1][1] * 100)
        else:
            next_pred_label = LABELS[top3_preds_list[1][0]]  # Index needs lookup
            next_pred_prob = int(top3_preds_list[1][1] * 100)
        interpretation_text += f"The next most likely finding was considered to be <b>{next_pred_label}</b> ({next_pred_prob}%). "

    if heatmap_analysis:
        interpretation_text += f"Visual analysis using Grad-CAM indicates the model focused on a <b>{heatmap_analysis['extent'].lower()}</b> area of interest in the <b>{heatmap_analysis['location'].lower()}</b>. "
        if pred_idx >= 2 and "Central/Macular" in heatmap_analysis['location']:
            interpretation_text += "This central focus may suggest a risk of macular involvement, which requires clinical correlation."
    elif pred_idx > 0:
        interpretation_text += "No single high-confidence area of interest could be isolated, suggesting potentially diffuse or subtle features."
    else:
        interpretation_text += "The model found no specific regions of interest indicative of retinopathy."
    
    if iq['score'] < 60:
        interpretation_text += f"<br><b>Warning:</b> Low image quality score ({iq['score']}/100) may impact prediction accuracy. Flags: {', '.join(iq['flags'])}."


    # Triage and Explanation logic
    triage = {}
    if label == "No DR": triage = {"level": "routine", "timeframe": "12 months", "who": "Diabetic Eye Screening"}
    elif label == "Mild": triage = {"level": "routine", "timeframe": "6-12 months", "who": "Optometrist/Ophthalmologist"}
    elif label == "Moderate": triage = {"level": "expedited", "timeframe": "2-4 weeks", "who": "Ophthalmology Clinic"}
    elif label == "Severe": triage = {"level": "urgent", "timeframe": "within 1 week", "who": "Retina Specialist"}
    elif label == "Proliferative": triage = {"level": "emergency", "timeframe": "within 48 hours", "who": "Retina Specialist (Emergency)"}
    else: triage = {"level": "unknown", "timeframe": "re-image", "who": "Technician/Clinician"}
    
    explanation = [f"Prediction of {label} is based on features detected by the model."]
    if iq["score"] < 50: explanation.append(f"Image quality is poor ({', '.join(iq['flags'])}), reducing confidence.")
    
    json_out = {
        **metadata,
        "image_id": metadata.get("image_id") or f"img_{int(time.time())}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "image_quality": iq,
        "predicted_label": label,
        "confidence": confidence,
        "triage": triage,
        "explanation_detailed": explanation,
        "general_recommendations": get_general_recommendations(pred_idx),
        "clinical_correlates": get_clinical_correlates(pred_idx),
        "detailed_analysis": {
            "interpretation_summary": interpretation_text,
            "top_3_predictions": top3_preds_dict,
            "heatmap_findings": heatmap_analysis
        },
        "disclaimer": "This is decision-support only. Confirm findings with a qualified ophthalmologist.",
    }

    patient_one_line = f"The AI screening suggests a finding of {label}. " + \
        {"Severe": "An urgent specialist review is recommended.", "Proliferative": "An urgent specialist review is recommended.",
         "Moderate": "A specialist assessment is recommended soon.", "Mild": "Continue with routine monitoring.",
         "No DR": "No signs of diabetic retinopathy were detected."}.get(label, "A re-take may be needed.")

    clinician_bullets = [
        f"Predicted ICDR grade: {label} (AI confidence: {confidence}%)",
        f"Image quality score: {iq['score']}/100. Flags: {', '.join(iq['flags'])}",
        f"Triage: {triage['level'].upper()} referral to {triage['who']} within {triage['timeframe']}.",
        *explanation,
        "Primary AI interpretation summary provided in the detailed analysis section."
    ]

    return {"json": json_out, "patient": {"one_line": patient_one_line}, "clinician": {"bullets": clinician_bullets}}