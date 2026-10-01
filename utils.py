import os
from datetime import datetime

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as ReportLabImage,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from pytorch_grad_cam import EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# DIABETIC RETINOPATHY CLASS INFORMATION
# ============================================================

class_info = {
    0: (
        "No Diabetic Retinopathy",
        0,
        "Annual screening recommended."
    ),

    1: (
        "Mild Diabetic Retinopathy",
        1,
        "Monitor condition and consider ophthalmologist evaluation."
    ),

    2: (
        "Moderate Diabetic Retinopathy",
        2,
        "Ophthalmologist referral recommended."
    ),

    3: (
        "Severe Diabetic Retinopathy",
        3,
        "Urgent ophthalmic evaluation recommended."
    ),

    4: (
        "Proliferative Diabetic Retinopathy",
        4,
        "Prompt ophthalmic evaluation is recommended."
    )
}


# ============================================================
# SEVERITY COLORS
# ============================================================

severity_colors = {
    "No Diabetic Retinopathy": colors.green,
    "Mild Diabetic Retinopathy": colors.yellow,
    "Moderate Diabetic Retinopathy": colors.orange,
    "Severe Diabetic Retinopathy": colors.red,
    "Proliferative Diabetic Retinopathy": colors.darkred
}


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(img):
    """
    Preprocess RGB retinal image for EfficientNet-B4.

    Steps:
    1. Resize to 512x512
    2. Graham-style illumination/contrast enhancement
    3. Normalize to [0,1]
    """

    img = cv2.resize(
        img,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    # Graham-style preprocessing
    blurred = cv2.GaussianBlur(
        img,
        (0, 0),
        512 / 30
    )

    img = cv2.addWeighted(
        img,
        4,
        blurred,
        -4,
        128
    )

    img = img / 255.0

    return img.astype(np.float32)


# ============================================================
# RETINA MASK
# ============================================================

def get_retina_mask(image):
    """
    Creates a simple mask to reduce CAM activation
    outside the visible retinal region.

    This is NOT a retinal/lesion segmentation model.
    """

    image_uint8 = np.clip(
        image * 255,
        0,
        255
    ).astype(np.uint8)

    gray = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        10,
        1,
        cv2.THRESH_BINARY
    )

    mask = cv2.medianBlur(
        mask.astype(np.uint8),
        9
    )

    return mask


# ============================================================
# CAM NORMALIZATION
# ============================================================

def normalize_cam(cam):
    """
    Normalize CAM values to [0,1].
    """

    cam = cam.astype(np.float32)

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam)

    cam = (
        cam - cam_min
    ) / (
        cam_max - cam_min
    )

    return cam


# ============================================================
# LESION HIGHLIGHT GENERATION
# ============================================================

def create_lesion_highlight(
    original,
    grayscale_cam
):
    """
    Creates a pseudo lesion-highlight visualization
    from CAM activation.

    IMPORTANT:
    This is an attention-based visualization and NOT
    true lesion segmentation.
    """

    # Create retina mask
    retina_mask = get_retina_mask(
        original.astype(np.float32) / 255.0
    )

    # Adaptive CAM threshold
    threshold = np.percentile(
        grayscale_cam,
        85
    )

    cam_mask = (
        grayscale_cam > threshold
    ).astype(np.uint8)

    # Restrict activation to retina region
    mask = cam_mask * retina_mask

    # Morphological cleanup
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_DILATE,
        kernel
    )

    # Remove very small regions
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_mask = np.zeros_like(mask)

    for contour in contours:

        if cv2.contourArea(contour) > 200:

            cv2.drawContours(
                clean_mask,
                [contour],
                -1,
                1,
                -1
            )

    # Create highlighted image
    lesion_highlight = original.copy()

    # Red highlight
    lesion_highlight[
        clean_mask > 0
    ] = [255, 0, 0]

    lesion_highlight = cv2.addWeighted(
        original,
        0.7,
        lesion_highlight,
        0.4,
        0
    )

    return lesion_highlight


# ============================================================
# GENERATE PDF REPORT
# ============================================================

def create_pdf_report(
    pdf_path,
    comparison_path,
    patient_id,
    patient_name,
    severity,
    score,
    confidence,
    action
):
    """
    Creates the final Nethra Vision AI PDF report.
    """

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=(
            8.5 * inch,
            11 * inch
        ),
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    elements = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title_style = styles["Title"]

    elements.append(
        Paragraph(
            "<b>Nethra Vision AI</b>",
            title_style
        )
    )

    elements.append(
        Paragraph(
            "Diabetic Retinopathy Screening Report",
            styles["Heading2"]
        )
    )

    elements.append(
        Spacer(1, 10)
    )

    # --------------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------------

    patient_data = [
        ["Patient ID", patient_id],
        [
            "Patient Name",
            patient_name if patient_name else "Not provided"
        ],
        [
            "Date",
            datetime.now().strftime(
                "%Y-%m-%d %H:%M"
            )
        ]
    ]

    patient_table = Table(
        patient_data,
        colWidths=[
            2 * inch,
            4.5 * inch
        ]
    )

    patient_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.black
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    elements.append(
        patient_table
    )

    elements.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    severity_color = severity_colors.get(
        severity,
        colors.grey
    )

    severity_table = Table(
        [
            [
                f"SEVERITY: {severity.upper()}"
            ]
        ],
        colWidths=[
            6.5 * inch
        ]
    )

    severity_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                severity_color
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.black
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica-Bold"
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                14
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                10
            )
        ])
    )

    elements.append(
        severity_table
    )

    elements.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # ANALYSIS TABLE
    # --------------------------------------------------------

    analysis_data = [
        [
            "Parameter",
            "Result"
        ],
        [
            "Severity Level",
            severity
        ],
        [
            "Severity Score",
            f"{score} / 4"
        ],
        [
            "Model Confidence",
            f"{confidence:.2%}"
        ],
        [
            "Suggested Action",
            action
        ]
    ]

    analysis_table = Table(
        analysis_data,
        colWidths=[
            2.2 * inch,
            4.3 * inch
        ]
    )

    analysis_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    elements.append(
        analysis_table
    )

    elements.append(
        Spacer(1, 20)
    )

    # --------------------------------------------------------
    # VISUAL ANALYSIS
    # --------------------------------------------------------

    elements.append(
        Paragraph(
            "<b>Visual Analysis</b>",
            styles["Heading2"]
        )
    )

    elements.append(
        Spacer(1, 8)
    )

    # Keep the image within the page width
    elements.append(
        ReportLabImage(
            comparison_path,
            width=6.5 * inch,
            height=1.7 * inch
        )
    )

    elements.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    disclaimer = (
        "<b>Important Note:</b> "
        "This report is generated by an AI-based screening "
        "system and is intended for screening, research, "
        "visualization, and decision-support purposes. "
        "It is not a substitute for examination or diagnosis "
        "by a qualified eye-care professional."
    )

    elements.append(
        Paragraph(
            disclaimer,
            styles["Normal"]
        )
    )

    elements.append(
        Spacer(1, 15)
    )

    elements.append(
        Paragraph(
            "<i>Generated by Nethra Vision AI</i>",
            styles["Normal"]
        )
    )

    # Build PDF
    doc.build(elements)


# ============================================================
# MAIN REPORT GENERATION FUNCTION
# ============================================================

def generate_report(
    model,
    image,
    output_dir,
    patient_id="Unknown",
    patient_name=""
):
    """
    Complete Nethra Vision AI pipeline:

    Image
       ↓
    Preprocessing
       ↓
    EfficientNet-B4
       ↓
    Prediction + Confidence
       ↓
    EigenCAM + LayerCAM
       ↓
    CAM visualization
       ↓
    Attention-based lesion highlight
       ↓
    Comparison image
       ↓
    PDF report
    """

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = next(
        model.parameters()
    ).device

    model.eval()

    # --------------------------------------------------------
    # Prepare original image
    # --------------------------------------------------------

    original = cv2.resize(
        image,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    preprocessed = preprocess_image(
        original
    )

    # --------------------------------------------------------
    # Convert image to PyTorch tensor
    # --------------------------------------------------------

    input_tensor = torch.tensor(
        preprocessed,
        dtype=torch.float32
    ).permute(
        2,
        0,
        1
    ).unsqueeze(
        0
    ).to(device)

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

    # Predicted class
    pred = torch.argmax(
        probs,
        dim=1
    ).item()

    # Confidence
    confidence = probs[
        0,
        pred
    ].item()

    # Class information
    severity, score, action = class_info[
        pred
    ]

    # --------------------------------------------------------
    # CLASS-SPECIFIC CAM TARGET
    # --------------------------------------------------------

    targets = [
        ClassifierOutputTarget(pred)
    ]

    # --------------------------------------------------------
    # TARGET LAYER
    # --------------------------------------------------------

    # EfficientNet-B4 late convolutional block
    target_layer = model._blocks[-2]

    # --------------------------------------------------------
    # EIGENCAM
    # --------------------------------------------------------

    with EigenCAM(
        model=model,
        target_layers=[target_layer]
    ) as cam1:

        cam_eigen = cam1(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    # --------------------------------------------------------
    # LAYERCAM
    # --------------------------------------------------------

    with LayerCAM(
        model=model,
        target_layers=[target_layer]
    ) as cam2:

        cam_layer = cam2(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    # --------------------------------------------------------
    # COMBINE CAMs
    # --------------------------------------------------------

    cam_eigen = normalize_cam(
        cam_eigen
    )

    cam_layer = normalize_cam(
        cam_layer
    )

    grayscale_cam = (
        cam_eigen + cam_layer
    ) / 2.0

    grayscale_cam = normalize_cam(
        grayscale_cam
    )

    # --------------------------------------------------------
    # CREATE GRAD-CAM VISUALIZATION
    # --------------------------------------------------------

    original_float = (
        original.astype(
            np.float32
        ) / 255.0
    )

    gradcam = show_cam_on_image(
        original_float,
        grayscale_cam,
        use_rgb=True
    )

    # --------------------------------------------------------
    # CREATE ATTENTION-BASED LESION HIGHLIGHT
    # --------------------------------------------------------

    lesion_highlight = create_lesion_highlight(
        original,
        grayscale_cam
    )

    # --------------------------------------------------------
    # COMPARISON IMAGE
    # --------------------------------------------------------

    comparison_path = os.path.join(
        output_dir,
        "comparison.png"
    )

    plt.figure(
        figsize=(16, 4)
    )

    # Original
    plt.subplot(
        1,
        4,
        1
    )

    plt.imshow(
        original
    )

    plt.title(
        "Original"
    )

    plt.axis(
        "off"
    )

    # Preprocessed
    plt.subplot(
        1,
        4,
        2
    )

    plt.imshow(
        preprocessed
    )

    plt.title(
        "Preprocessed"
    )

    plt.axis(
        "off"
    )

    # CAM
    plt.subplot(
        1,
        4,
        3
    )

    plt.imshow(
        gradcam
    )

    plt.title(
        "AI Attention Map"
    )

    plt.axis(
        "off"
    )

    # Highlight
    plt.subplot(
        1,
        4,
        4
    )

    plt.imshow(
        lesion_highlight
    )

    plt.title(
        "Attention Highlight"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        comparison_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # CREATE PDF
    # --------------------------------------------------------

    pdf_path = os.path.join(
        output_dir,
        "DR_Report.pdf"
    )

    create_pdf_report(
        pdf_path=pdf_path,
        comparison_path=comparison_path,
        patient_id=patient_id,
        patient_name=patient_name,
        severity=severity,
        score=score,
        confidence=confidence,
        action=action
    )

    # --------------------------------------------------------
    # RETURN RESULTS TO STREAMLIT
    # --------------------------------------------------------

    return {
        "severity": severity,
        "score": score,
        "confidence": confidence,
        "action": action,

        "prediction_class": pred,

        "probabilities": probs[
            0
        ].detach().cpu().numpy(),

        "original": original,

        "preprocessed": preprocessed,

        "gradcam": gradcam,

        "lesion_highlight": lesion_highlight,

        "comparison_path": comparison_path,

        "pdf_path": pdf_path
    }