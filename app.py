import streamlit as st
import cv2
import numpy as np
import uuid
import plotly.graph_objects as go
import urllib.parse

from model import load_model
from utils import generate_report

import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Nethra Vision AI",
    page_icon="👁️",
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def get_model():

    return load_model()


model = get_model()


# =========================================================
# SEVERITY COLORS
# =========================================================

severity_colors = {

    "No Diabetic Retinopathy":
        "#00C853",

    "Mild Diabetic Retinopathy":
        "#FFD600",

    "Moderate Diabetic Retinopathy":
        "#FF9100",

    "Severe Diabetic Retinopathy":
        "#FF3D00",

    "Proliferative Diabetic Retinopathy":
        "#D50000"
}


# =========================================================
# HEADER
# =========================================================

st.title(
    "👁️ Nethra Vision AI"
)

st.caption(
    "AI-assisted Diabetic Retinopathy Screening"
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.subheader(
    "Patient Information"
)


patient_id = st.text_input(
    "Patient ID",
    value=str(
        uuid.uuid4()
    )[:8]
)


patient_name = st.text_input(
    "Patient Name"
)


# =========================================================
# IMAGE INPUT
# =========================================================

st.subheader(
    "Retinal Image"
)


col1, col2 = st.columns(2)


with col1:

    uploaded = st.file_uploader(
        "Upload Image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )


with col2:

    captured = st.camera_input(
        "Capture Image"
    )


image_input = (
    uploaded
    if uploaded is not None
    else captured
)


# =========================================================
# IMAGE PROCESSING
# =========================================================

if image_input:

    file_bytes = np.asarray(
        bytearray(
            image_input.getvalue()
        ),
        dtype=np.uint8
    )


    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    if image is None:

        st.error(
            "Could not read the image."
        )

        st.stop()


    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    st.image(
        image,
        caption="Input Fundus Image",
        use_container_width=True
    )


    # =====================================================
    # GENERATE REPORT
    # =====================================================

    if st.button(
        "🔬 Generate Report",
        use_container_width=True
    ):

        output_dir = os.path.join(
            "reports",
            patient_id
        )

        with st.spinner(
            "Analyzing retinal image..."
        ):

            output = generate_report(

                model,

                image,

                output_dir,

                patient_id,

                patient_name
            )


        st.success(
            "Analysis complete."
        )


        severity = output[
            "severity"
        ]

        score = output[
            "score"
        ]

        confidence = output[
            "confidence"
        ]

        action = output[
            "action"
        ]

        pdf_path = output[
            "pdf_path"
        ]

        comparison = output[
            "comparison_path"
        ]


        # =================================================
        # SEVERITY
        # =================================================

        color = severity_colors[
            severity
        ]


        st.markdown(

            f"""
            <div style="
                background:{color};
                padding:16px;
                border-radius:12px;
                color:white;
                text-align:center;
                font-size:22px;
                font-weight:bold;
                margin:15px 0;">
                {severity}
            </div>
            """,

            unsafe_allow_html=True
        )


        # =================================================
        # METRICS
        # =================================================

        m1, m2 = st.columns(2)


        with m1:

            st.metric(
                "Severity Score",
                f"{score} / 4"
            )


        with m2:

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )


        # =================================================
        # CONFIDENCE GAUGE
        # =================================================

        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=confidence * 100,

                title={
                    "text":
                    "Model Confidence"
                },

                gauge={

                    "axis": {
                        "range":
                        [0, 100]
                    },

                    "bar": {
                        "color":
                        color
                    },

                    "steps": [

                        {
                            "range":
                            [0, 50],
                            "color":
                            "#eeeeee"
                        },

                        {
                            "range":
                            [50, 75],
                            "color":
                            "#dddddd"
                        },

                        {
                            "range":
                            [75, 100],
                            "color":
                            "#cccccc"
                        }
                    ]
                }
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # =================================================
        # VISUAL ANALYSIS
        # =================================================

        st.subheader(
            "AI Visual Analysis"
        )


        st.image(
            comparison,
            use_container_width=True
        )


        # =================================================
        # ACTION
        # =================================================

        st.info(
            f"Suggested action: {action}"
        )


        # =================================================
        # DOWNLOAD PDF
        # =================================================

        st.subheader(
            "Report"
        )


        with open(
            pdf_path,
            "rb"
        ) as f:

            st.download_button(

                "📄 Download Medical Report",

                f,

                file_name=
                f"{patient_id}_DR_Report.pdf",

                mime="application/pdf",

                use_container_width=True
            )


        # =================================================
        # WHATSAPP
        # =================================================

        message = (

            "Nethra Vision AI Screening Report\n\n"

            f"Patient ID: {patient_id}\n"

            f"Patient Name: "
            f"{patient_name or 'Not provided'}\n\n"

            f"Severity: {severity}\n"

            f"Severity Score: {score}/4\n"

            f"Confidence: {confidence:.2%}\n\n"

            "Report generated by "
            "Nethra Vision AI."
        )


        whatsapp_url = (
            "https://wa.me/?text="
            + urllib.parse.quote(message)
        )


        st.markdown(
            f"""
            <a href="{whatsapp_url}"
               target="_blank">
                📲 Share via WhatsApp
            </a>
            """,
            unsafe_allow_html=True
        )