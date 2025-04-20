import streamlit as st
import subprocess
import tempfile
import os

st.set_page_config(page_title="CVD Risk Estimator", layout="centered")
st.title("🫀 CVD Risk Estimator from Facial Photo")

uploaded_file = st.file_uploader("Upload a patient photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Run CVD Prediction"):
        with st.spinner("Analyzing photo..."):
            result = subprocess.run(
                ["python3", "final_integrated.py", "--test_image", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            st.text("Model Output:")
            st.code(result.stdout)
            if result.stderr:
                st.error(f"Errors:\n{result.stderr}")
