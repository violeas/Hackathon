import streamlit as st
from infer import run_inference
from PIL import Image

st.title("Urban Expansion Detector")

before = st.file_uploader("Upload BEFORE image")
after = st.file_uploader("Upload AFTER image")

if before and after:
    before_img = Image.open(before)
    after_img = Image.open(after)

    before_img.save("temp_before.png")
    after_img.save("temp_after.png")

    pred, conf = run_inference("temp_before.png", "temp_after.png")

    st.image(before_img, caption="Before")
    st.image(after_img, caption="After")

    st.write(f"Prediction: {pred}")
    st.write(f"Confidence: {conf:.2f}")
    