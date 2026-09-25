import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢")

st.title("🔢 MNIST Digit Classifier")
st.write("Draw a digit (0-9) below or upload an image, and the model will predict it.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")


model = load_model()


def preprocess(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a (1, 28, 28, 1) normalized array the model expects."""
    img = img.convert("L")              # grayscale
    img = ImageOps.invert(img)          # white digit on black bg, like MNIST
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


input_mode = st.radio("Input method:", ["Draw", "Upload image"], horizontal=True)

image_to_predict = None

if input_mode == "Draw":
    try:
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=15,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result.image_data is not None:
            image_to_predict = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            ).convert("L")
            # already white-on-black, so undo the invert step later
            image_to_predict = ImageOps.invert(image_to_predict)
    except ImportError:
        st.warning(
            "streamlit-drawable-canvas isn't installed. "
            "Add it to requirements.txt, or use 'Upload image' instead."
        )

else:
    uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Uploaded image", width=150)

if image_to_predict is not None and st.button("Predict"):
    processed = preprocess(image_to_predict)
    preds = model.predict(processed)[0]
    predicted_digit = int(np.argmax(preds))

    st.subheader(f"Prediction: {predicted_digit}")
    st.bar_chart({"probability": preds})
