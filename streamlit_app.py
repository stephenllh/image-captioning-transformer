import numpy as np
from PIL import Image
import streamlit as st
from src.inference import inference


st.title("Image captioning with Transformer")
# st.info("Use transformer to give a caption to uploaded images.")

uploaded_file = st.file_uploader("Upload an image.", type=["png", "jpg"])

ckpt_path = "models/best.ckpt"  # TODO: upload model to github, then get URL.


if __name__ == "__main__":
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        image = np.array(image)
        predicted_caption_list = inference(image, ckpt_path)

        predicted_caption = ""
        for i, s in enumerate(predicted_caption_list):
            if s != "<eos>":
                predicted_caption += s + " "
            else:
                predicted_caption = predicted_caption[:-1]
                predicted_caption += "."
                predicted_caption_list = predicted_caption_list[:i]
                break
        st.write("Caption: ", predicted_caption.capitalize())
        print(predicted_caption_list)

        st.selectbox("Select option", options=predicted_caption_list)
