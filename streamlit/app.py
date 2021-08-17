import os
import numpy as np
from PIL import Image
import cv2
import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import streamlit as st
from src.learner import ImageCaptioningLearner
from src.get_data import download_and_unzip


MODEL_URL = "https://github.com/stephenllh/image-captioning-transformer/releases/latest/download/model.zip"

if not os.path.exists("./model.ckpt"):
    download_and_unzip(MODEL_URL, extract_to="./")


st.title("Image captioning with Transformer")
uploaded_file = st.file_uploader("Upload an image.", type=["png", "jpg"])


@st.cache
def load_model(ckpt_path):
    learner = ImageCaptioningLearner.load_from_checkpoint(ckpt_path)
    learner.eval()
    return learner


def inference(image, ckpt_path):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tfms = alb.Compose(
        [
            alb.Resize(224, 224),
            alb.Normalize(),
            ToTensorV2(),
        ]
    )
    image = tfms(image=image)["image"]
    image = image.unsqueeze(dim=0)

    learner = load_model(ckpt_path)
    vocab = learner.vocab
    max_length = learner.model_config["max_seq_length"]

    target_indexes = [vocab["<bos>"]] + [vocab["<pad>"]] * (max_length - 1)
    for i in range(max_length - 1):
        caption = torch.LongTensor(target_indexes).unsqueeze(0)
        mask = torch.zeros((1, max_length), dtype=torch.bool)
        mask[:, i + 1 :] = True

        with torch.no_grad():
            pred, _ = learner(image, caption, mask)

        pred_token = pred.argmax(dim=-1)[:, i].item()
        target_indexes[i + 1] = pred_token

        if pred_token == vocab["<eos>"]:
            break

    target_tokens = [vocab.get_itos()[i] for i in target_indexes]
    return target_tokens[1:]


if __name__ == "__main__" and uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    image = np.array(image)
    predicted_caption_list = inference(image, "model.ckpt")

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
