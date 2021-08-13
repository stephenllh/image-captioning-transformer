import cv2
import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from .learner import ImageCaptioningLearner


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

    learner = ImageCaptioningLearner.load_from_checkpoint(ckpt_path)
    learner.eval()
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


if __name__ == "__main__":
    image_path = "download (2).jpg"
    ckpt_path = "ckpt/last.ckpt"
    image = cv2.imread(image_path)
    predicted_caption = inference(image_path, ckpt_path)

    predicted_caption_ = ""
    for s in predicted_caption:
        predicted_caption_ += s + " "
        if s == "<eos>":
            predicted_caption_ = predicted_caption_[:-9]
            predicted_caption_ += "."
            break
    print(predicted_caption_.capitalize())
