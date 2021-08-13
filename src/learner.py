import torch
from torch import nn
import pytorch_lightning as pl
from .model import ImageCaptioningModel


class ImageCaptioningLearner(pl.LightningModule):
    def __init__(self, vocab, model_config, learner_config):
        super().__init__()
        self.vocab = vocab
        self.model_config = model_config
        self.learner_config = learner_config
        self.model = ImageCaptioningModel(vocab_size=len(vocab), **model_config)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
        self.save_hyperparameters()

    def forward(self, image, caption, mask):
        pred, attention = self.model(image, caption, mask)
        return pred, attention

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learner_config["lr"])
        return {"optimizer": optimizer, "monitor": "val_loss"}

    def step(self, batch, mode="train"):
        image = batch["image"]
        captions = batch[
            "captions_tokens"
        ]  # List of tensors of shape (batch_size, sentence_length)
        attention_masks = batch[
            "attention_masks"
        ]  # List of tensors of shape (batch_size, sentence_length)

        loss = 0.0
        preds = []  # for logging purposes
        for caption, mask in zip(captions, attention_masks):
            pred, _ = self.model(image, caption[:, :-1], mask[:, :-1])
            pred = pred.view(-1, len(self.vocab))
            preds.append(pred)
            caption = caption[:, 1:].contiguous().view(-1)
            loss += self.criterion(pred, caption)
        loss /= len(captions)
        self.log(f"{mode}_loss", loss, prog_bar=True)

        return loss if mode == "train" else preds[0]

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="val")
