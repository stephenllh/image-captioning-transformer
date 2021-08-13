from torchtext.data.utils import get_tokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset import create_caption_data, Flickr8DataModule
from learner import ImageCaptioningLearner
from utils import load_config

# import argparse
# parser = argparse.ArgumentParser()
# args = parser.parse_args()


def train():
    config = load_config("config.yaml")
    tokenizer = get_tokenizer("basic_english")
    dataframe, vocab = create_caption_data("input/Flickr8k.token.txt", tokenizer)
    data_module = Flickr8DataModule()
    learner = ImageCaptioningLearner(vocab, config)
    callbacks = [
        ModelCheckpoint(**config["callbacks"]["checkpoint"]),
        EarlyStopping(**config["callbacks"]["early_stopping"]),
    ]
    trainer = pl.Trainer(
        gpus=config["trainer"]["gpu"],
        max_epochs=config["trainer"]["epochs"],
        default_root_dir="./",
        callbacks=callbacks,
        precision=(16 if config["trainer"]["fp16"] else 32),
    )
    trainer.fit(learner, data_module)


if __name__ == "__main__":
    train()
