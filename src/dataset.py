from collections import Counter
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchtext
import cv2
import pytorch_lightning as pl
import albumentations as alb
from albumentations.pytorch import ToTensorV2


MAX_SEQ_LENGTH = 20


def create_caption_data(filename, tokenizer):
    """Loads captions (text) data and maps them to corresponding images.
    Then, creates a dataframe for it and a torchtext Vocab

    Args:
        filename
        tokenizer

    Returns:
        dataframe: Dictionary mapping image names and the corresponding captions
        vocab: torchtext Vocab object
    """
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        image2caption_mapping = {}
        text_data = []

    for line in caption_data:
        line = line.rstrip("\n")
        image_filename, caption = line.split("\t")
        image_filename = image_filename.split(".")[0] + ".jpg"

        text_data.append(caption)

        # Remove the filename that does not exist in the folder
        if image_filename == "2258277193_586949ec62.jpg":
            continue

        # Create a dictionary that maps an image filename to a list of captions.
        if image_filename in image2caption_mapping:
            image2caption_mapping[image_filename].append(caption)
        else:
            image2caption_mapping[image_filename] = [caption]

    # Create a dataframe that contains image filename and the respective captions
    dataframe = pd.DataFrame(columns=["image_filename", "captions"])
    dataframe["image_filename"] = image2caption_mapping.keys()
    dataframe["captions"] = dataframe["image_filename"].map(
        lambda x: image2caption_mapping[x]
    )

    # Create torchtext vocab
    counter = Counter()
    for caption in text_data:
        counter.update(tokenizer(caption))

    vocab = torchtext.vocab.vocab(counter)
    for special_token, index in zip(["<unk>", "<pad>", "<bos>", "<eos>"], [0, 1, 2, 3]):
        vocab.insert_token(special_token, index)
    vocab.set_default_index(0)

    return dataframe, vocab


def get_transforms(train_val_test):
    train_tfms = alb.Compose(
        [
            alb.Resize(224, 224),
            alb.HorizontalFlip(p=0.5),
            alb.ColorJitter(brightness=0.2, contrast=0.2),
            alb.Normalize(),
            ToTensorV2(),
        ]
    )
    val_tfms = alb.Compose(
        [
            alb.Resize(224, 224),
            alb.Normalize(),
            ToTensorV2(),
        ]
    )
    return train_tfms if train_val_test == "train" else val_tfms


class Flickr8Dataset:
    def __init__(
        self, data_root_path, dataframe, vocab, tokenizer, max_seq_len, train_val_test
    ):
        self.data_root_path = data_root_path
        self.dataframe = dataframe.copy()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.tfms = get_transforms(train_val_test)

    def __getitem__(self, idx):
        # Image
        image_filename, captions = self.dataframe.iloc[idx.item()]
        image = cv2.imread(f"{self.data_root_path}/Flicker8k_Dataset/{image_filename}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.tfms(image=image)["image"]

        # Captions
        tokens_arrays = []
        attention_masks = []
        pad_idx = self.vocab["<pad>"]
        bos_idx = self.vocab["<bos>"]
        eos_idx = self.vocab["<eos>"]
        pad_starts = None

        for i, caption in enumerate(captions):
            tokens = [self.vocab[token] for token in self.tokenizer(caption)]
            tokens = [bos_idx] + tokens + [eos_idx]

            # Pad tokens and ensure the length equals self.max_seq_len
            if len(tokens) <= self.max_seq_len:
                pad_starts = len(tokens)
                tokens += [pad_idx] * (self.max_seq_len - len(tokens))
            else:
                tokens = tokens[: self.max_seq_len - 1] + [eos_idx]

            assert len(tokens) == self.max_seq_len  # TODO: move to test
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens_arrays.append(tokens)

            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            if pad_starts is not None:
                mask[pad_starts:] = True
            attention_masks.append(mask)

        out_dict = {
            "image": image,
            "captions_tokens": tokens_arrays,
            "captions": captions,
            "attention_masks": attention_masks,
        }
        return out_dict

    def __len__(self):
        return len(self.dataframe)


class Flickr8DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dm_config,
        data_root,
        dataframe,
        vocab,
        tokenizer,
        max_seq_len,
        val_pct=0.1,
    ):
        super().__init__()
        self.dm_config = dm_config
        self.data_root = data_root
        self.dataframe = dataframe
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.val_pct = val_pct

    def setup(self, stage=None):
        self.train_dataset = Flickr8Dataset(
            self.data_root,
            self.dataframe,
            self.vocab,
            self.tokenizer,
            self.max_seq_len,
            mode="train",
        )
        self.val_dataset = Flickr8Dataset(
            self.data_root,
            self.dataframe,
            self.vocab,
            self.tokenizer,
            self.max_seq_len,
            mode="val",
        )
        dataset_size = len(self.train_dataset)
        indices = torch.randperm(dataset_size)
        split = int(self.val_pct * dataset_size)
        self.train_idx, self.val_idx = indices[split:], indices[:split]

    def train_dataloader(self):
        train_sampler = SubsetRandomSampler(self.train_idx)
        return DataLoader(
            self.train_dataset,
            batch_size=self.dm_config["batch_size"],
            num_workers=self.dm_config["num_workers"],
            drop_last=True,
            sampler=train_sampler,
        )

    def val_dataloader(self):
        val_sampler = SubsetRandomSampler(self.val_idx)
        return DataLoader(
            self.val_dataset,
            batch_size=self.dm_config["batch_size"],
            num_workers=self.dm_config["num_workers"],
            drop_last=True,
            sampler=val_sampler,
        )
