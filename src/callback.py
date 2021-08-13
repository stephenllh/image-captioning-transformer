import torch
import pytorch_lightning as pl
import wandb


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=4):
        super().__init__()
        self.val_samples = val_samples
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):

        self.val_samples["image"] = self.val_samples["image"].to(
            device=pl_module.device
        )[: self.num_samples]
        self.val_samples["captions_tokens"] = [
            tokens[: self.num_samples].to(device=pl_module.device)
            for tokens in self.val_samples["captions_tokens"]
        ]
        self.val_samples["attention_masks"] = [
            mask[: self.num_samples].to(device=pl_module.device)
            for mask in self.val_samples["attention_masks"]
        ]

        logits = pl_module.step(self.val_samples, "val")
        preds = torch.argmax(logits, dim=-1)
        preds = preds.view(self.num_samples, -1)

        pred_captions = []
        for i in range(self.num_samples):
            s = ""
            for pred in preds[i]:
                if pl_module.vocab.get_itos()[pred] != "<eos>":
                    s += pl_module.vocab.get_itos()[pred] + " "
                else:
                    break
            pred_captions.append(s[:-1])

        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Prediction: {pred_caption}\nLabel: {y}")
                    for x, pred_caption, y in zip(
                        self.val_samples["image"],
                        pred_captions,
                        self.val_samples["captions"][: self.num_samples][0],
                    )
                ],
                "global_step": trainer.global_step,
            }
        )
