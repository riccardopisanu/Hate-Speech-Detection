import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import clip
from torch.optim import AdamW


class CLIP_MLP_Classifier(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.clip_model, _ = clip.load(
            self.cfg.model.clip_model, device="cpu", jit=False
        )

        self.embed_dim = self.clip_model.visual.output_dim

        input_dim = self.embed_dim * 2
        output_dim = self.cfg.data.num_classes

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=cfg.data.num_classes
        )

        self.f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=cfg.data.num_classes, average="macro"
        )
        self.precision = torchmetrics.Precision(
            task="multiclass", num_classes=cfg.data.num_classes, average="macro"
        )
        self.recall = torchmetrics.Recall(
            task="multiclass", num_classes=cfg.data.num_classes, average="macro"
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def common_step(self, batch):
        images, texts, labels = batch["image"], batch["text"], batch["label"]

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images.float())
            text_features = self.clip_model.encode_text(texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        fused_features = torch.cat((image_features, text_features), dim=1)
        logits = self.mlp(fused_features.float())
        loss = self.loss_fn(logits, labels)

        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch)
        preds = torch.argmax(logits, dim=1)

        self.log("val/loss", loss)
        self.log("val/acc", self.acc(preds, labels))
        self.log("val/f1", self.f1(preds, labels))
        self.log("val/precision", self.precision(preds, labels))
        self.log("val/recall", self.recall(preds, labels))

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch)
        preds = torch.argmax(logits, dim=1)

        self.log("test/acc", self.acc(preds, labels))
        self.log("test/f1", self.f1(preds, labels))
        self.log("test/precision", self.precision(preds, labels))
        self.log("test/recall", self.recall(preds, labels))

    def configure_optimizers(self):
        optimizer = AdamW(
            self.mlp.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        return optimizer

    def forward(self, batch):
        images, texts = batch["image"], batch["text"]
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images.float())
            text_features = self.clip_model.encode_text(texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        fused_features = torch.cat((image_features, text_features), dim=1)

        return self.mlp(fused_features.float())
