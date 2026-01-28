import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import clip
from torch.optim import AdamW

# La dimensione dell'embedding per ViT-B/32 è 512
EMBEDDING_DIM = 512 

class CLIP_MLP_Classifier(pl.LightningModule):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Metriche
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.data.num_classes)
        self.auroc = torchmetrics.AUROC(task='multiclass', num_classes=cfg.data.num_classes)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=cfg.data.num_classes, average='macro')
        # Carica il backbone CLIP (usando il percorso corretto)
        self.clip_model, _ = clip.load(self.cfg.model.clip_model, device="cuda", jit=False)
        
        # Congela il CLIP Encoder (non lo addestriamo)
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Dimensione di input dopo la concatenazione: 512 (immagine) + 512 (testo) = 1024
        input_dim = EMBEDDING_DIM * 2 
        output_dim = self.cfg.data.num_classes # 2 per la Task A (Binary)
        
        # MLP per la classificazione finale (fusione)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()

    def common_step(self, batch):
        images, texts, labels = batch['image'], batch['text'], batch['label']

        # 1. Estrazione features immagine (Vision Encoder)
        image_features = self.clip_model.encode_image(images.float())
        
        # 2. Estrazione features testo (Text Encoder)
        text_features = self.clip_model.encode_text(texts)

        # Normalizzazione (come richiesto da CLIP)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 3. Fusione: Concatenazione delle features (superficiale)
        fused_features = torch.cat((image_features, text_features), dim=1)

        # 4. Classificazione
        logits = self.mlp(fused_features.float())
        
        loss = self.loss_fn(logits, labels)
        
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch)
        
        preds = torch.argmax(logits, dim=1)
        
        #self.log('val/loss', loss)
        #self.log('val/auroc', self.auroc(preds, labels))
        self.log('val/auroc', self.auroc(logits, labels)) # <--- CORREZIONE QUI
        self.log('test/f1', self.f1(preds, labels)) # <-- CORREZIONE 1: Usa self.f1
        self.log('val/acc', self.acc(preds, labels))


    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch)
        preds = torch.argmax(logits, dim=1)
        
        self.log('test/acc', self.acc(preds, labels))
        #self.log('test/auroc', self.auroc(preds, labels))
        self.log('test/auroc', self.auroc(logits, labels)) # <--- CORREZIONE QUI        
        self.log('test/f1', self.f1(preds, labels))

    def configure_optimizers(self):
        optimizer = AdamW(self.mlp.parameters(), 
                            lr=self.cfg.training.learning_rate, 
                            weight_decay=self.cfg.training.weight_decay)
        return optimizer

    # Nel file CLIP_MLP.py (all'interno della classe CLIP_MLP_Classifier):

    def forward(self, batch):
        """
        Metodo richiesto da trainer.predict() per passare i dati al modello.
        """
        # Chiama la logica centrale di common_step (ma solo le feature e i logits)
        # Rimuovi l'estrazione di 'loss' se non necessario per la predizione.
    
        images, texts, labels = batch['image'], batch['text'], batch['label']

        # Logica di encoding e concatenazione copiata da common_step
        image_features = self.clip_model.encode_image(images.float())
        text_features = self.clip_model.encode_text(texts)
    
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
        fused_features = torch.cat((image_features, text_features), dim=1)
    
        logits = self.mlp(fused_features.float())
    
        return logits # Deve restituire solo l'output principale (logits o probabilità)
