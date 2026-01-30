import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from clip import clip
from tqdm import tqdm
import os
from functools import partial
import torch.nn.functional as F
from transformers import AutoTokenizer
torch.set_default_dtype(torch.float32)

# Import locale (assumendo models.py nella stessa cartella)
from models import LinearClassifier, CosineClassifier, LinearProjection, CLIP_Text, Adapter

class MemeCLIP(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # --- FIX RPISANU: Espone map_dim per evitare AttributeError ---
        self.map_dim = cfg.map_dim
        # -------------------------------------------------------------

        # Gestione Alias per evitare errori se cfg.num_classes non è alla radice
        try:
            num_classes = cfg.num_classes
        except AttributeError:
            num_classes = cfg.data.num_classes

        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.auroc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')

        self.clip_model, _ = clip.load(self.cfg.clip_variant, device="cuda", jit=False)
        self.clip_model.float()

        pre_output_input_dim = self.cfg.map_dim
        
        # Gestione sicura drop_probs
        drop_prob_1 = cfg.drop_probs[1] if hasattr(cfg, 'drop_probs') else cfg.model.drop_probs[1]
        drop_prob_2 = cfg.drop_probs[2] if hasattr(cfg, 'drop_probs') else cfg.model.drop_probs[2]

        pre_output_layers = [nn.Dropout(p=drop_prob_1)]
        output_input_dim = pre_output_input_dim

        self.classifier = CosineClassifier(feat_dim=output_input_dim, num_classes=num_classes, dtype=self.clip_model.dtype)
        self.init_head_text_feat()
        self.text_encoder = CLIP_Text(self.clip_model)
        self.img_adapter = Adapter(self.map_dim, 4).to(self.clip_model.dtype)
        self.text_adapter = Adapter(self.map_dim, 4).to(self.clip_model.dtype)
        
        # CLIP projection removal (Output becomes 1024 for ViT-L/14)
        self.clip_model.visual.proj = None

        for _, p in self.clip_model.named_parameters():
            p.requires_grad_(False)

        for name, param in self.classifier.named_parameters():
            param.requires_grad_(True)
        
        # Gestione sicura unmapped_dim e num_mapping_layers
        unmapped_dim = cfg.unmapped_dim if hasattr(cfg, 'unmapped_dim') else cfg.model.unmapped_dim
        num_mapping_layers = cfg.num_mapping_layers if hasattr(cfg, 'num_mapping_layers') else cfg.model.num_mapping_layers

        # --- FIX DIMENSIONE ASIMMETRICA (ViT-L/14) ---
        # Image Map usa la dimensione definita nel config (1024)
        self.image_map = LinearProjection(unmapped_dim, self.map_dim, num_mapping_layers, cfg.drop_probs)

        # Text Map DEVE usare la dimensione reale del testo di CLIP (768), non quella dell'immagine!
        # Leggiamo la dimensione direttamente dai pesi del modello per essere sicuri.
        if hasattr(self.clip_model, 'text_projection'):
            real_text_dim = self.clip_model.text_projection.shape[0]
        else:
            real_text_dim = 768 # Fallback sicuro per ViT-L/14
            
        print(f"[MemeCLIP DEBUG] Image Dim: {unmapped_dim}, Text Dim: {real_text_dim}")
        self.text_map = LinearProjection(real_text_dim, self.map_dim, num_mapping_layers, cfg.drop_probs)
        # ---------------------------------------------

        self.soft = nn.Softmax(dim=1)
        
        # Gestione num_pre_output_layers
        num_pre = cfg.num_pre_output_layers if hasattr(cfg, 'num_pre_output_layers') else cfg.model.num_pre_output_layers

        if num_pre >= 1:
            pre_output_layers.extend(
                [nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=drop_prob_2)])
            output_input_dim = self.map_dim

        for _ in range(1, num_pre):
            pre_output_layers.extend(
                [nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=drop_prob_2)])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch):
        pass

    def init_head_text_feat(self):
        print("Initialize head with text features")
        template = "a photo of a {}."
        
        # Gestione sicura class_names
        c_names = self.cfg.class_names if hasattr(self.cfg, 'class_names') else self.cfg.model.class_names
        
        prompts = [template.format(c.replace("_", " ")) for c in c_names]
        
        # Gestione Device
        device = self.cfg.device if hasattr(self.cfg, 'device') else "cuda"
        
        prompts = clip.tokenize([p for p in prompts], context_length=77, truncate=True).to(device)
        text_features = self.clip_model.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        
        # Usiamo la proiezione solo se esiste (potrebbe essere None se l'abbiamo rimossa, ma qui serve)
        if self.clip_model.visual.proj is not None:
             text_features = text_features @ self.clip_model.visual.proj.t()
             
        text_features = F.normalize(text_features, dim=-1)
        self.classifier.apply_weight(text_features)

    # Questo metodo verrà sovrascritto dalla patch nel main.py, 
    # ma lo lasciamo qui per completezza sintattica.
    def common_step(self, batch):
        return {} 

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch)
        total_loss = output['loss']
        self.log('train/total_loss', total_loss)
        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log(f'train/auroc', output['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch)
        total_loss = output['loss']
        self.log(f'val/total_loss', total_loss)
        self.log(f'val/loss', output['loss'])
        self.log(f'val/accuracy', output['accuracy'], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'val/auroc', output['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'val/f1', output['f1'], on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        output = self.common_step(batch)
        self.log(f'test/accuracy', output['accuracy'])
        self.log(f'test/auroc', output['auroc'])
        self.log(f'test/f1', output['f1'])
        return output

    def on_train_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()

    def on_validation_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()

    def on_test_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
        ]
        
        # Gestione sicura learning rate e weight decay
        lr = self.cfg.lr if hasattr(self.cfg, 'lr') else self.cfg.training.learning_rate
        wd = self.cfg.weight_decay if hasattr(self.cfg, 'weight_decay') else self.cfg.training.weight_decay
        
        optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=wd)
        return optimizer

def create_model(cfg):
    model = MemeCLIP(cfg)
    return model
