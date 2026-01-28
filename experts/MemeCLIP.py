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
from models import LinearClassifier, CosineClassifier, LinearProjection, CLIP_Text, Adapter

class MemeCLIP(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.num_mapping_layers = cfg.num_mapping_layers
        self.unmapped_dim = cfg.unmapped_dim
        self.map_dim = cfg.map_dim
        self.num_pre_output_layers = cfg.num_pre_output_layers
        self.drop_probs = cfg.drop_probs
        self.ratio = cfg.ratio
        self.scale = cfg.scale

        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes = cfg.data.num_classes)
        self.auroc = torchmetrics.AUROC(task='multiclass', num_classes = cfg.data.num_classes)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes = cfg.data.num_classes, average='macro')

        self.clip_model, _ = clip.load(self.cfg.model.clip_model, device="cuda", jit=False)
        self.clip_model.float()

        pre_output_input_dim = self.cfg.map_dim
        pre_output_layers = [nn.Dropout(p=cfg.drop_probs[1])]
        output_input_dim = pre_output_input_dim

        self.classifier = CosineClassifier(feat_dim = output_input_dim, num_classes=cfg.data.num_classes, dtype=self.clip_model.dtype)
        self.init_head_text_feat()
        self.text_encoder =  CLIP_Text(self.clip_model)
        self.img_adapter = Adapter(self.map_dim, 4).to(self.clip_model.dtype)
        self.text_adapter = Adapter(self.map_dim, 4).to(self.clip_model.dtype)
        self.clip_model.visual.proj = None

        for _, p in self.clip_model.named_parameters():
            p.requires_grad_(False)
        
        for name, param in self.classifier.named_parameters():
            param.requires_grad_(True)

        self.image_map = LinearProjection(self.unmapped_dim, self.map_dim,
                                          self.num_mapping_layers, self.drop_probs)
        self.text_map = LinearProjection(768, self.map_dim,
                                         self.num_mapping_layers, self.drop_probs)
        
        self.soft = nn.Softmax(dim=1)
            
        if self.cfg.num_pre_output_layers >= 1:
            pre_output_layers.extend(
                [nn.Linear(pre_output_input_dim, self.cfg.map_dim), nn.ReLU(), nn.Dropout(p=cfg.drop_probs[2])])
            output_input_dim = self.cfg.map_dim

        for _ in range(1, self.cfg.num_pre_output_layers):
            pre_output_layers.extend(
                [nn.Linear(self.cfg.map_dim, self.cfg.map_dim), nn.ReLU(), nn.Dropout(p=cfg.drop_probs[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 1. Estrazione dati (Nota: NON usiamo le labels qui, così funziona anche per test ciechi)
        images, texts = batch['image'], batch['text']

        # 2. ENCODING & PROIEZIONE (Copiato dalla logica di common_step)
        
        # Encoder Visivo
        image_embeds = self.clip_model.visual(images.float())
        
        # Encoder Testuale
        text_embeds = self.clip_model.encode_text(texts)

        # Proiezioni lineari
        image_projection = self.image_map(image_embeds)
        txt_projection = self.text_map(text_embeds)

        # Adattamento
        image_features = self.img_adapter(image_projection)
        text_features = self.text_adapter(txt_projection)

        # Combinazione
        text_features = self.cfg.ratio * text_features + (1 - self.cfg.ratio) * txt_projection
        image_features = self.cfg.ratio * image_features + (1 - self.cfg.ratio) * image_projection

        # Normalizzazione
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Fusione
        features = torch.mul(image_features, text_features)

        # Classificazione
        features_pre_output = self.pre_output(features)
        logits = self.classifier(features_pre_output).squeeze(dim=1)

        # 3. Output: Restituisce le probabilità (Softmax)
        return torch.softmax(logits, dim=1)    
    def init_head_text_feat(self):

        print("Initialize head with text features")
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in self.cfg.class_names]
        prompts = clip.tokenize([p for p in prompts], context_length=77, truncate=True).to(self.cfg.device)
        text_features = self.clip_model.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        text_features = text_features @ self.clip_model.visual.proj.t()
        text_features = F.normalize(text_features, dim=-1)
        self.classifier.apply_weight(text_features)

    def common_step(self, batch):
        # 1. ESTRAZIONE DATI GREZZI DAL BATCH (Chiavi corrette fornite dal DataLoader)
        images, texts, labels = batch['image'], batch['text'], batch['label']

        # 2. ENCODING CLIP (Genera gli embedding iniziali)
    
        # Assicurati che l'encoder di CLIP sia nel dispositivo corretto (già fatto in __init__)
    
        # Encoder Visivo (produce embedding grezzi)
        # Rimuovi l'embedder visuale 'proj' che e' stato settato a None nel costruttore
        image_embeds = self.clip_model.visual(images.float())
    
        # Encoder Testuale (produce embedding grezzi)
        # NOTA: Qui si assume che texts sia già un tensor di IDs (tokenizzato),
        # ma il tuo DataLoader fornisce ancora la stringa grezza se non lo hai corretto.
    
        # Se il DataLoader fornisce stringhe grezze, devi tokenizzare qui (non ideale, ma robusto)
        # tokens = clip.tokenize(texts, truncate=True).to(images.device)
        # text_embeds = self.clip_model.encode_text(tokens)
    
        # Se il DataLoader e' stato corretto e fornisce il tensor di input:
        text_embeds = self.clip_model.encode_text(texts) 
    
        # --- Inizio Logica di Proiezione e Fusione Originale ---
    
        # Proiezioni lineari iniziali
        image_projection = self.image_map(image_embeds)
        txt_projection = self.text_map(text_embeds)

        # Adattamento (Adapter + ResNet-like Connection)
        image_features = self.img_adapter(image_projection)
        text_features = self.text_adapter(txt_projection)

        # Combinazione (Residual connection per gli Adapter)
        text_features = self.cfg.ratio * text_features + (1 - self.cfg.ratio) * txt_projection
        image_features = self.cfg.ratio * image_features + (1 - self.cfg.ratio) * image_projection

        # Normalizzazione finale (Cruciale per i modelli CLIP-based)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
        # Fusione (Element-wise Multiplication)
        features = torch.mul(image_features, text_features) 

        # Classificazione (Passaggi finali)
        features_pre_output = self.pre_output(features)
        logits = self.classifier(features_pre_output).squeeze(dim=1)
    
        # Estrazione delle predizioni per le metriche (NON FARE .data.max(1) o sigmoid qui!)
        # Le metriche preferiscono logits grezzi o probabilità.
        preds = torch.argmax(logits, dim=1) # Predizioni finali per Accuracy/F1
        probs = torch.softmax(logits, dim=1) # Probabilità per AUROC/Loss
    
        # 3. OUTPUT
        output = {}
        output['loss'] = self.cross_entropy_loss(logits, labels) # Usa 'labels' corretto
    
        # Metriche (usano preds per acc/f1, probs/logits per auroc)
        output['accuracy'] = self.acc(preds, labels)
        output['auroc'] = self.auroc(probs, labels) 
        output['f1'] = self.f1(preds, labels)

        return output

    '''
    def common_step(self, batch):

        image_embeds = batch['image_features']
        text_embeds = batch['text_features']

        image_projection = self.image_map(image_embeds)
        txt_projection = self.text_map(text_embeds)

        image_features = self.img_adapter(image_projection)
        text_features = self.text_adapter(txt_projection)

        text_features = self.cfg.ratio  * text_features + (1 - self.cfg.ratio ) * txt_projection
        image_features = self.cfg.ratio  * image_features + (1 - self.cfg.ratio ) * image_projection

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        features = torch.mul(image_features, text_features)

        features_pre_output = self.pre_output(features)
        logits = self.classifier(features_pre_output).squeeze(dim=1) 
        preds_proxy = torch.sigmoid(logits)
        _ , preds = logits.data.max(1)

        output = {}
        output['loss'] = self.cross_entropy_loss(logits, batch['labels'])
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])
        output['f1'] = self.f1(preds, batch['labels'])

        return output
    '''

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
        optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.weight_decay)

        return optimizer

def create_model(cfg):
    model = MemeCLIP(cfg)
    return model
