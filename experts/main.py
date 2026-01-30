import argparse
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from configs import get_cfg_defaults

# Import necessari per il fix di sicurezza e CLIP
from yacs.config import CfgNode
from clip import clip

# Importa i moduli (Gestione dei percorsi locali)
from my_datasets import load_dataset
from CLIP_MLP import CLIP_MLP_Classifier

# Import dinamico per MemeCLIP (Assumiamo che il file sia nella stessa cartella)
try:
    from MemeCLIP import MemeCLIP
except ImportError:
    MemeCLIP = None # Gestiremo l'errore dopo se serve

# =============================================================================
# PATCH: Funzione che insegna a MemeCLIP a leggere immagini RAW
# =============================================================================
def patched_common_step(self, batch):
    """
    Sostituisce il metodo common_step originale di MemeCLIP.
    Adattato per il Dataset RPisanu (chiavi: 'image', 'text').
    """
    # 1. Otteniamo i dati dal batch (Nomi corretti dal dataset)
    images = batch['image']      # [Batch, 3, 224, 224]
    text_tokens = batch['text']  # [Batch, 77] (Già tokenizzato dal dataset!)
    labels = batch['label']

    # 2. Encoding ON-THE-FLY con CLIP (congelato)
    with torch.no_grad():
        # Encode Immagini
        image_embeds = self.clip_model.encode_image(images)
        image_embeds = image_embeds.float() # Casting a float32

        # Encode Testo
        # NOTA: Il dataset ci da già i token, li passiamo direttamente
        text_embeds = self.clip_model.encode_text(text_tokens)
        text_embeds = text_embeds.float()

    # 3. Normalizzazione (Fondamentale per CLIP)
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # --- DA QUI IN POI È IL CODICE ORIGINALE DI MEMECLIP ---
    image_projection = self.image_map(image_embeds)
    txt_projection = self.text_map(text_embeds)

    image_features = self.img_adapter(image_projection)
    text_features = self.text_adapter(txt_projection)

    # Recupera il ratio dal config (gestione flessibile)
    try:
        ratio = self.cfg.model.ratio
    except AttributeError:
        ratio = self.cfg.ratio

    text_features = ratio * text_features + (1 - ratio) * txt_projection
    image_features = ratio * image_features + (1 - ratio) * image_projection

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    features = torch.mul(image_features, text_features)

    features_pre_output = self.pre_output(features)
    logits = self.classifier(features_pre_output).squeeze(dim=1)

    # Gestione Label (Long vs Float)
    if labels.dtype == torch.float:
            labels = labels.long()

    preds_proxy = torch.sigmoid(logits)
    _ , preds = logits.data.max(1)

    output = {}
    output['loss'] = self.cross_entropy_loss(logits, labels)
    output['accuracy'] = self.acc(preds, labels)
    output['auroc'] = self.auroc(preds_proxy, labels)
    output['f1'] = self.f1(preds, labels)

    return output

# =============================================================================
# MAIN
# =============================================================================
def main():
    # 1. Gestione Argomenti
    parser = argparse.ArgumentParser(description="Meme Detection Training")
    parser.add_argument("--config", type=str, default="", help="Path al file YAML di configurazione")
    parser.add_argument("--dataset", type=str, default="", help="Nome del dataset (opzionale, override)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Override opzioni YACS da riga di comando")
    args = parser.parse_args()

    # 2. Caricamento Configurazione
    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)

    # Override manuale dataset se passato da riga di comando
    if args.dataset:
        cfg.data.dataset_name = args.dataset

    cfg.freeze()
    print(f"Configurazione caricata per: {cfg.data.dataset_name}")

    # 3. Setup Seed per riproducibilità
    pl.seed_everything(42)

    # 4. Caricamento Dati
    try:
        train_dataset = load_dataset(cfg, split='train')
        val_dataset = load_dataset(cfg, split='dev')
        test_dataset = load_dataset(cfg, split='test')

        # Dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            drop_last=True 
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers
        )
    except Exception as e:
        print(f"[MAIN ERROR] Errore nel caricamento dati: {e}")
        return

    # 5. Inizializzazione Modello (SCELTA DINAMICA)
    print("Inizializzazione modello...")
    
    ModelClass = None # Placeholder per la classe da usare nel caricamento finale

    if cfg.model.name == 'MemeCLIP':
        if MemeCLIP is None:
            raise ImportError("Impossibile trovare MemeCLIP.py. Assicurati che sia nella cartella experts/")
        
        print(f"[MODEL INFO] Loading MemeCLIP Architecture")
        model = MemeCLIP(cfg)
        
        # --- APPLICAZIONE PATCH ---
        # Sovrascriviamo il metodo della classe istanziata con il nostro metodo patchato
        model.common_step = patched_common_step.__get__(model, MemeCLIP)
        print("[MODEL INFO] Patch 'Raw Image' applicata a MemeCLIP.")
        
        ModelClass = MemeCLIP

    else:
        # Default: CLIP_MLP
        print(f"[MODEL INFO] Loading CLIP_MLP Architecture")
        model = CLIP_MLP_Classifier(cfg)
        ModelClass = CLIP_MLP_Classifier

    # 6. Setup Trainer
    dir_name = f"{cfg.data.dataset_name}_{cfg.model.name}"
    os.makedirs(os.path.join(cfg.training.checkpoint_dir, dir_name), exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.training.checkpoint_dir, dir_name),
        filename="epoch={epoch}-step={step}",
        monitor="val/f1", 
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10 
    )

    # 7. Training (Se max_epochs > 0)
    if cfg.training.max_epochs > 0:
        print("Avvio Training...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Percorso del miglior modello
        best_model_path = checkpoint_callback.best_model_path
    else:
        print("[INFO] Max epochs = 0. Salto il training e testo il modello inizializzato (Zero-Shot / Random).")
        best_model_path = None # Usiamo il modello corrente in memoria

    # 8. Test Finale
    print("Preparazione Test...")
    
    if best_model_path:
        print(f"Caricamento miglior checkpoint: {best_model_path}")
        
        # Fix sicurezza YACS
        torch.serialization.add_safe_globals([CfgNode])

        # Carica usando la classe corretta
        final_model = ModelClass.load_from_checkpoint(checkpoint_path=best_model_path, cfg=cfg)
        
        # Se è MemeCLIP, dobbiamo ri-applicare la patch anche al modello caricato!
        if cfg.model.name == 'MemeCLIP':
            final_model.common_step = patched_common_step.__get__(final_model, MemeCLIP)
            
    else:
        print("Uso il modello corrente (senza training/checkpoint).")
        final_model = model

    print("Avvio Test...")
    trainer.test(final_model, dataloaders=test_loader)

if __name__ == "__main__":
    main()
