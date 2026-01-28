import torch.nn as nn
from configs import cfg, update_dataset_logic # Importiamo anche la funzione logic
from my_datasets import load_dataset # Assicurati che il nome file sia corretto
from CLIP_MLP import CLIP_MLP_Classifier

import os
import glob
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

def main(): # Rimuoviamo cfg dagli argomenti, lo carichiamo internamente
    # --- SETUP ARGPARSE ---
    parser = argparse.ArgumentParser(description="Training Multi-Dataset")
    parser.add_argument("--config", type=str, default=None, help="Path al file YAML di configurazione")
    parser.add_argument("--dataset", type=str, default=None, help="Override nome dataset (MAMI, MultiOFF, HatefulMemes)")
    args = parser.parse_args()

    # --- MERGE CONFIGURAZIONE ---
    if args.config:
        cfg.merge_from_file(args.config)
    
    if args.dataset:
        cfg.data.dataset_name = args.dataset

    # --- AGGIORNAMENTO LOGICA DATASET (Fondamentale!) ---
    update_dataset_logic(cfg)

    seed_everything(cfg.seed, workers=True)

    # 1. CARICAMENTO DEL DATASET
    dataset_train = load_dataset(cfg=cfg, split='train')
    dataset_val = load_dataset(cfg=cfg, split='dev')
    dataset_test = load_dataset(cfg=cfg, split='test')

    # 2. CREAZIONE DEI DATALOADER
    train_loader = DataLoader(dataset_train, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(dataset_val, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(dataset_test, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    # 3. INIZIALIZZAZIONE DEL MODELLO
    model = CLIP_MLP_Classifier(cfg)

    # 4. SETUP DEL TRAINER (Cartella dinamica per dataset)
    monitor = "val/auroc"
    checkpoint_subdir = f"{cfg.data.dataset_name}_{cfg.name}"
    full_checkpoint_dir = os.path.join(os.path.abspath(cfg.training.checkpoint_dir), checkpoint_subdir)
    os.makedirs(full_checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=full_checkpoint_dir,
        filename='epoch={epoch}-step={step}',
        monitor=monitor,
        mode='max',
        verbose=True,
        save_weights_only=True,
        save_top_k=1
    )

    trainer = Trainer(
        accelerator='gpu',
        devices=cfg.training.gpus,
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback],
        deterministic=False
    )

    # 5. TRAINING
    if not cfg.test_only:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 6. CARICAMENTO AUTOMATICO E TEST
    list_of_files = glob.glob(os.path.join(full_checkpoint_dir, 'epoch=*-step=*.ckpt'))
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getmtime)
        print(f"\n--- Testing checkpoint: {latest_file} ---")
        final_model = CLIP_MLP_Classifier.load_from_checkpoint(checkpoint_path=latest_file, cfg=cfg)
    else:
        final_model = model

    trainer.test(final_model, dataloaders=test_loader)

if __name__ == "__main__":
    main()
