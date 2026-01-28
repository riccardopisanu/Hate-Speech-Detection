import os
from yacs.config import CfgNode

# Inizializzazione di base della configurazione
cfg = CfgNode()

# =========================================================================
# 1. DEFINIZIONE DELLO SCHEMA (CRUCIALE)
# Deve avvenire immediatamente dopo CfgNode() per prevenire i KeyErrors.
# =========================================================================

cfg.model = CfgNode()
cfg.data = CfgNode()
cfg.training = CfgNode()

# =========================================================================
# 2. DEFINIZIONI GLOBALI
# =========================================================================


cfg.name = 'MemeCLIP'
cfg.label = 'misogynous'        # Chiave usata per la Task A di MAMI
cfg.seed = 42
cfg.test_only = False
cfg.device = 'cuda'
cfg.gpus = [0]                  # Utilizza la prima GPU
cfg.reproduce = False

# =========================================================================
# 3. SEZIONE MODELLO (corrisponde alla sezione 'model' nel YAML)
# =========================================================================

cfg.model.name = 'CLIP'
cfg.model.clip_model = "ViT-L/14" # Default, verrà sovrascritto da ViT-B/32 nel YAML

# =========================================================================
# 4. SEZIONE DATI (corrisponde alla sezione 'data' nel YAML)
# =========================================================================

# Identificatore del dataset
cfg.data.dataset_name = 'MAMI' 

# Parametri e percorsi
cfg.data.root_dir = ''             # Percorso delle immagini (sarà sovrascritto)
cfg.data.train_file = ''           # Percorso al CSV di train (sarà sovrascritto)
cfg.data.val_file = ''             # Percorso al CSV di validation (sarà sovrascritto)
cfg.data.test_file = ''            # Percorso al CSV di test (sarà sovrascritto)

# Parametri del DataLoader e delle immagini
cfg.data.batch_size = 16
cfg.data.image_size = 224

cfg.data.label = 'misogynous' # Default per il tipo di label

cfg.data.num_workers = 4 # Default standard (o 0), verrà sovrascritto da 8 nel tuo YAML
# AGGIUNTA FONDAMENTALE (Risolve KeyError: data.num_classes)
cfg.data.num_classes = 2           # Deve essere definito qui per lo schema YACS

# Mappatura per compatibilità con il codice legacy (se il codice cerca 'img_folder')
cfg.data.img_folder = cfg.data.root_dir

# =========================================================================
# 5. SEZIONE TRAINING (corrisponde alla sezione 'training' nel YAML)
# =========================================================================

# Iperparametri
cfg.training.optimizer = 'AdamW'
cfg.training.learning_rate = 1e-4
cfg.training.max_epochs = 10
cfg.training.weight_decay = 1e-4

cfg.training.num_epochs = 10 # Default, verrà sovrascritto da 15 nel tuo YAML
cfg.training.gpus = [0] # Default (verrà sovrascritto da gpus: [0] nel tuo YAML)

# Percorsi di output
cfg.training.log_dir = ''
cfg.training.checkpoint_dir = ''

# Costruzione del percorso del checkpoint (si basa sul percorso che sarà sovrascritto dal YAML)
cfg.training.checkpoint_path = os.path.join(cfg.training.checkpoint_dir, 'checkpoints')
cfg.training.checkpoint_file = os.path.join(cfg.training.checkpoint_path, 'model.ckpt')


# =========================================================================
# 6. LOGICA DINAMICA CLASSI (Multi-Dataset)
# =========================================================================

def update_dataset_logic(_cfg):
    """Utility per aggiornare classi e label in base al dataset scelto."""
    if _cfg.data.dataset_name == 'MAMI':
        _cfg.data.label = 'misogynous'
        _cfg.class_names = ['Not Misogynous', 'Misogynous']
    elif _cfg.data.dataset_name == 'MultiOFF':
        _cfg.data.label = 'label'
        _cfg.class_names = ['Non-Offensive', 'Offensive']
    elif _cfg.data.dataset_name == 'HatefulMemes':
        _cfg.data.label = 'label'
        _cfg.class_names = ['Not Hateful', 'Hateful']
    
    _cfg.data.num_classes = len(_cfg.class_names)

# Inizializzazione di default
update_dataset_logic(cfg)

# Parametri architettura
cfg.data.num_classes = len(cfg.class_names) # <--- NUOVA POSIZIONE PER IL CALCOLO
cfg.num_mapping_layers = 1
cfg.unmapped_dim = 768
cfg.map_dim = 1024
cfg.num_pre_output_layers = 1
cfg.drop_probs = [0.1, 0.4, 0.2]
cfg.ratio = 0.2
cfg.scale = 30
cfg.print_model = True
