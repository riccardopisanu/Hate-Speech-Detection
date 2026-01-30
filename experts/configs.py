import os
from yacs.config import CfgNode

# Inizializzazione di base della configurazione
cfg = CfgNode()

# =========================================================================
# 1. DEFINIZIONE DELLO SCHEMA GERARCHICO (Il nostro ordine)
# =========================================================================
cfg.model = CfgNode()
cfg.data = CfgNode()
cfg.training = CfgNode()

# =========================================================================
# 2. ALIAS DI COMPATIBILITÀ (PER MEMECLIP) - FONDAMENTALE
# =========================================================================
# MemeCLIP cerca queste variabili alla radice (es. cfg.num_classes).
# Le definiamo qui per evitare "AttributeError".
cfg.num_classes = 2
cfg.clip_variant = "ViT-L/14"
cfg.lr = 1e-4
cfg.weight_decay = 1e-2
cfg.class_names = []
cfg.device = 'cuda'

# Parametri architettura (devono esistere anche alla radice)
cfg.map_dim = 1024
cfg.unmapped_dim = 768
cfg.num_mapping_layers = 1
cfg.num_pre_output_layers = 3
cfg.drop_probs = [0.1, 0.4, 0.2]
cfg.ratio = 0.2

# =========================================================================
# 3. DEFINIZIONI GLOBALI
# =========================================================================
cfg.name = 'MemeCLIP'
cfg.label = 'misogynous'
cfg.seed = 42
cfg.test_only = False
cfg.gpus = [0]
cfg.reproduce = False

# =========================================================================
# 4. SEZIONE MODELLO
# =========================================================================
cfg.model.name = 'CLIP'
cfg.model.clip_model = "ViT-L/14"
cfg.model.hidden_dim = 512
cfg.model.dropout = 0.5

# Copia speculare per YACS (MemeCLIP legge config annidati dal yaml)
cfg.model.clip_variant = "ViT-L/14"
cfg.model.unmapped_dim = 768
cfg.model.map_dim = 1024
cfg.model.num_mapping_layers = 1
cfg.model.num_pre_output_layers = 3
cfg.model.drop_probs = [0.1, 0.4, 0.2]
cfg.model.ratio = 0.2
cfg.model.class_names = []

# =========================================================================
# 5. SEZIONE DATI
# =========================================================================
cfg.data.dataset_name = 'MAMI'
cfg.data.root_dir = ''
cfg.data.train_file = ''
cfg.data.val_file = ''
cfg.data.test_file = ''
cfg.data.batch_size = 16
cfg.data.image_size = 224
cfg.data.label = 'misogynous'
cfg.data.num_workers = 4
cfg.data.num_classes = 2
cfg.data.img_folder = '' 

# =========================================================================
# 6. SEZIONE TRAINING
# =========================================================================
cfg.training.optimizer = 'AdamW'
cfg.training.learning_rate = 1e-4
cfg.training.max_epochs = 10
cfg.training.weight_decay = 1e-4
cfg.training.num_epochs = 10
cfg.training.gpus = [0]
cfg.training.log_dir = './logs'
cfg.training.checkpoint_dir = './checkpoints'
cfg.training.device = 'cuda'

# =========================================================================
# 7. LOGICA DINAMICA (Sincronizzazione Alias)
# =========================================================================
# Parametri legacy
cfg.scale = 30
cfg.print_model = True

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

    # Aggiorniamo num_classes
    _cfg.data.num_classes = len(_cfg.class_names)
    
    # --- SINCRONIZZAZIONE ALIAS (MOLTO IMPORTANTE) ---
    # Copiamo i valori dalle sottosezioni alla radice per MemeCLIP
    _cfg.num_classes = _cfg.data.num_classes
    _cfg.model.class_names = _cfg.class_names
    
    # Se questi valori vengono cambiati dal YAML (model.x), aggiorniamo la radice
    _cfg.clip_variant = _cfg.model.clip_variant
    _cfg.map_dim = _cfg.model.map_dim
    _cfg.unmapped_dim = _cfg.model.unmapped_dim
    _cfg.lr = _cfg.training.learning_rate
    _cfg.weight_decay = _cfg.training.weight_decay

# Eseguiamo la logica una volta per inizializzare i default
update_dataset_logic(cfg)

# =========================================================================
# 8. FUNZIONE DI EXPORT
# =========================================================================
def get_cfg_defaults():
    """Ritorna una copia della configurazione di default"""
    # Rieseguiamo la logica di update prima di ritornare per sicurezza
    update_dataset_logic(cfg) 
    return cfg.clone()
