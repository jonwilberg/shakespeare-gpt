"""Configuration file for the project."""

# Preprocessing
DATA_DIR = "data"
TRAIN_PATH = DATA_DIR + "/spa.txt"
NONBREAKING_SPA_PATH = DATA_DIR + "/nonbreaking_prefix.es"
NONBREAKING_ENG_PATH = DATA_DIR + "/nonbreaking_prefix.en"
EOS_TOKEN = "eos"
SOS_TOKEN = "sos"
VOCAB_SIZE = 10_000
MAX_TOKENS = 64

# Model training
VAL_SHARE = 0.20
BUFFER_SIZE = 100_000
BATCH_SIZE = 50
DROPOUT_RATE = 0.1
CHECKPOINT_DIR = "checkpoints" 
N_EPOCHS = 1
PRINT_FREQ = 100
OPTIMIZER_KWARGS = {
    "beta_1": 0.9,
    "beta_2": 0.98,
    "epsilon": 1e-9
}

# Model structure
D_MODEL = 64
FFN_DIM = 512
N_LAYERS = 4
N_HEADS = 8
