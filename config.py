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
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
DROPOUT_RATE = 0.1
CHECKPOINT_DIR = "checkpoints" 
N_EPOCHS = 1
PRINT_FREQ = 100

# Model structure
D_MODEL = 64
FFN_DIM = 512
N_LAYERS = 4
N_HEADS = 8
