"""Configuration file for the project."""

# Dataset
DATA_DIR = "data"
RAW_DATA_PATH = DATA_DIR + "/spa.txt"
TRAIN_DATA_PATH = DATA_DIR + "/tfds/train"
VAL_DATA_PATH = DATA_DIR + "/tfds/validation"
VALIDATION_SHARE = 0.20

# Tokenizer
TOKENIZER_PATH = "tokenizer"
BERT_TOKENIZER_PARAMS = {"lower_case": True}
RESERVED_TOKENS = ["[PAD]", "[UNK]", "[START]", "[END]"]
VOCAB_SIZE = 3000

# Model training
MAX_TOKENS = 40
BUFFER_SIZE = 100_000
BATCH_SIZE = 50
DROPOUT_RATE = 0.1
CHECKPOINT_DIR = "checkpoints"
N_EPOCHS = 10
PRINT_FREQ = 100
OPTIMIZER_KWARGS = {"beta_1": 0.9, "beta_2": 0.98, "epsilon": 1e-9}

# Model structure
D_MODEL = 64
FFN_DIM = 512
N_LAYERS = 4
N_HEADS = 8

# Translator
TRANSLATOR_PATH = "translator"
