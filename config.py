"""Configuration file for the project."""

# Dataset
DATA_DIR = "data"
RAW_DATA_PATH = DATA_DIR + "/shakespeare.txt"
TRAIN_DATA_PATH = DATA_DIR + "/tfds/train"
VAL_DATA_PATH = DATA_DIR + "/tfds/validation"
VALIDATION_SHARE = 0.10

# Tokenizer
VOCAB_PATH = DATA_DIR + "/vocab.txt"
TOKENIZER_PATH = "tokenizer"
BERT_TOKENIZER_PARAMS = {"lower_case": False, "keep_whitespace": False}
LEARN_PARAMS = {}
NEWLINE_TOKEN = " NEWLINE "
DOUBLEN_TOKEN = " DOUBLEN "
RESERVED_TOKENS = []
VOCAB_SIZE = 6000

# Model training
MAX_TOKENS = 128
N_SAMPLES = 100_000
BUFFER_SIZE = 100_000
BATCH_SIZE = 20
DROPOUT_RATE = 0.1
CHECKPOINT_DIR = "checkpoints"
N_EPOCHS = 10
PRINT_FREQ = 100
OPTIMIZER_KWARGS = {"beta_1": 0.9, "beta_2": 0.98, "epsilon": 1e-9}

# Model structure
D_MODEL = 128
FFN_DIM = D_MODEL * 4
N_LAYERS = 8
N_HEADS = 8

# GPT
GPT_PATH = "gpt"
