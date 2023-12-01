"""Configuration file for the project."""

# Dataset
DATA_DIR = "data"
RAW_DATA_PATH = DATA_DIR + "/en-nb.txt"
TRAIN_DATA_PATH = DATA_DIR + "/tfds/train"
VAL_DATA_PATH = DATA_DIR + "/tfds/validation"
VALIDATION_SHARE = 0.20
SENTENCE_MAX_LEN = 500
N_SENTENCES = 1_000_000

# Tokenizer
TOKENIZER_PATH = "tokenizer"
BERT_TOKENIZER_PARAMS = {
    "lower_case": False,
    "normalization_form": "NFC",  # NFC norm to avoid 'å' to 'a' convert
}
RESERVED_TOKENS = ["[PAD]", "[UNK]", "[START]", "[END]"]
N_SAMPLES_TOKENIZER = 50_000
VOCAB_SIZE = 8000

# Model training
MAX_TOKENS = 64
BUFFER_SIZE = 100_000
BATCH_SIZE = 50
DROPOUT_RATE = 0.1
CHECKPOINT_DIR = "checkpoints"
N_EPOCHS = 3
PRINT_FREQ = 100
OPTIMIZER_KWARGS = {"beta_1": 0.9, "beta_2": 0.98, "epsilon": 1e-9}

# Model structure
D_MODEL = 64
FFN_DIM = 512
N_LAYERS = 4
N_HEADS = 8

# Translator
TRANSLATOR_PATH = "translator"
