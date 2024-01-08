# Train Transformer decoder

import tensorflow as tf
import tensorflow_text as text  # noqa: F401

import config
from model import GPT, CustomSchedule, Decoder, masked_accuracy, masked_loss


def split_input_target(sequence: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Split a sequence of tokens into input and target tensors.

    Args:
        sequence: A tensor of shape (seq_len, vocab_size)

    Returns:
        Input sequence and target sequence
    """
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def make_batches(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Split the dataset into input, target pairs and batch them.

    Args:
        ds: Dataset of tokenized sequences

    Returns:
        Batched dataset of input, target pairs
    """
    return (
        ds.map(split_input_target, tf.data.AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


def load_dataset() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load the dataset and split into batches.

    Returns:
        Training and validation batches
    """
    train_dataset = tf.data.Dataset.load(config.TRAIN_DATA_PATH)
    val_dataset = tf.data.Dataset.load(config.VAL_DATA_PATH)
    train_batches = make_batches(train_dataset)
    val_batches = make_batches(val_dataset)
    return train_batches, val_batches


def get_compiled_decoder() -> tf.keras.Model:
    """Get a compiled decoder model.

    Returns:
        Compiled decoder model
    """
    learning_rate = CustomSchedule(config.D_MODEL)
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=learning_rate, **config.OPTIMIZER_KWARGS
    )
    decoder = Decoder(
        num_layers=config.N_LAYERS,
        d_model=config.D_MODEL,
        num_heads=config.N_HEADS,
        dff=config.FFN_DIM,
        vocab_size=config.VOCAB_SIZE,
        dropout_rate=config.DROPOUT_RATE,
    )
    decoder.compile(
        loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
    )
    return decoder


if __name__ == "__main__":
    train_batches, val_batches = load_dataset()
    tokenizer = tf.saved_model.load(config.TOKENIZER_PATH)
    decoder = get_compiled_decoder()

    # Build the transformer by applying it to a sample
    for input_text, target_text in train_batches.take(1):
        break
    decoder(input_text)
    print(decoder.summary())

    decoder.fit(
        train_batches, epochs=config.N_EPOCHS, validation_data=val_batches
    )

    gpt = GPT(decoder=decoder, tokenizer=tokenizer)
    tf.saved_model.save(gpt, export_dir=config.GPT_PATH)
