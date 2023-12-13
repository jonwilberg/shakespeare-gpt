"""Model definition based on tensorflow.org/text/tutorials/transformer"""
import numpy as np
import tensorflow as tf


def positional_encoding(length: int, depth: int) -> tf.Tensor:
    """Generates positional encodings for an input sequence.

    Positional encodings are essential in sequence-to-sequence models, such as
    Transformers, to provide information about the position of elements in the
    input sequence.

    Args:
        length: The length of the input sequence.
        depth: The depth of the positional encoding, equal to the size of the
            hidden layers in the Transformer.

    Returns:
        Positional encoding tensor with shape (length, depth), cast to float32.
    """
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1
    )

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    """Custom Keras layer for adding positional encoding to token embedding."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, mask_zero=True
        )
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass combining token embeddings and positional encodings.

        Args:
            x: Input tensor representing token indices of the input sequence.

        Returns:
            Output tensor with positional encodings added to token embeddings.
        """
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # Set the relative scale of the embedding and positional encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    """Layer with multi-head attention, residual connection and layer norm."""

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CausalSelfAttention(BaseAttention):
    """Causal self-attention layer for processing the decoder sequence."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply BaseAttention with causal filter, decoder as query/key/value.

        Args:
            x: Input tensor representing the decoder sequence

        Returns:
            Output tensor
        """
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    """Feed-forward network."""

    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply forward pass of FeedForward network.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    """Decoder layer with a cross-attention, self-attention and feed-forward."""

    def __init__(
        self, d_model: int, num_heads: int, dff: int, dropout_rate: int = 0.1
    ):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Perform a forward pass on the decoder layer.

        Args:
            x: Input tensor representing the decoder sequence
            context: Output from the encoder

        Returns:
            Output tensor
        """
        x = self.causal_self_attention(x=x)
        x = self.ffn(x)
        return x  # Shape (batch_size, seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """Encoder with a positional embedding layer and several decoder layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        vocab_size: int,
        dropout_rate: int = 0.1,
    ):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ]
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Perform a forward pass on the decoder.

        Args:
            x: Input tensor representing the decoder sequence
            context: Output from the encoder

        Returns:
            Decoder output
        """
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
        logits = self.final_layer(x)  # (batch_size, target_len, vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def masked_loss(label: tf.Tensor, pred: tf.Tensor) -> float:
    """Computes the loss for predictions with a mask applied to ignore padding.

    Args:
        label: A tensor of true labels. Zeros are ignored in loss calculation.
        pred: A tensor of predictions made by the model.

    Returns:
        Average loss after applying the mask.
    """
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label: tf.Tensor, pred: tf.Tensor) -> float:
    """Computes accuracy for predictions with a mask applied to ignore padding.

    Args:
        label: A tensor of true labels. Zeros are ignored in loss calculation.
        pred: A tensor of predictions made by the model.

    Returns:
        Average loss after applying the mask.
    """
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)
