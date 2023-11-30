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


class CrossAttention(BaseAttention):
    """Cross-attention layer connecting the decoder with the encoder."""

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Apply BaseAttention layer with decoder query and encoder key/value.

        Args:
            x: Input tensor representing the decoder target sequence.
            context: Context tensor from the encoder.

        Returns:
            Output tensor
        """
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):
    """Self-attention layer for processing the encoder context."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply BaseAttention with encoder context as query, key and value.

        Args:
            x: Input tensor representing the encoder context

        Returns:
            Output tensor
        """
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


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


class EncoderLayer(tf.keras.layers.Layer):
    """Encoder layer with a self-attention layer and a feed-forward network."""

    def __init__(
        self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1
    ):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model=d_model, dff=dff)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Perform a forward pass on the encoder layer.

        Args:
            x: Input tensor representing the encoder context

        Returns:
            Output tensor
        """
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    """Encoder with a positional embedding layer and several encoder layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        vocab_size: int,
        dropout_rate: int = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model
        )
        self.enc_layers = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Perform a forward pass on the encoder.

        Args:
            x: Input tensor representing token indices in the input sequence.

        Returns:
            Encoded output tensor
        """
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x  # Shape (batch_size, seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    """Decoder layer with a cross-attention, self-attention and feed-forward."""

    def __init__(
        self, d_model: int, num_heads: int, dff: int, dropout_rate: int = 0.1
    ):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
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
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

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
        self.last_attn_scores = None

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

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x  # Shape (batch_size, seq_len, d_model)


class Transformer(tf.keras.Model):
    """Transformer model implemented with Keras."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_vocab_size: int,
        target_vocab_size: int,
        dropout_rate: int = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate,
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate,
        )
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Perform a forward pass on the Transformer.

        Args:
            inputs: To use a Keras model with `.fit` you must pass all your
                inputs in the first argument. Tuple with two input tensors
                representing the input token indices and the decoder sequence
                respectively.

        Returns:
            Transformer output
        """
        context, x = inputs
        context = self.encoder(context)  # (batch_size, context_len, d_model)
        x = self.decoder(x, context)  # (batch_size, target_len, d_model
        logits = self.final_layer(
            x
        )  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
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


class Translator(tf.Module):
    """Translation module using a transformer to translate Spanish into English.

    This class encapsulates a transformer model for language translation. It
    expects a sentence in the source language, tokenizes it, and uses the
    transformer model to generate the translated sentence in the target
    language. The class supports dynamic token generation and also provides
    attention weights from the transformer model, which can be useful for
    understanding the model's focus during translation.

    Attributes:
        tokenizers: Tokenizers for the source and target languages.
        transformer: A transformer model used for the translation task.
        max_length: Maximum length of translated sentence in terms of tokens.
    """

    def __init__(
        self, tokenizers, transformer: tf.keras.Model, max_length: int
    ):
        self.tokenizers = tokenizers
        self.transformer = transformer
        self.max_length = max_length

    def __call__(
        self, sentence: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Translates a sentence from Spanish to English.

        This method tokenizes the input sentence, feeds it into the transformer
        model, and generates a translated sentence in the target language. It
        also returns the tokens of the translated sentence and the attention
        weights from the transformer model.

        Args:
            sentence: A tensor containing the sentence to be translated.

        Returns:
            A tuple containing the following elements:
               - text: A tensor with the translated sentence.
               - tokens: A tensor with the tokens of the translated sentence.
               - attention_weights: Tensor with Transformer attention weights.
        """
        # Input sentence is English, hence adding `[START]` and `[END]` tokens.
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.eng.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # As the output language is Spanish, initialize the output with the
        # Spanish `[START]` token.
        start_end = self.tokenizers.spa.tokenize([""])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that
        # the dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(self.max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer(
                [encoder_input, output], training=False
            )

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[
                :, -1:, :
            ]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = self.tokenizers.spa.detokenize(output)[0]  # Shape: `()`.

        tokens = self.tokenizers.spa.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        result, tokens, attention_weights = self.translator(sentence)
        return result
