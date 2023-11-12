"""Model definition based on tensorflow.org/text/tutorials/transformer"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time


def positional_encoding(length: int, depth: int) -> tf.Tensor:
    """Generates positional encodings for an input sequence of a given length and depth.

    Positional encodings are essential in sequence-to-sequence models, such as transformers,
    to provide information about the position of elements in the input sequence.

    Args:
        length: The length of the input sequence.
        depth: The depth of the positional encoding, equal to the size of the hidden layers.

    Returns:
        A positional encoding tensor with shape (length, depth), cast to float32.
    """
    depth = depth/2
    
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    
    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)
    
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    """Custom Keras layer for adding positional embeddings to token embeddings in a transformer model."""
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)
    
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass of the layer, combining token embeddings and positional encodings.

        Args:
            x: Input tensor representing the token indices in the input sequence

        Returns:
            Output tensor with positional embeddings added to the token embeddings
        """
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    """Keras layer combining multi-head attention with a residual connection and layer normalization."""
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    """Cross-attention layer connecting the decoder query with the encoder key/value."""
    
    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Apply BaseAttention layer with decoder target sequence as query and encoder context as key and value.

        Args:
            x: Input tensor representing the decoder target sequence
            context: Context tensor from the encoder

        Returns:
            Output tensor
        """
        attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)
        
        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):
    """Self-attention layer for processing the encoder context."""
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply BaseAttention layer with encoder context as query, key and value.

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
        """Apply BaseAttention layer with a causal filter and the decoder sequence as query, key and value.

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
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Apply the two dense layers with ReLU in-between, dropout, residual connection and layer normalization.
        
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
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
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
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int, dropout_rate: int = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Perform a forward pass on the encoder.
        
        Args:
            x: Input tensor representing the token indices in the input sequence

        Returns:
            Encoded output tensor
        """
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x  # Shape (batch_size, seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    """Decoder layer with a cross-attention layer, a self-attention layer and a feed-forward network."""
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: int = 0.1):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
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
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int, dropout_rate: int = 0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
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
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=input_vocab_size, dropout_rate=dropout_rate)
    self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=target_vocab_size,dropout_rate=dropout_rate)
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
    context, x  = inputs
    context = self.encoder(context)  # (batch_size, context_len, d_model)
    x = self.decoder(x, context)  # (batch_size, target_len, d_model
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits
