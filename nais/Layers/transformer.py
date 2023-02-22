
import tensorflow.keras.layers as tfl
import tensorflow as tf 

class TransformerBlock(tfl.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tfl.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tfl.Dense(ff_dim, activation="relu"), tfl.Dense(embed_dim)]
        )
        self.layernorm1 = tfl.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tfl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tfl.Dropout(rate)
        self.dropout2 = tfl.Dropout(rate)

    def call(self, inputs, training):
        if isinstance(inputs, (list, tuple)):
            query, value = inputs 
        else: 
            query, value = inputs, inputs 
        
        attn_output = self.att(query, value)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(query + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)