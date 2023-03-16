
import tensorflow.keras.layers as tfl
import tensorflow as tf 

from nais.Layers import Patches1D

class TransformerBlock(tfl.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tfl.MultiHeadAttention(num_heads=num_heads, 
                                          key_dim=embed_dim)  
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
    
class PatchTransformerBlock(tfl.Layer):
    def __init__(self, patch_size, patch_stride, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.patching = Patches1D(patch_size, patch_stride)
        self.att = tfl.MultiHeadAttention(num_heads=num_heads,
                                          key_dim=embed_dim)  
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
        
        query_shape = query.shape
        
        query = self.patching(query)
        value = self.patching(value)
        
        attn_output = self.att(query, value)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(query + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)