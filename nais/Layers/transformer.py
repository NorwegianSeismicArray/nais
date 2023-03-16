
import tensorflow.keras.layers as tfl
import tensorflow as tf 
import numpy as np

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
    def __init__(self, patch_size, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rate = rate 
        self.ff_dim = ff_dim
        self.patching = Patches1D(patch_size, patch_size)
        
    def build(self, input_shape):
        if isinstance(input_shape, list):
            query, value = input_shape
        else: 
            query, value = input_shape, input_shape
        b, t, c = query
        num_patches = int(np.ceil(t/self.patch_size))
        self.reshape_query = tfl.Reshape((num_patches, -1))
        self.pos_embedding = tfl.Embedding(num_patches, c*self.patch_size)
        
        self.transformer = TransformerBlock(embed_dim=c*self.patch_size, 
                                            num_heads=self.num_heads, 
                                            ff_dim=c*self.patch_size, 
                                            rate=self.rate)
        
        b, t, c = value
        self.reshape_value = tfl.Reshape((int(np.ceil(t/self.patch_size)), -1))

        self.reshape_output = tfl.Reshape((-1,c))        
                
    def call(self, inputs, training):
        if isinstance(inputs, (list, tuple)):
            query, value = inputs 
        else: 
            query, value = inputs, inputs
        
        query = self.patching(query)
        value = self.patching(value)
        query = self.reshape_query(query)
        value = self.reshape_value(value)
        
        pos = self.pos_embedding(tf.range(0, query.shape[1], delta=1, dtype=tf.int32))
        
        query += pos
        
        out = self.transformer([query, value])
        
        return self.reshape_output(out)