"""Keras models"""
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.svm import LinearSVC
from tensorflow.keras import Model, Input
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, initializers, constraints
from tensorflow.keras.layers import Layer, Input, Embedding, Attention, Flatten, LSTM, GRU,\
    Dropout, Activation, Dense, Bidirectional, Concatenate, \
    Lambda, Convolution1D, SpatialDropout1D, BatchNormalization, GlobalMaxPooling1D, GlobalAveragePooling1D



class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, name):
        super(TokenAndPositionEmbedding, self).__init__(name=name)
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def REC(Tx, alarm_num_in, alarm_num_out=None, hparams=None):
    '''
    Returns a model based on bidirectional LSTM
    
    Args:
        Tx: length of input sequence
        alarm_num_in: number of distinct alarm codes that can appear in an input sequence 
        alarm_num_out: number of labels that are predicted in output
        hparams: a dictionary that contains every parameter used for the model

    Example:
        #dictionary with parameters used to build the model
        hparams = {
            "hp_embedding_dim": 32,
            "hp_num_units": 128,
            "dropout_rate": 0.3
                }
        Tx = 109 #length of input sequence
        alarm_num_in = 154  
        alarm_num_out = 10  

        model = REC(Tx, alarm_num_in, alarm_num_out, hparams)
    '''
    if alarm_num_out is None:
        alarm_num_out = alarm_num_in
    if 'hp_activation' in hparams.keys():
        activation = hparams['hp_activation']
    else:
        warnings.warn('Warning! Using default value for activation: tanh')
        activation = 'tanh'
    if 'hp_embedding_dim' in hparams.keys():
        embedding_dim = hparams['hp_embedding_dim']
    else:
        warnings.warn('Warning! Using default value for embedding_dim: 64')
        embedding_dim = 64
    if 'hp_num_units' in hparams.keys():
        lstm_units = hparams['hp_num_units']
    else:
        warnings.warn('Warning! Using default value for lstm_units: 128')
        lstm_units = 128
    if 'hp_dropout_rate' in hparams.keys():
        dropout_rate = hparams['hp_dropout_rate']
    else:
        warnings.warn('Warning! Using default value for dropout_rate : 0.5')
        dropout_rate = 0.5

    alarm_codes = Input(shape=(Tx,), dtype='int32')
    embedding_layer = Embedding(input_dim=alarm_num_in, output_dim=embedding_dim,
                                input_length=Tx, trainable=True, mask_zero=True, name="embedding_layer")
    embedding = embedding_layer(alarm_codes)
    X = Bidirectional(LSTM(units=lstm_units, activation=activation))(embedding)
    X = BatchNormalization()(X)
    X = Dropout(rate=dropout_rate)(X)
    X = Dense(units=alarm_num_out)(X)
    X = BatchNormalization()(X)
    X = Activation('sigmoid')(X)
    model = Model(inputs=alarm_codes, outputs=X)
    return model


def ATT(Tx, alarm_num_in, alarm_num_out=None, hparams=None):
    '''
    Returns a model based on bidirectional LSTM and custom attention mechanism
    
    Args:
        Tx: length of input sequence
        alarm_num_in: number of distinct alarm codes that can appear in an input sequence 
        alarm_num_out: number of labels that are predicted in output
        hparams: a dictionary that contains every parameter used for the model

    Example:
        #dictionary with parameters used to build the model
        hparams = {
            "hp_embedding_dim": 32,
            "hp_num_units": 128,
            "dropout_rate": 0.3
                }
        Tx = 109 #length of input sequence
        alarm_num_in = 154  
        alarm_num_out = 10  

        model = ATT(Tx, alarm_num_in, alarm_num_out, hparams)
    '''
    if alarm_num_out is None:
        alarm_num_out = alarm_num_in
    if 'hp_activation' in hparams.keys():
        activation = hparams['hp_activation']
    else:
        warnings.warn('Warning! Using default value for activation: tanh')
        activation = 'tanh'
    if 'hp_embedding_dim' in hparams.keys():
        embedding_dim = hparams['hp_embedding_dim']
    else:
        warnings.warn('Warning! Using default value for embedding_dim: 64')
        embedding_dim = 64
    if 'hp_num_units' in hparams.keys():
        lstm_units = hparams['hp_num_units']
    else:
        warnings.warn('Warning! Using default value for lstm_units: 128')
        lstm_units = 128
    if 'hp_dropout_rate' in hparams.keys():
        dropout_rate = hparams['hp_dropout_rate']
    else:
        warnings.warn('Warning! Using default value for dropout_rate : 0.5')
        dropout_rate = 0.5
    alarm_codes = Input(shape=(Tx,), dtype='int32')
    embedding_layer = Embedding(input_dim=alarm_num_in, output_dim=embedding_dim,
                                input_length=Tx, trainable=True, mask_zero=True)
    embedding = embedding_layer(alarm_codes)
    X = Bidirectional(
        LSTM(units=lstm_units, activation=activation, return_sequences=True))(embedding)
    X = BatchNormalization()(X)
    X = Dropout(rate=dropout_rate)(X)
    X = AttentionWithContext()(X)
    X = BatchNormalization()(X)
    X = Dense(units=alarm_num_out, activation='sigmoid')(X)
    model = Model(inputs=alarm_codes, outputs=X)
    return model


def TRM(Tx, alarm_num_in, alarm_num_out=None, hparams=None):
    """
    Returns the model with transformer-based architecture

    Args:
        Tx: length of input sequence
        alarm_num_in: number of distinct alarm codes that can appear in an input sequence 
        alarm_num_out: number of labels that are predicted in output
        hparams: a dictionary that contains every parameter used for the model

    Example:
        #dictionary with parameters used to build the model
        hparams = {
            "hp_embedding_dim": 32,
            "hp_num_heads": 2,
            "hp_ff_dim": 128,
            "hp_nn_dim": 128,
            "dropout_rate": 0.3
                }
        Tx = 109 #length of input sequence
        alarm_num_in = 154  
        alarm_num_out = 10  

        model = TRM(Tx, alarm_num_in, alarm_num_out, hparams)
    """
    if alarm_num_out is None:
        alarm_num_out = alarm_num_in
    if 'hp_embedding_dim' in hparams.keys():
        embed_dim = hparams['hp_embedding_dim']
    else:
        warnings.warn('Warning! Using default value for embedding_dim: 64')
        embed_dim = 64
    if 'hp_dropout_rate' in hparams.keys():
        dropout = hparams['hp_dropout_rate']
    else:
        warnings.warn('Warning! Using default value for dropout_rate : 0.5')
        dropout = 0.5
    if 'hp_num_heads' in hparams.keys():
        num_heads = hparams['hp_num_heads']
    else:
        warnings.warn('Warning! Using default value for num_heads : 2')
        num_heads = 2
    if 'hp_activation' in hparams.keys():
        activation = hparams['hp_activation']
    else:
        warnings.warn('Warning! Using default value for activation: relu')
        activation = 'relu'
    if 'hp_nn_dim' in hparams.keys():
        nn_dim = hparams['hp_nn_dim']
    else:
        warnings.warn('Warning! Using default value for nn_dim: 128')
        nn_dim = 128
    if 'hp_ff_dim' in hparams.keys():
        ff_dim = hparams['hp_ff_dim']
    else:
        warnings.warn('Warning! Using default value for ff_dim: 128')
        ff_dim = 128

    inputs = layers.Input(shape=(Tx,))
    embedding_layer = TokenAndPositionEmbedding(Tx, alarm_num_in, embed_dim, name="TokenAndPositionEmbedding")
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x) 
    x = layers.Dense(nn_dim, activation=activation)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x) 
    outputs = layers.Dense(alarm_num_out, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs)
