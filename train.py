from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights
from numpy import random
from arch_framework import AddNormalization, FeedForward, EncoderLayer, Encoder
import pandas as pd
import numpy as np
from datasets import load_dataset,Dataset
from transformers import BertTokenizer

# To prepare prompt template from dataframe
def prepare_prompt(df):
    return f"""
    <movie_name>: {df['title']}, 
    <release_year>: {df['release_year']} ,
    <genre>: {df['genre']}, 
    <plot_summary>:{df['plot_summary']} ,
    <cast>: {df['cast']},
    <rating_imdtrain.to_csv('train.csv',index=False)b>: {df['rating_imdb']},
    <rating_rotten_tomatoes>: {df['rating_rotten_tomatoes']}
""".strip()

# Prepare trainable dataset from dataset
def prepare_dataset(dataset):
    full_prompt = prepare_prompt(dataset)
    full_prompt_token = tokenizer(full_prompt,
                                  padding=True,
                                  truncation=True,
                                  max_length = 256)
    
    full_prompt_token['labels'] = full_prompt_token['input_ids']
    return full_prompt_token


# Load train and valid CSVs
train_df =  pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize training dataframe
tokenized_train = train_df.map(prepare_dataset)
print(f"Tokenized training daaframe: {tokenized_train}")

# Tokenize valid dataframe
tokenized_valid = valid_df.map(prepare_dataset)
print(f"Tokenized training daaframe: {tokenized_valid}")



# Training hyper parameters
enc_vocab_size = 20 # Vocabulary size for the encoder
input_seq_length = 5  # Maximum length of the input sequence
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 6  # Number of layers in the encoder stack

batch_size = 64  # Batch size from the training process
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

input_seq = tokenized_train

encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
print(encoder(input_seq, padding_mask=None, training=True))

