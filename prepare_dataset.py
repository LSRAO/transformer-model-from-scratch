from numpy.random import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64
import pandas as pd
from datasets import load_dataset,Dataset
from pickle import load, dump, HIGHEST_PROTOCOL


class PrepareDataset:
    def __init__(self, n_sentences=10000, train_split=1.0):
        self.n_sentences = n_sentences  # Number of sentences to include in the dataset
        self.train_split = train_split  # Ratio of the training data split

    # Fit a tokenizer
    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)

        return tokenizer

    def find_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)

    def find_vocab_size(self, tokenizer, dataset):

        tokenizer.fit_on_texts(dataset)
        return len(tokenizer.word_index) + 1
    
    def make_list(self,dataset):
            # return f"""<START> Movie_name : {dataset['title']}, Length : {dataset['length']}, Release_year : {dataset['release_year']}, Genere : {dataset['genre']}, Plot_summary : {dataset['plot_summary']}, Cast : {dataset['cast']}, imdb_rating : {dataset['rating_imdb']}, Rating_rotten_tomatoes : {dataset['rating_rotten_tomatoes']} <EOS>"""
            return f"""<START> PAGE: {dataset['page']}, section_title: {dataset['section_title']}, text: {dataset['text']} <EOS>"""
    
    def save_tokenizer(self, tokenizer, name):
        with open(name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)

    def __call__(self, filename):
        # Load the CSV file
        train = Dataset.from_csv(filename)
        # print(train.describe())

        # Extract input and target columns 

        
        ds_string = []
        for i in train:
            ds_string.append(self.make_list(i))
        # Random shuffle the dataset
        shuffle(ds_string)
        # print(ds_string[0])
        # return 0

        # Split the dataset
        train_size = int(self.n_sentences * self.train_split)
        train = ds_string[:train_size]
        # print(train)

        # Prepare tokenizer for the encoder input
        enc_tokenizer = self.create_tokenizer([inp for inp in train])
        enc_seq_length = self.find_seq_length([inp for inp in train])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer, [inp for inp in train])

        # Prepare tokenizer for the decoder input
        dec_tokenizer = self.create_tokenizer([tgt for tgt in train])
        dec_seq_length = self.find_seq_length([tgt for tgt in train])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer, [tgt for tgt in train])

        # Encode and pad the input sequences
        trainX = enc_tokenizer.texts_to_sequences([inp for inp in train])
        trainX = pad_sequences(trainX, maxlen=enc_seq_length, padding='post')
        trainX = convert_to_tensor(trainX, dtype=int64)

        # Encode and pad the target sequences
        trainY = dec_tokenizer.texts_to_sequences([tgt for tgt in train])
        trainY = pad_sequences(trainY, maxlen=dec_seq_length, padding='post')
        trainY = convert_to_tensor(trainY, dtype=int64)

        # Save the encoder tokenizer
        self.save_tokenizer(enc_tokenizer, 'enc')

        # Save the decoder tokenizer
        self.save_tokenizer(dec_tokenizer, 'dec')

        return trainX, trainY, train, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size


# Prepare the training data
# dataset = PrepareDataset()
# trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('train.csv')

# print(train_orig[0], '\n', trainX[0, :])
# print(trainY)