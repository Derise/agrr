from typing import List, Tuple

import numpy as np
from vocab_generation import FTVocabulary

from keras.utils import Sequence


class BatchGenerator(Sequence):
    """
    Generates batches to fit model
    """

    def __init__(self, vocab: FTVocabulary,
                 samples: List, batch_size,
                 shuffle: bool=False, with_weights: bool=False):
        self.batch_size = batch_size  # type: int
        self.samples = samples
        self.shuffle = shuffle
        self.with_weights = with_weights
        self.vocab = vocab  # type: FTVocabulary

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        start_pos = index * self.batch_size
        end_pos = (index + 1) * self.batch_size

        x, y, sample_weights = self.to_tensor(
            self.samples[start_pos:min(end_pos, len(self.samples))])

        if self.with_weights:
            return x, y, sample_weights
        else:
            return x, y

    def to_tensor(self, samples: List[Tuple[List[str], List[str]]]) -> Tuple[np.array, np.array, np.array]:
        """
        Transform samples into tensors

        :param samples: samples as list of tuples(words, tags).
        :return: word vectors, vectors for tags
        """
        n = len(samples)
        samples_len_high = 300  # type: int

        word_vectors = np.zeros((n, samples_len_high, self.vocab.input_vector_size), dtype=np.float)
        y = np.zeros((n, samples_len_high, self.vocab.target_vector_size), dtype=np.int)
        sample_weights = np.zeros((n, samples_len_high), dtype=np.float)
        for i, sample in enumerate(samples):
            for j, (word, tag) in enumerate(zip(*sample[:-1])):
                word_vectors[i, -len(sample[0]) + j] = self.vocab.vocab_input[word]
                y[i, -len(sample[0]) + j] = self.vocab.vocab_target[tag]
                sample_weights[i, -len(sample[0]) + j] = self.vocab.target_weights[tag]
        return word_vectors, y, sample_weights

    def get_samples(self, samples: List[Tuple[List[str], List[str]]]) -> np.array:
        n = len(samples)
        samples_len_high = 300

        word_vectors = np.zeros((n, samples_len_high, self.vocab.input_vector_size), dtype=np.float)
        for i, sample in enumerate(samples):
            for j, word in enumerate(sample[0]):
                word_vectors[i, -len(sample[0]) + j] = self.vocab.vocab_input[word]
        return word_vectors


class BatchGeneratorBinary(Sequence):
    """
    Generates batches to fit model
    """

    def __init__(self, vocab: FTVocabulary,
                 samples: List, batch_size,
                 shuffle: bool = False):
        self.batch_size = batch_size  # type: int
        self.samples = samples
        self.shuffle = shuffle
        self.vocab = vocab  # type: FTVocabulary

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        start_pos = index * self.batch_size
        end_pos = (index + 1) * self.batch_size

        x, y = self.to_tensor(
            self.samples[start_pos:min(end_pos, len(self.samples))])
        return x, y

    def to_tensor(self, samples: List[Tuple[List[str], List[str], int]]) -> Tuple[np.array, np.array]:
        """
        Transform samples into tensors

        :param samples: samples as list of tuples(words, tags).
        :return: word vectors, vectors for tags
        """
        n = len(samples)
        samples_len_high = max(len(s[0]) for s in samples)  # type: int

        word_vectors = np.zeros((n, samples_len_high, self.vocab.input_vector_size), dtype=np.float)
        y = np.zeros(n, dtype=np.int)
        for i, sample in enumerate(samples):
            for j, word in enumerate(sample[0]):
                word_vectors[i, -len(sample[0]) + j] = self.vocab.vocab_input[word]
            y[i] = sample[2]
        return word_vectors, y

    def get_samples(self, samples: List[Tuple[List[str], List[str], int]]) -> np.array:
        n = len(samples)
        samples_len_high = max(len(s[0]) for s in samples)

        word_vectors = np.zeros((n, samples_len_high, self.vocab.input_vector_size), dtype=np.float)
        for i, sample in enumerate(samples):
            for j, word in enumerate(sample[0]):
                word_vectors[i, -len(sample[0]) + j] = self.vocab.vocab_input[word]
        return word_vectors
