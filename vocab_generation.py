import fastText
from settings import FAST_TEXT_MODEL

from tokenizer import Tokenizer
import numpy as np


class FTVocabulary:
    """
    Class generates and deals with word embeddings from fastText.
    """
    def __init__(self):
        # vocab does not contain tokens for tags
        # key - word, value - word vector
        self.vocab_input = {}
        # vocab with tokens for tags
        self.vocab_target = {'nG': np.array([1, 0, 0, 0, 0, 0], dtype=np.int),
                             'R2': np.array([0, 1, 0, 0, 0, 0], dtype=np.int),
                             'R1': np.array([0, 0, 1, 0, 0, 0], dtype=np.int),
                             'cR2': np.array([0, 0, 0, 1, 0, 0], dtype=np.int),
                             'cR1': np.array([0, 0, 0, 0, 1, 0], dtype=np.int),
                             'cV': np.array([0, 0, 0, 0, 0, 1], dtype=np.int),
                             }
        self.index_to_target = {0: 'nG',
                                1: 'R2',
                                2: 'R1',
                                3: 'cR2',
                                4: 'cR1',
                                5: 'cV'}
        self.target_weights = {'nG': 1, 'R2': 2, 'R1': 3, 'cR2': 2, 'cR1': 3, 'cV': 5}
        self.ru_model = fastText.load_model(FAST_TEXT_MODEL)
        self.input_vector_size = self.ru_model.get_dimension()
        self.target_vector_size = len(self.vocab_target)

    def extend_vocab(self, path):
        """
        Extends the vocabulary with words from path
        :param path: corpus path
        :return: None
        """
        for words, _, _ in Tokenizer().generate_samples(path):
            for word in words:
                if word not in self.vocab_input:
                    self.vocab_input[word] = self.ru_model.get_word_vector(word)

    def release_memory(self):
        """
        fastText model consumes a lot of memory so it is recommended to be released after vocab generation.
        :return: None
        """
        del self.ru_model
