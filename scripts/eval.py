from vocab_generation import FTVocabulary
from tokenizer import Tokenizer

from batch_generator import BatchGeneratorBinary, BatchGenerator
from keras.models import load_model
from settings import AGRR_TEST_SET, BINARY_MODEL, SEQ_LABELING_MODEL

import numpy as np
import pandas as pd
import csv
import os

from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.attention import MultiHeadSelfAttention
from keras_transformer.transformer import LayerNormalization


def generate_vocab(paths: list):
    vocab = FTVocabulary()
    for path in paths:
        vocab.extend_vocab(path)
    vocab.release_memory()
    return vocab


class Predictor:
    def __init__(self, model_binary_path: str, model_seq_labeling_path: str, samples, pd_samples, vocab):
        self.vocab = vocab
        self.samples = samples
        self.pd_samples = pd_samples
        self.custom_objects = {
            'TransformerCoordinateEmbedding': TransformerCoordinateEmbedding,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'LayerNormalization': LayerNormalization
        }
        self.model_binary = self.load_model(model_binary_path)
        self.model_seq_labeling = self.load_model(model_seq_labeling_path)

    def load_model(self, path):
        model = load_model(path, custom_objects=self.custom_objects)
        return model

    def predict_proba_binary(self, samples):
        """
        Predict tags for samples among all classes (probability distribution)
        """
        x = BatchGeneratorBinary(self.vocab, samples, 64).get_samples(samples)

        return self.model_binary.predict(x, batch_size=64, verbose=1)

    def predict_proba_seq_labeling(self, samples):

        x = BatchGenerator(self.vocab, samples, 64).get_samples(samples)
        return self.model_seq_labeling.predict(x, batch_size=64, verbose=1)

    def predict_tags(self, ind, probs_s):
        ret = []
        for probs in probs_s[-len(self.samples[ind][0]):]:
            ret.append(self.vocab.index_to_target[np.argmax(probs)])
        return ret

    def get_sample_characters(self, sample, pred_tags):
        ret = []
        for word, pred_tag in zip(sample, pred_tags):
            for character in word:
                ret.append([character, pred_tag])
        return ret

    def extend_sample_characters(self, text, s_characters):
        ret = []
        i_tagged = 0
        for i_text in range(len(text)):
            if i_tagged >= len(s_characters):
                ret.append([text[i_text], 'nG'])
            elif s_characters[i_tagged][0] == text[i_text]:
                ret.append(s_characters[i_tagged])
                i_tagged += 1
            else:
                ret.append([text[i_text], 'nan'])
        return ret

    def fill_nan(self, s_characters):
        for i_text, sample_character in enumerate(s_characters):
            if sample_character[1] == 'nan':
                if i_text == 0:
                    s_characters[i_text][1] = 'nG'
                else:
                    previous_tag = s_characters[i_text - 1][1]
                    for j in range(i_text + 1, len(s_characters)):
                        if s_characters[j][1] != 'nan':
                            next_tag = s_characters[j][1]
                            tag_to_set = next_tag if next_tag == previous_tag else 'nG'
                            for k in range(i_text, j):
                                s_characters[k][1] = tag_to_set
                            break
                    if s_characters[i_text][1] == 'nan':
                        s_characters[i_text][1] = 'nG'

    def evaluate(self):
        indexes_with_gapping = []
        for (sample, prob, (index, row)) in zip(
                self.samples, self.predict_proba_binary(self.samples), self.pd_samples.iterrows()):
            if prob[0] < 0.5:
                self.pd_samples.at[index, 'class'] = '0'
            else:
                self.pd_samples.at[index, 'class'] = '1'
                indexes_with_gapping.append(index)
        probs_seq_labeling = self.predict_proba_seq_labeling([self.samples[i] for i in indexes_with_gapping])

        for index, probs_sample in zip(indexes_with_gapping, probs_seq_labeling):
            predicted_tags = self.predict_tags(index, probs_sample)

            sample_characters = self.get_sample_characters(self.samples[index][0], predicted_tags)

            sample_characters_extended = self.extend_sample_characters(
                self.pd_samples.iloc[index]["text"], sample_characters)

            self.fill_nan(sample_characters_extended)

            i = 0
            while i < len(sample_characters_extended):
                if sample_characters_extended[i][1] != 'nG':
                    tag = sample_characters_extended[i][1]
                    bound = str(i) + ':'
                    while i < len(sample_characters_extended) and \
                            sample_characters_extended[i][1] == tag:
                        i += 1
                    bound = bound + str(i)
                    if self.pd_samples.at[index, tag] == '':
                        self.pd_samples.at[index, tag] = bound
                    else:
                        self.pd_samples.at[index, tag] = self.pd_samples.at[index, tag] + ' ' + bound
                else:
                    i += 1
            bounds = str()
            tag = 'R2'
            if self.pd_samples.at[index, 'R2'] == '':
                tag = 'R1'
            if self.pd_samples.at[index, 'R1'] == '':
                pass
            else:
                for bound in self.pd_samples.at[index, tag].split(' '):
                    bounds = bounds + bound.split(':')[0] + ':' + bound.split(':')[0] + ' '
                bounds = bounds[:-1]
                self.pd_samples.at[index, 'V'] = bounds


if __name__ == "__main__":
    ft_vocab = generate_vocab([AGRR_TEST_SET])
    test_samples = []
    Tokenizer().load_samples(AGRR_TEST_SET, test_samples)
    pandas_samples = pd.read_csv(AGRR_TEST_SET, sep='\t', quoting=csv.QUOTE_NONE)
    for column in ['class', 'cV', 'cR1', 'cR2', 'R1', 'R2', 'V']:
        pandas_samples[column] = ''
    predictor = Predictor(BINARY_MODEL, SEQ_LABELING_MODEL, test_samples, pandas_samples, ft_vocab)
    predictor.evaluate()

    try:
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"))
    except FileExistsError:
        pass
    pandas_samples.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results/parsed.csv"),
                          sep='\t', quoting=csv.QUOTE_NONE)

