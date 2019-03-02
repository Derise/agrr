import pandas as pd
import csv
import nltk


class Tokenizer:
    """
    Word tokenizer for entities in corpus.
    Implemented with nltk lib and external russian module from
    https://github.com/Mottl/ru_punkt
    """
    def __init__(self):
        # symbols nltk don't treat as expected
        self.wrong_characters = ['\xad', '\ufeff', '\x97', '®', '™', '►', '→', '\u20e3', '¨', '‼', '†', '⏳',
                                 '⌛', '\uf066', '▲', '·', '̈',
                                 '’', '¬', '′', '″', '‘', '∽', '°', '…', '§', '±', '€', '№', '£', '©', '∆',
                                 '\uf076', '■', '\uf0bb', '⏰', '\'']

    def generate_samples(self, corpus_path):
        corpus_data = pd.read_csv(corpus_path, sep='\t', quoting=csv.QUOTE_NONE)
        for _, row in corpus_data.iterrows():
            (words, tags, label) = self.tokenize(row)
            yield (words, tags, label)

    def load_samples(self, corpus_path, *args):
        if len(args) == 2:
            for (words, tags, label) in self.generate_samples(corpus_path):
                for (i, group) in enumerate(args[1]):
                    if group[0] <= len(words) <= group[1]:
                        args[0][i].append((words, tags, label))
                        break
        else:
            for (words, tags, label) in self.generate_samples(corpus_path):
                args[0].append((words, tags, label))

    def tokenize(self, row):
        """Generates words and tokens from annotated corpora.
        Args:
          row: a row in pandas dataframe
        Returns:
          tuple: (words, tags, label)
        """
        text_splitted_by_labels = []
        for column in ['cV', 'cR1', 'cR2', 'R1', 'R2']:
            if not pd.isnull(row[column]):
                for bounds_str in row[column].split():
                    bounds = list(map(int, bounds_str.split(':')))
                    text_splitted_by_labels.append(((bounds, column), row["text"][bounds[0]:bounds[1]]))
        if text_splitted_by_labels:
            pos = 0
            for text_portion in sorted(text_splitted_by_labels, key=lambda x: x[0][0][0]):
                if pos != text_portion[0][0][0]:
                    text_splitted_by_labels.append((([pos, text_portion[0][0][0]], 'nG'),
                                                    row["text"][pos:text_portion[0][0][0]]))
                pos = text_portion[0][0][1]
            text_splitted_by_labels.append((([pos, len(row["text"])], 'nG'),
                                            row["text"][pos:len(row["text"])]))
        else:
            text_splitted_by_labels.append((([0, len(row["text"])], 'nG'),
                                            row["text"][0:len(row["text"])]))
        text_splitted_by_labels.sort(key=lambda x: x[0][0][0])
        tokenized_text = []
        tags = []
        for text_portion in text_splitted_by_labels:
            tokenized_portion = nltk.word_tokenize(text_portion[1], language="russian")
            for word in tokenized_portion:
                word = ''.join([character for character in word if character not in self.wrong_characters])
                if word:
                    tokenized_text.append(word)
                    tags.append(text_portion[0][1])
        label = int(row["class"]) if not pd.isnull(row["class"]) else 0
        return tokenized_text, tags, label
