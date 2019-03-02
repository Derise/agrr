import os
import logging
import sys

from model import train_binary, train_seq_labeling, build_binary, build_seq_labeling
from vocab_generation import FTVocabulary
from tokenizer import Tokenizer


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    from settings import AGRR_TRAIN_SET, AGRR_DEV_SET, BINARY_MODEL, SEQ_LABELING_MODEL

    ft_vocab = FTVocabulary()
    ft_vocab.extend_vocab(AGRR_TRAIN_SET)
    ft_vocab.extend_vocab(AGRR_DEV_SET)
    ft_vocab.release_memory()

    train_samples = []
    Tokenizer().load_samples(AGRR_TRAIN_SET, train_samples)
    val_samples = []
    Tokenizer().load_samples(AGRR_DEV_SET, val_samples)
    samples = (train_samples, val_samples)

    binary_model = build_binary(ft_vocab)
    if os.path.exists(BINARY_MODEL):
        print("Binary model already exists.")
    else:
        train_binary(samples=samples, model_path=BINARY_MODEL, model=binary_model, vocab=ft_vocab, batch_size=8,
                     epochs_num=1)

    train_samples = [sample for sample in train_samples if sample[2] == 1]
    val_samples = [sample for sample in val_samples if sample[2] == 1]
    samples = (train_samples, val_samples)

    seq_labeling_model = build_seq_labeling(ft_vocab)
    if os.path.exists(SEQ_LABELING_MODEL):
        print("Sequence labeling model already exists.")
    else:
        train_seq_labeling(samples=samples, model_path=SEQ_LABELING_MODEL, model=seq_labeling_model, vocab=ft_vocab,
                           batch_size=8, epochs_num=1)
