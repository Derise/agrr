from typing import Tuple
import os

import numpy as np

from keras.layers import Input, Dense, TimeDistributed, Masking, Bidirectional, GRU
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from batch_generator import BatchGeneratorBinary, BatchGenerator
from vocab_generation import FTVocabulary
from settings import SEQ_LABELING_MODEL_FOLDER, BINARY_MODEL_FOLDER

from keras_transformer.transformer import TransformerBlock
from keras_transformer.position import TransformerCoordinateEmbedding


def build_binary(vocab):
    """
    Model description.
    """
    words = Input(shape=(None, vocab.input_vector_size), name='words')
    words_masked = Masking(mask_value=0.0, input_shape=(None, vocab.input_vector_size))(words)

    layer = Bidirectional(GRU(128, dropout=0.5, recurrent_dropout=0.2,
                              return_sequences=True, name='GRU_1'), merge_mode='ave')(words_masked)
    layer = Bidirectional(GRU(128, dropout=0.2, recurrent_dropout=0.2,
                              return_sequences=False, name='GRU_2'), merge_mode='ave')(layer)

    output = Dense(1, activation='sigmoid')(layer)

    model = Model(inputs=[words], outputs=[output])

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.000625, epsilon=1e-09),
                       metrics=['binary_accuracy'])
    print(model.summary())
    return model


def build_seq_labeling(vocab):
    """

    :return:
    """
    words = Input(shape=(300, vocab.input_vector_size), name='words')

    transformer_input = words
    transformer_depth = 5

    transformer_block = TransformerBlock(
        name='transformer',
        num_heads=6,
        residual_dropout=0.1,
        attention_dropout=0.1,
        use_masking=False)
    add_coordinate_embedding = TransformerCoordinateEmbedding(
        transformer_depth,
        name='coordinate_embedding')

    transformer_output = transformer_input
    for step in range(transformer_depth):
        transformer_output = transformer_block(
            add_coordinate_embedding(transformer_output, step=step))

    output = TimeDistributed(Dense(vocab.target_vector_size, activation='softmax'))(transformer_output)

    model = Model(inputs=[words], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.000625, epsilon=1e-09),
                       metrics=['categorical_accuracy'], sample_weight_mode='temporal')
    print(model.summary())
    return model


def train_binary(samples: Tuple, model_path: str, model: Model, vocab: FTVocabulary, batch_size: int,
                 random_seed: int = 34, epochs_num: int = 50) -> None:

    np.random.seed(random_seed)
    best_loss_score = np.inf
    for i in range(epochs_num):
        train_set_generator = BatchGeneratorBinary(vocab=vocab, samples=samples[0], batch_size=batch_size,
                                                   shuffle=True)
        val_set_generator = BatchGeneratorBinary(vocab=vocab, samples=samples[1], batch_size=batch_size,
                                                 shuffle=False)
        history = model.fit_generator(generator=train_set_generator, epochs=1 + i, verbose=2,
                                      class_weight={0: 1., 1: 2.},
                                      callbacks=[TensorBoard(
                                           log_dir=os.path.join(BINARY_MODEL_FOLDER, "logs"))],
                                      validation_data=val_set_generator, workers=1,
                                      use_multiprocessing=False, initial_epoch=i,
                                      max_queue_size=1).history
        if history.get('val_loss')[-1] < best_loss_score:
            best_loss_score = history.get('val_loss')[-1]
            model.save(filepath=model_path)
        elif batch_size < 256:
            batch_size *= 2
            print(batch_size)
            model.save(filepath=model_path)
        else:
            print('Max batch_size achieved, but no improvements during the last step.\nEarly stopping.')
            break


def train_seq_labeling(samples: Tuple, model_path: str, model: Model, vocab: FTVocabulary, batch_size: int,
                       random_seed: int = 34, epochs_num: int = 50) -> None:

    np.random.seed(random_seed)
    best_loss_score = np.inf
    for i in range(epochs_num):
        train_set_generator = BatchGenerator(vocab=vocab, samples=samples[0], batch_size=batch_size,
                                             shuffle=True, with_weights=True)
        val_set_generator = BatchGenerator(vocab=vocab, samples=samples[1], batch_size=batch_size,
                                           shuffle=False, with_weights=True)
        history = model.fit_generator(generator=train_set_generator, epochs=1 + i, verbose=2,
                                      callbacks=[TensorBoard(
                                           log_dir=os.path.join(SEQ_LABELING_MODEL_FOLDER, "logs"))],
                                      validation_data=val_set_generator, workers=1,
                                      use_multiprocessing=False, initial_epoch=i,
                                      max_queue_size=1).history
        if history.get('val_loss')[-1] < best_loss_score:
            best_loss_score = history.get('val_loss')[-1]
            model.save(filepath=model_path)
        elif batch_size < 64:
            batch_size *= 2
            print(batch_size)
            model.save(filepath=model_path)
        else:
            print('Max batch_size achieved, but no improvements during the last step.\nEarly stopping.')
            break
