import os
from pkg_resources import resource_filename

DATA_PATH = resource_filename(__name__, "organizers")
MODELS_FOLDER = resource_filename(__name__, "models")
BINARY_MODEL_FOLDER = os.path.join(MODELS_FOLDER, "binary_classification")
SEQ_LABELING_MODEL_FOLDER = os.path.join(MODELS_FOLDER, "multilabel_true")
EMBEDDINGS_PATH = "/usr/local/share/WordEmbeddings/fastText/Russian"
AGRR_TRAIN_SET = os.path.join(DATA_PATH, "test.csv")
AGRR_DEV_SET = os.path.join(DATA_PATH, "dev.csv")
AGRR_TEST_SET = os.path.join(DATA_PATH, "test.csv")
FAST_TEXT_MODEL = os.path.join(EMBEDDINGS_PATH, "cc.ru.300.bin")
BINARY_MODEL = os.path.join(BINARY_MODEL_FOLDER, "model.h5")
SEQ_LABELING_MODEL = os.path.join(SEQ_LABELING_MODEL_FOLDER, "model.h5")
