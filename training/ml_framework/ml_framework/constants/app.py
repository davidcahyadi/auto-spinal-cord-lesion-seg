
class Phase:
    TRAIN = 'train'
    VALIDATE = 'val'
    TEST = 'test'
    PREDICT = 'predict'


class Task:
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    SEGMENTATION = 'segmentation'


class MetadataHeader:
    LABEL = 'label'
    PHASE = 'phase'
    INPUT_PATH = 'input_path'
    DATASET = 'dataset'


class ProcessTarget:
    DATA = 'data'
    LABEL = 'label'
    PREDICTION = 'prediction'
