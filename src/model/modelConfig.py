import math


class ModelConfig(object):

    INPUT_X_SIZE = 60
    INPUT_Y_SIZE = 60
    INPUT_SIZE = INPUT_X_SIZE * INPUT_Y_SIZE
    INPUT_CHANNEL = 3

    OUTPUT_SIZE = 2

    LAYER1_CHANNEL = 6

    LAYER2_CHANNEL = 18

    LAYER3_CHANNEL = 36

    LAYER4_CHANNEL = 72

    FILTER_SIZE = 5

    CONVOLUTIONAL_STRIDES = 1

    MAX_POOLING_STRIDES_1 = 2
    MAX_POOLING_STRIDES_2 = 2
    MAX_POOLING_STRIDES_3 = 2
    MAX_POOLING_STRIDES_4 = 2

    CONVOLUTIONAL_LAYER_SIZE = 4

    # FULLY_CONNECTED_INPUT_SIZE = int(INPUT_SIZE / (math.pow(MAX_POOLING_STRIDES, CONVOLUTIONAL_LAYER_SIZE)))
    FULLY_CONNECTED_INPUT_SIZE = 4 * 4 * 72

    FULLY_CONNECTED_OUT_SIZE = INPUT_SIZE

    OUT_LAYER_INPUT_SIZE = FULLY_CONNECTED_OUT_SIZE
    OUT_LAYER_OUT_SIZE = OUTPUT_SIZE

    DROPOUT_PROBABILITY = 0.01

    LEARNING_RATE = 0.03
    BATCH_SIZE = 400
    EPOCH = 1

    L2 = 0.03

    @staticmethod
    def save_to_dict(model):
        return {
            "INPUT_X_SIZE": model.INPUT_X_SIZE,
            "INPUT_Y_SIZE": model.INPUT_Y_SIZE,
            "INPUT_SIZE": model.INPUT_SIZE,
            "INPUT_CHANNEL": model.INPUT_CHANNEL,
            "OUTPUT_SIZE": model.OUTPUT_SIZE,
            "LAYER1_CHANNEL": model.LAYER1_CHANNEL,
            "LAYER2_CHANNEL": model.LAYER2_CHANNEL,
            "LAYER3_CHANNEL": model.LAYER3_CHANNEL,
            "LAYER4_CHANNEL": model.LAYER4_CHANNEL,
            "FILTER_SIZE": model.FILTER_SIZE,
            "CONVOLUTIONAL_STRIDES": model.CONVOLUTIONAL_STRIDES,
            "MAX_POOLING_STRIDES_1": model.MAX_POOLING_STRIDES_1,
            "MAX_POOLING_STRIDES_2": model.MAX_POOLING_STRIDES_2,
            "MAX_POOLING_STRIDES_3": model.MAX_POOLING_STRIDES_3,
            "FULLY_CONNECTED_INPUT_SIZE": model.FULLY_CONNECTED_INPUT_SIZE,
            "FULLY_CONNECTED_OUT_SIZE": model.FULLY_CONNECTED_OUT_SIZE,
            "OUT_LAYER_INPUT_SIZE": model.OUT_LAYER_INPUT_SIZE,
            "OUT_LAYER_OUT_SIZE": model.OUT_LAYER_OUT_SIZE,
            "DROPOUT_PROBABILITY": model.DROPOUT_PROBABILITY,
            "LEARNING_RATE": model.LEARNING_RATE,
            "BATCH_SIZE": model.BATCH_SIZE,
            "EPOCH": model.EPOCH,
            "L2": model.L2
        }
