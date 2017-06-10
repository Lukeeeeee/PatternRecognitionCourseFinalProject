

class DataConfig(object):

    # TODO USE THIS ONE
    LABEL_PATH = ""
    CUT_DATASET_PATH = ""
    NEMO_DATASET_PATH = ""

    SUB_REGION_X = 30
    SUB_REGION_Y = 30
    STRIDE = 10

    PIC_LENGTH = 320
    PIC_HEIGHT = 240

    SUB_SAMPLE_RATE = 0.8

    @staticmethod
    def save_to_dict(conf):
        return {
            'LABEL_PATH': conf.LABEL_PATH,
            'CUT_DATASET_PATH': conf.CUT_DATASET_PATH,
            'NEMO_DATASET_PATH': conf.NEMO_DATASET_PATH,
            'SUB_REGION_X': conf.SUB_REGION_X,
            'SUB_REGION_Y': conf.SUB_REGION_Y,
            'STRIDE': conf.STRIDE,
            'PIC_LENGTH': conf.PIC_HEIGHT,
            'PIC_HEIGHT': conf.PIC_HEIGHT,
            'SUB_SAMPLE_RATE': conf.SUB_SAMPLE_RATE
        }

