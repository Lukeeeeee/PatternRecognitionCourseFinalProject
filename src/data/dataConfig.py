import data


class DataConfig(object):

    # TODO USE THIS ONE
    ORIGIN_LABEL_PATH = data.DATA_PATH + '/label.md'
    CUT_DATASET_PATH = data.DATA_PATH + '/cut_dataset_2/'
    NEMO_DATASET_PATH = data.DATA_PATH + '/nemo_dataset/'
    GENERATED_LABEL_PATH = data.DATA_PATH + '/label2/'

    SUB_REGION_X = 60
    SUB_REGION_Y = 60
    STRIDE = 10

    PIC_LENGTH = 320
    PIC_HEIGHT = 240

    SUB_SAMPLE_RATE = 0.3

    @staticmethod
    def save_to_dict(conf):
        return {
            'ORIGIN_LABEL_PATH': conf.ORIGIN_LABEL_PATH,
            'GENERATED_LABEL_PATH': conf.GENERATED_LABEL_PATH,
            'CUT_DATASET_PATH': conf.CUT_DATASET_PATH,
            'NEMO_DATASET_PATH': conf.NEMO_DATASET_PATH,
            'SUB_REGION_X': conf.SUB_REGION_X,
            'SUB_REGION_Y': conf.SUB_REGION_Y,
            'STRIDE': conf.STRIDE,
            'PIC_LENGTH': conf.PIC_HEIGHT,
            'PIC_HEIGHT': conf.PIC_HEIGHT,
            'SUB_SAMPLE_RATE': conf.SUB_SAMPLE_RATE
        }

