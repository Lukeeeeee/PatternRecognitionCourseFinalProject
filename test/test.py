"""
TEST 1.0

Usage:
    test.py train
    test.py test model_id <model_id> 

"""

import os
import sys

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

sys.path.append(CURRENT_PATH)
sys.path.append(CURRENT_PATH + '/../')

from docopt import docopt
from src.data.Data import Data
from src.model.Model import Model
from log import LOG_PATH


def train():
    a = Data()
    model = Model(data=a)
    model.log_config()

    model.train()
    model.save_model()
    # model.load_model(
    #     log_model_dir='/home/mars/ANN/dls/PatternRecognitionCourseFinalProject/log/6-11-17-21-16/model/model.ckpt-50000')
    # model.train()

    model.test(test_image_id=50, save_dir=model.test_image_save_dir)
    model.test(test_image_id=350, save_dir=model.test_image_save_dir)
    model.end()
    # model.debug()


def load_and_test(model_dir):
    a = Data()
    model = Model(data=a)
    model.load_model(log_model_dir=model_dir)
    model.test(test_image_id=50, save_dir=model.test_image_save_dir)
    model.test(test_image_id=350, save_dir=model.test_image_save_dir)
    for i in range(1, 500):
        model.test(i, save_dir=model.test_image_save_dir)

    model.end()

if __name__ == '__main__':

    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

    sys.path.append(CURRENT_PATH)
    sys.path.append(CURRENT_PATH + '/../')

    arguments = docopt(__doc__)
    if arguments['train']:
        train()
    else:
        path = LOG_PATH + '/../' + arguments['<model_id>']

        # load_and_test(model_dir=LOG_PATH + '/../demo/6-14-19-4-12/model/model.ckpt-500')

        load_and_test(model_dir=path)

    # command used to save mp4
    # ffmpeg -framerate 25 -i %d.bmp -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p results.mp4