from src.data.Data import Data
from src.data.dataConfig import DataConfig
from src.model.Model import Model
from src.model.modelConfig import ModelConfig
import datetime
import os


def train():

    a = Data(label_dir="../data/label.md")
    model = Model(data=a)
    model.log_config()

    model.train()
    model.save_model()
    # model.load_model(log_model_dir='/home/mars/ANN/dls/PatternRecognitionCourseFinalProject/log/6-11-17-21-16/model/model.ckpt-50000')
    # model.train()

    model.test(test_image_id=1)
    model.test(test_image_id=350)
    model.end()
    # model.debug()

if __name__ == '__main__':
    train()
