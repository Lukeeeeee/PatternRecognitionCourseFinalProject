from src.data.Data import Data
from src.model.Model import Model
from dataset import DATA_PATH
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

    model.test(test_image_id=1)
    model.test(test_image_id=350)
    model.end()
    # model.debug()


def load_and_test(model_dir):
    a = Data()
    model = Model(data=a)
    model.load_model(log_model_dir=model_dir)
    # model.train()

    model.test(test_image_id=1)
    model.test(test_image_id=100)
    model.test(test_image_id=470)
    model.end()

if __name__ == '__main__':
    load_and_test(model_dir=LOG_PATH + '/6-14-16-5-3/model/model.ckpt-0')
    # train()
