from src.data.Data import Data
from src.model.Model import Model


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
    model.test(test_image_id=350)
    model.end()

if __name__ == '__main__':
    load_dir = '/home/mars/ANN/dls/PatternRecognitionCourseFinalProject/log/6-12-16-46-53/model/model.ckpt-10000'
    # train()
