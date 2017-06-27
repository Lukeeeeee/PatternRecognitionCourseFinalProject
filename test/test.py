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

    model.test(test_image_id=1)
    model.test(test_image_id=350)
    model.end()
    # model.debug()


def load_and_test(model_dir):
    a = Data()
    model = Model(data=a)
    model.load_model(log_model_dir=model_dir)
    # model.train()

    for i in range(1, 500):
        model.test(i, save_dir=model.test_image_save_dir)
    # model.test(test_image_id=1)
    # model.test(test_image_id=100)
    # model.test(test_image_id=470)
    model.end()

if __name__ == '__main__':
    load_and_test(model_dir=LOG_PATH + '/../log/6-14-19-4-12/model/model.ckpt-500')
    # train()
    #
    # ffmpeg -framerate 25 -i %d.bmp -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4