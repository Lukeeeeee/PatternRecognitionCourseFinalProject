from src.data.Data import Data
from src.data.dataConfig import DataConfig
from src.model.Model import Model
from src.model.modelConfig import ModelConfig
import datetime
import os


def log_config():
    ti = datetime.datetime.now()
    log_dir = ('../../log/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-'
               + str(ti.minute) + '-' + str(ti.second))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    self.config_log_file = open(log_dir + 'config.txt', "a")

def main()




if __name__ == '__main__':
    main()