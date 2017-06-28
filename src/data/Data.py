from __future__ import print_function
from PIL import Image
from PIL import ImageDraw
import os
import numpy as np
from dataConfig import DataConfig as dataConf
import random
import dataset
import log


class Data(object):
    def __init__(self):
        self.origin_label_dir = dataConf.ORIGIN_LABEL_PATH
        self.cut_dataset_dir = dataConf.CUT_DATASET_PATH
        self.nemo_dataset_dir = dataConf.NEMO_DATASET_PATH
        self.generated_label_dir = dataConf.GENERATED_LABEL_PATH

        f = open(self.origin_label_dir, mode='r')
        label = f.readlines()
        self.label = {}
        for line in label:
            nums = line.split(' ')
            region = [int(nums[i]) for i in range(1, 5)]
            self.label[nums[0]] = region
        self.current_train_data_batch_start = 0
        self.train_data_x, self.train_data_label = self.load_train_data()
        self.train_data_batch_x = self.generate_batch_data(data=self.train_data_x, batch_size=100)
        self.train_data_batch_label = self.generate_batch_data(data=self.train_data_label, batch_size=100)

    @staticmethod
    def cut_image_by_grid(image_dir, save_dir, size=(60, 60), stride=10):

        im = Image.open(image_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for j in range(0, im.size[1], stride):
            for i in range(0, im.size[0], stride):
                if i+size[0] > im.width or j + size[1] > im.height:
                    break
                region = (i, j, i + size[0], j + size[1])
                sub_image = im.crop(region)
                dir = save_dir + str(i) + '_' + str(j) + '.bmp'
                sub_image.save(dir)
        print("New grid image set saved at " + save_dir)

    @staticmethod
    def measure_similarity_of_two_region(label_region, test_region):

        intersect_region_x_min = max(min(label_region[0], label_region[2]), min(test_region[0], test_region[2]))
        intersect_region_x_max = min(max(label_region[0], label_region[2]), max(test_region[0], test_region[2]))

        intersect_region_y_min = max(min(label_region[1], label_region[3]), min(test_region[1], test_region[3]))
        intersect_region_y_max = min(max(label_region[1], label_region[3]), max(test_region[1], test_region[3]))

        if intersect_region_x_min < intersect_region_x_max and intersect_region_y_min < intersect_region_y_max:
            value = (intersect_region_x_max - intersect_region_x_min) * (intersect_region_y_max - intersect_region_y_min)\
                    * 1.0 / ((test_region[3] - test_region[1]) * (test_region[2] - test_region[0]))
            return value

        else:
            return 0

    def calc_similarity_of_labeled_data(self):
        # only used once!

        for key, value in self.label.iteritems():

            file_dir = self.generated_label_dir + str(key) + '/'
            file_name = file_dir + str(key) + ".md"

            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            with open(file_name, "w") as f:
                for i in range(0, dataConf.PIC_LENGTH, 10):
                    for j in range(0, dataConf.PIC_HEIGHT, 10):
                        test_region = (i, j, i+dataConf.SUB_REGION_X, j+dataConf.SUB_REGION_Y)
                        results = self.measure_similarity_of_two_region(label_region=value, test_region=test_region)
                        print("%d %d %d %d %.5lf" %
                              (test_region[0], test_region[1], test_region[2], test_region[3], results), file=f)

    def load_cut_image(self, image_id, label=False):
        # [0..310][0..230]
        load_dir = self.cut_dataset_dir + str(image_id) + '/'
        image_data_x = []
        image_data_label = []

        label_data = {}
        if label is True:
            with open(self.generated_label_dir + str(image_id) + '/' + str(image_id) + '.md') as f:
                lines = f.readlines()
                for line in lines:
                    if len(line) > 1:
                        elements = line.split(' ')
                        index = str(elements[0]) + str(elements[1])
                        # label_data[index] = elements[4]

                        label_data[index] = (1.0 - float(elements[4]), float(elements[4]))

        for i in range(0, 270, 10):
            for j in range(0, 190, 10):
                file = str(i) + "_" + str(j) + ".bmp"
                data = np.asanyarray(Image.open(load_dir + file))
                index = str(i) + str(j)
                if label:
                    if label_data[index][0] >= 0.99999:
                        prob = random.uniform(0, 1)
                        if prob < dataConf.SUB_SAMPLE_RATE:
                            image_data_x.append(data)
                            image_data_label.append(label_data[index])
                    else:
                        image_data_x.append(data)
                        image_data_label.append(label_data[index])

                else:
                    image_data_x.append(data)

        image_data_x = np.array(image_data_x)
        image_data_label = np.array(image_data_label)

        if label:
            return image_data_x, image_data_label
        else:
            return image_data_x

    def load_test_data(self, test_image_id):
        test_data = []
        new_data = self.load_cut_image(image_id=test_image_id)
        test_data.append(new_data)

        test_data = np.array(test_data)

        test_data = np.reshape(test_data, (-1, dataConf.SUB_REGION_X, dataConf.SUB_REGION_Y, 3))

        return test_data

    def load_train_data(self):

        train_data_x, train_data_label = self.load_cut_image(image_id=1, label=True)

        for i in range(2, 501):
            if str(i) in self.label:
                new_data_x, new_data_label = self.load_cut_image(image_id=i, label=True)

                # train_data_x.append(new_data_x)
                # train_data_label.append(new_data_label)
                train_data_x = np.concatenate([train_data_x, new_data_x])
                train_data_label = np.concatenate([train_data_label, new_data_label])

        # train_data_x = np.array(train_data_x)
        train_data_x = np.reshape(train_data_x, (-1, dataConf.SUB_REGION_X, dataConf.SUB_REGION_Y, 3))

        # train_data_label = np.array(train_data_label)
        train_data_label = np.reshape(train_data_label, (-1, 2))

        return train_data_x, train_data_label

    @staticmethod
    def generate_batch_data(batch_size, data):
        size = data.shape[0]
        batch_count = size / batch_size
        new_data = [data[i*batch_size: (i+1)*batch_size] for i in range(batch_count)]
        return new_data

    def return_one_batch_data(self):
        t = 0
        for x, label in zip(self.train_data_batch_x, self.train_data_batch_label):
            yield x, label
            t += 1

    def draw_new_label(self, image_id, region_list, save_dir=None):
        im = Image.open(self.nemo_dataset_dir + str(image_id) + '.bmp')
        drawer = ImageDraw.Draw(im=im)

        for region in region_list:
            x_min = region[0]
            y_min = region[1]
            x_max = region[2]
            y_max = region[3]
            drawer.polygon(xy=[(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
        if save_dir:
            im.save(save_dir + str(image_id) + '.bmp')
            print("New image " + save_dir + str(image_id) + '.bmp ' + "saved.")
        else:
            im.show()
        im.close()
        pass

    def draw_new_image(self, data):
        im_0 = Image.open(self.nemo_dataset_dir + "1" + '.bmp')

        im = Image.new(mode=im_0.mode, size=(30, 30))
        im.frombytes(data=data)
        im.show()
        pass


if __name__ == '__main__':

    # for i in xrange(1, 500):
    #     Data.cut_image_by_grid(image_dir=dataConf.NEMO_DATASET_PATH + str(i) + '.bmp', save_dir=dataConf.CUT_DATASET_PATH + '/' + str(i) + '/')

    a = Data()
    # a.calc_similarity_of_labeled_data()

    #load_dir = '../../data/cut_dataset/' + '1' + '/'
    # data = np.asanyarray(Image.open(load_dir + '130_70.bmp'))
    # a.draw_new_image(data=data)


