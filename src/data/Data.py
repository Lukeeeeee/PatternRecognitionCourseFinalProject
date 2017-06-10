from __future__ import print_function
from PIL import Image
import os
import numpy as np
from dataConfig import DataConfig as dataConf


class Data(object):
    def __init__(self, label_dir):
        f = open(label_dir, mode='r')
        label = f.readlines()
        self.label = {}
        for line in label:
            nums = line.split(' ')
            region = [int(nums[i]) for i in range(1, 5)]
            self.label[nums[0]] = region
        self.current_train_data_batch_start = 0
        self.train_data_x, self.train_data_label = self.load_train_data()
        # self.test_data = self.load_test_data()
        self.train_data_batch_x = self.generate_batch_data(data=self.train_data_x, batch_size=20)
        self.train_data_batch_label = self.generate_batch_data(data=self.train_data_label, batch_size=20)

    @staticmethod
    def cut_image_by_grid(image_dir, save_dir, size=(30, 30), stride=10):

        im = Image.open(image_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for j in range(0, im.size[1], stride):
            for i in range(0, im.size[0], stride):
                region = (i, j, i + size[0], j + size[1])
                sub_image = im.crop(region)
                dir = save_dir + str(i) + '_' + str(j) + '.bmp'
                sub_image.save(dir)
        print("New grid image set saved at " + save_dir)

    @staticmethod
    def show_a_pic(dir):
        im = Image.open(dir)
        im.show()

    @staticmethod
    def measure_similarity_of_two_region(label_region, test_region):

        intersect_region_x_min = max(min(label_region[0], label_region[2]), min(test_region[0], test_region[2]))
        intersect_region_x_max = min(max(label_region[1], label_region[3]), max(test_region[1], test_region[3]))

        intersect_region_y_min = max(min(label_region[0], label_region[2]), min(test_region[0], test_region[2]))
        intersect_region_y_max = min(max(label_region[1], label_region[3]), max(test_region[1], test_region[3]))

        if intersect_region_x_min < intersect_region_x_max and intersect_region_y_min < intersect_region_y_max:
            return (intersect_region_x_max - intersect_region_x_min) * (intersect_region_y_max - intersect_region_y_min)\
                   * 1.0 / ((test_region[3] - test_region[1]) * (test_region[2] - test_region[0]))

        else:
            return 0

    def calc_similarity_of_labeled_data(self):
        # only used once!

        for key, value in self.label.iteritems():
            
            label_dir = '../../data/label/' + key + '/'

            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            file_dir = label_dir + str(key) + ".md"
            with open(file_dir, "w") as f:
                for i in range(0, dataConf.PIC_LENGTH, 10):
                    for j in range(0, dataConf.PIC_HEIGHT, 10):
                        test_region = (i, j, i+dataConf.SUB_REGION_X, j+dataConf.SUB_REGION_Y)
                        results = self.measure_similarity_of_two_region(label_region=value, test_region=test_region)
                        print("%d %d %d %d %.5lf" %
                              (test_region[0], test_region[1], test_region[2], test_region[3], results), file=f)

    @staticmethod
    def load_cut_image(image_id, label=False):
        # [0..310][0..230]
        load_dir = '../../data/cut_dataset/' + str(image_id) + '/'
        image_data_x = []
        image_data_label = []

        label_data = {}
        if label is True:
            with open('../../data/label/' + str(image_id) + '/' + str(image_id) + '.md') as f:
                lines = f.readlines()
                for line in lines:
                    if len(line) > 1:
                        elements = line.split(' ')
                        index = str(elements[0]) + str(elements[1])
                        # label_data[index] = elements[4]
                        label_data[index] = (1.0 - float(elements[4]), float(elements[4]))

        for i in range(0, 310, 10):
            for j in range(0, 230, 10):
                file = str(i) + "_" + str(j) + ".bmp"
                data = np.asanyarray(Image.open(load_dir + file))
                index = str(i) + str(j)

                image_data_x.append(data)
                image_data_label.append(label_data[index])

        image_data_x = np.array(image_data_x)
        image_data_label = np.array(image_data_label)

        if label:
            return image_data_x, image_data_label
        else:
            return image_data_x

    def load_test_data(self):
        test_data = []
        for i in range(1, 500):
            if str(i) not in self.label:
                new_data = self.load_cut_image(image_id=i)
                test_data.append(new_data)

        test_data = np.array(test_data)

        test_data = np.reshape(test_data, (-1, 30, 30, 3))
        return test_data

    def load_train_data(self):

        train_data_x = []
        train_data_label = []

        for i in range(1, 500):
            if str(i) in self.label:
                new_data_x, new_data_label = self.load_cut_image(image_id=i, label=True)
                train_data_x.append(new_data_x)
                train_data_label.append(new_data_label)
        train_data_x = np.array(train_data_x)
        train_data_x = np.reshape(train_data_x, (-1, 30, 30, 3))

        train_data_label = np.array(train_data_label)
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


if __name__ == '__main__':

    a = Data(label_dir="../../data/label.md")


