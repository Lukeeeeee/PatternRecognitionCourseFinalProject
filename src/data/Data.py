from PIL import Image
import os


class Data(object):
    def __init__(self, label_dir):
        f = open(file=label_dir)
        label = f.readlines()
        self.label = {}
        for line in label:
            nums = line.splitlines()
            region = [int(nums[i]) for i in range(1, 4)]
            self.label[nums[0]] = region

    @staticmethod
    def cut_image_by_grid(image_dir, save_dir, size=(30, 30), stride=10):

        im = Image.open(image_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for j in range(0, im.size[1], stride):
            for i in range(0, im.size[0], stride):
                region = (i, j, i + size[0], j + size[1])
                sub_image = im.crop(region)
                dir = save_dir + str(i) + '_' + str(j) + '.bmp', 'bmp'
                sub_image.save(dir)
        print("New grid image set" + "Saved at " + save_dir)

    @staticmethod
    def show_a_pic(dir):
        im = Image.open(dir)
        im.show()

    def measure_similarity_of_two_region(self, pic_id, test_region):
        label_region = self.label[str(pic_id)]

        intersect_region_x_min = max(min(label_region[0], label_region[2]), min(test_region[0], test_region[2]))
        intersect_region_x_max = min(max(label_region[1], label_region[3]), max(test_region[1], test_region[3]))

        intersect_region_y_min = max(min(label_region[0], label_region[2]), min(test_region[0], test_region[2]))
        intersect_region_y_max = min(max(label_region[1], label_region[3]), max(test_region[1], test_region[3]))

        if intersect_region_x_min < intersect_region_x_max and intersect_region_y_min < intersect_region_y_max:
            return (intersect_region_x_max - intersect_region_x_min) * \
                   (intersect_region_y_max - intersect_region_y_min)

        else:
            return 0


    def return_test_data(self):
        pass

    def return_batch_train_data(self, batch_size):
        pass








