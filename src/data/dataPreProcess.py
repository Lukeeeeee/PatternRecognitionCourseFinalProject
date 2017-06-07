from PIL import Image
import os

def read_image(dir):
    im = Image.open(dir)
    return im


def cut_image(im, save_dir, size=(30, 30), stride=10, ):
    x, y = [(im.size[i] - size[i]) / stride + 1 for i in range(2)]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for j in range(0, im.size[1], stride):
        for i in range(0, im.size[0], stride):
            region = (i, j, i + size[0], j + size[1])
            sub_image = im.crop(region)
            sub_image.save(save_dir + str(i) + '_' + str(j) + '.bmp', 'bmp')
    pass


if __name__ == '__main__':

    for i in xrange(1, 500):
        im = read_image(dir='../../data/nemo_dataset/' + str(i) + '.bmp')

        cut_image(im, save_dir='../../data/cut_dataset/' + str(i) + '/')
