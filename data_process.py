import struct
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from imgaug import augmenters as iaa

def load_mnist(kind='train'):
    with open('data/mnist/%s-labels.idx1-ubyte' % kind, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype = np.uint8)
    with open('data/mnist/%s-images.idx3-ubyte' % kind, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)
    return images, labels

def generate_equation(digits, digits_labels, symbols, symbols_labels):
    
    def _generate_multi_mnist(imgs, labels):
        num_digits = np.random.randint(1, 8) # number of digits
        label = []
        next_index = np.random.randint(imgs.shape[0])
        img = imgs[next_index]
        label.append(labels[next_index])
        for _ in range(num_digits-1):
            gap = np.random.randint(1,12) # the gap between digits when concatenating
            next_index = np.random.randint(img.shape[0])
            img = np.concatenate((img[:, :-gap], img[:, -gap:] + imgs[next_index][:, :gap], imgs[next_index][:, gap:]), axis=1)
            label.append(labels[next_index])
        return img, label

    # first number
    img_1, label_1 = _generate_multi_mnist(digits, digits_labels)
    # second number
    img_2, label_2 = _generate_multi_mnist(digits, digits_labels)
    # symbol
    real_symbols = ['+', '-', '*', '/']
    label = label_1
    next_index = np.random.randint(len(symbols))
    gap = np.random.randint(1, 12)
    img = np.concatenate((img_1[:, :-gap], img_1[:, -gap:] + symbols[next_index][:, :gap], symbols[next_index][:, gap:]), axis=1)
    label += [real_symbols[symbols_labels[next_index][0]]]
    gap = np.random.randint(1, 12)
    img = np.concatenate((img[:, :-gap], img[:, -gap:] + img_2[:, :gap], img_2[:, gap:]), axis=1)
    label += label_2

    return img, label

def _prepro(o,image_size=[28,28]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
    Default return: np.array 
        Grayscale image, shape: (28, 28)
    
    """
    y = 0.2126 * o[:, :, :, 0] + 0.7152 * o[:, :, :, 1] + 0.0722 * o[:, :, :, 2]
    resized = np.array([scipy.misc.imresize(img, image_size) for img in y])
    resized = (255-resized).astype(np.uint8)
    return resized
    #return np.expand_dims(resized.astype(np.float32),axis=2)

def generate_symbol_dataset():
    """

    """
    images = []
    labels = [] # 0: +, 1: -, 2: *, 3: /

    # data augmenter
    seq = iaa.Sequential([
        #iaa.Crop(px=(0, 4)), # crop images from each side by 0 to 4px (randomly chosen)
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.Flipud(0.5), 
        iaa.GaussianBlur(sigma=(0, 0.5)), # blur images with a sigma of 0 to 0.5
        iaa.PerspectiveTransform(scale=(0.05, 0.1)) # advanced crop
    ])

    symbols = ['+', '-', 'x', 'd']
    for j in range(len(symbols)):
        # read origin symbol handwriting (10 imgs per symbol)
        imgs = []
        for i in range(1, 11):
            imgs.append(plt.imread('data/symbol/%s_%d.png'%(symbols[j], i)))
        # data augment part
        imgs = _prepro(np.array(imgs * 250))
        images_aug = seq.augment_images(imgs)
        # gather data part
        images.append(images_aug)
        labels.append([[j]] * 2500)

    images = np.vstack(images)
    labels = np.vstack(labels)

    # shuffle part
    shuffle_index = np.random.choice(len(images), size=len(images), replace=False)
    images = images[shuffle_index]
    labels = labels[shuffle_index]

    return images, labels

def shuffle_dataset(images, labels):
    """
    shuffl the dataset when a epoch is in the end
    """
    shuffle_index = np.random.choice(len(images), size=len(images), replace=False)
    images = images[shuffle_index]
    labels = labels[shuffle_index]

    return images, labels

def read_dataset():
    """
    read the splited images and labels
    """
    
    labels = pickle.load(open('labels.p', 'rb'))
    dic = {'+': 10, '-': 11, '*': 12, '/': 13}
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            labels[i][j] = dic[labels[i][j]] if labels[i][j] in dic.keys() else labels[i][j]
    
    imgs = []
    labels_ = []
    for i in range(10000):
        for root, dirs, files in os.walk('data/split/%d'%(i)):
            # throw the data which is divided unclearly
            if len(labels[i]) != len(files):
                continue
            files = sorted(files, key=lambda x:int(x.split('.')[0]))
            for j in range(len(files)):
                imgs.append(plt.imread('data/split/%d/'%(i)+files[j]))
                labels_.append(labels[i][j])
    # one-hot
    #labels_ = np.hstack(labels_)
    n_values = np.max(labels_) + 1
    labels = np.eye(n_values)[labels_]
    
    imgs = np.array(imgs)

    return imgs[:25000], labels[:25000], imgs[25000:], labels[25000:]

def read_sample(filepath):
    """
    read sample
    """
    imgs = []
    for root, dirs, files in os.walk(filepath):
        files = sorted(files, key=lambda x:int(x.split('.')[0]))
        for file in files:
            imgs.append(plt.imread(filepath+file))
    imgs = np.array(imgs)
    return imgs

def main():
    # make equations and labels for dividing
    digits, digits_labels = load_mnist()
    digits = digits.reshape(-1, 28, 28)
    symbols, symbols_labels = generate_symbol_dataset()
    labels = []
    for i in range(10000):
        print(i)
        img, label = generate_equation(digits, digits_labels, symbols, symbols_labels)
        plt.imsave('data/equations/%d.png'%(i), img)
        labels.append(label)
    pickle.dump(labels, open('labels.p', 'wb'))

if __name__ == '__main__':
    main()