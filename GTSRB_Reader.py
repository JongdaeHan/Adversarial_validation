# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import torch
from torchvision.transforms.transforms import Resize
from adversarial_examples.Mixed_Dataset import Mixed_Dataset
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import csv

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
class GTSRB_Reader():
    def __init__(self):
        pass

    def readTrafficSigns(self, rootpath):
        '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''
        images = [] # images
        labels = [] # corresponding labels
        # loop over all 42 classes
        for c in range(0,43):
            prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
            gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader)
            # loop over all images in current annotations file
            for row in gtReader:
                # images.append(self.tf(plt.imread(prefix + row[0]))) # the 1th column is the filename
                images.append(prefix + row[0])
                labels.append(int(row[7])) # the 8th column is the label
            gtFile.close()
        return images, labels

    def readTestTrafficSigns(self, rootpath):
        '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''
        images = [] # images
        labels = [] # corresponding labels
        # loop over all 42 classes
        gtFile = open(rootpath + '/GT-final_test.csv')
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader) # skip header
        
        for row in gtReader:
            images.append(rootpath + row[0])
            labels.append(int(row[7]))
        gtFile.close()

        return images, labels

if __name__ == '__main__':

    g = GTSRB_Reader()

    clean_train_data_examples, clean_train_label_examples = g.readTrafficSigns('./datasets/GTSRB/training/Images')
    # print(clean_train_label_examples)
    # print(clean_train_data_examples)
    dataset = Mixed_Dataset(clean_train_data_examples, clean_train_label_examples)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

    clean_train_data_examples, clean_train_label_examples = [], []

    for (data, label) in data_loader:
        clean_train_data_examples.append(data[0]) # 배치 정보를 제거
        clean_train_label_examples.append(label[0]) # 배치 정보를 제거

    mixed_train_dataset = Mixed_Dataset(clean_train_data_examples, clean_train_label_examples)
    mixed_train_loader = torch.utils.data.DataLoader(mixed_train_dataset, batch_size=64, shuffle=True)

    for (data, label) in mixed_train_loader:
        print(label)