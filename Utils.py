'''
    Author:
        Jon Perry
'''
import os
import glob
import itertools
from PIL import Image
import tensorflow as tf
import torch
import cv2
import numpy as np
import labels
import pickle


class Utils:
    '''
    '''
    def __init__(self):
        '''
        '''
        self.save_weights_path = "CityScape_saves/"
        self.train_images = "train_paths.txt"
        self.train_gt = "train_paths.txt"
        self.val_images = "val_paths.txt"
        self.val_gt = "val_paths.txt"
        self.input_height = 512
        self.input_width = 512
        self.input_ch = 3
        self.epochs = 100
        self.batch_size = 32
        self.n_classes = 19
        self.steps_per_epoch = 90
        self.validation_steps = 15


class CityScape(Utils):
    '''
    '''
    def __init__(self):
        '''
        '''
        super().__init__()
        assert self.train_images[-4:] == '.txt'
        assert self.train_gt[-4:] == '.txt'
        assert self.val_gt[-4:] == '.txt'
        assert self.val_images[-4:] == '.txt'
        self.label_colors = pickle.load(open('label_colors.pkl', 'rb'))
        self.train_imgs = [line.rstrip().split()[0] for line in open(self.train_images, 'r').readlines()] 
        self.validation_imgs = [line.rstrip().split()[0] for line in open(self.val_images, 'r').readlines()] 
        self.train_labels =  [line.rstrip().split()[1] for line in open(self.train_gt, 'r').readlines()] 
        self.validation_labels = [line.rstrip().split()[1] for line in open(self.val_gt, 'r').readlines()] 
        assert len(self.train_imgs) == len(self.train_labels) and len(self.validation_imgs) == len(self.validation_labels)
        #self.train_imgs = self.getImages(self.train_imgs)
        #self.train_labels = self.getLabels(self.train_labels)
        #self.validation_imgs = self.getImages(self.validation_imgs)
        #self.validation_labels = self.getLabels(self.validation_labels)
        

    def create_dataset(self):
        '''
        '''
        dataset_train = Dataset(self.train_imgs, self.train_labels, self.batch_size)
        dataset_validate = Dataset(self.validation_imgs, self.validation_labels, self.batch_size)
        return dataset_train, dataset_validate
    
    def getImages(self, imgs):
        '''
        '''
        _imgs = list()
        for im in imgs:
            try:
                #img = Image.open(im)
                #img = img.resize((self.input_width, self.input_height))
                #img = np.array(img)
                img = cv2.imread(im, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.input_width, self.input_height))
            except Exception as e:
                print(im, e)
                img = np.zeros((self.input_height, self.input_width, self.input_ch))
            _imgs.append(img)
        return _imgs

    def getLabels(self, labels):
        '''
        '''
        _labels = list()
        for label in labels:
            np_label = np.zeros((self.input_height, self.input_width, self.n_classes))
            try:
                img = Image.open(label)
                img = img.resize((self.input_width, self.input_height))
                img = np.array(img)
                img = cv2.imread(label, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.input_width, self.input_height))
                for c in range(self.n_classes):
                    np_label[:, :, c] = (img == self.label_colors[c])[:,:,0].astype(int)
            except Exception as e:
                print(e)
            np_label = np.reshape(np_label, (self.input_width * self.input_height, self.n_classes))
            _labels.append(np_label)
        return _labels  

class OpenEDS(Utils):
    '''
    '''
    def __init__(self):
        '''
        '''
        super().__init__()
        assert self.train_images[-1] == '/'
        assert self.train_gt[-1] == '/'
        assert self.val_gt[-1] == '/'
        assert self.val_images[-1] == '/'
        self.train_imgs = glob.glob(self.train_images + "*.png")
        self.train_labels = glob.glob(self.train_gt + "*.png")
        self.validation_imgs = glob.glob(self.val_images + "*.png")
        self.validation_labels = glob.glob(self.val_gt + "*.png")
        assert len(self.train_imgs) == len(self.train_labels) and len(self.validation_imgs) == len(self.validation_labels)
        self.train_imgs = self.getImages(self.train_imgs)
        self.train_labels = self.getLabels(self.train_labels)
        self.validation_imgs = self.getImages(self.validation_imgs)
        self.validation_labels = self.getLabels(self.validation_labels)
    
    def create_dataset(self):
        '''
        '''
        dataset_train = Dataset(self.train_imgs, self.train_labels, self.batch_size)
        dataset_validate = Dataset(self.validation_imgs, self.validation_labels, self.batch_size)
        return dataset_train, dataset_validate

    def getImages(self, imgs):
        '''
        '''
        _imgs = list()
        for im in imgs:
            try:
                img = cv2.imread(im, cv2.IMREAD_COLOR)
            except Exception as e:
                print(im, e)
                img = np.zeros((self.input_height, self.input_width, self.input_ch))
            _imgs.append(img)
        return _imgs

    def getLabels(self, labels):
        '''
        '''
        _labels = list()
        for label in labels:
            np_label = np.zeros((self.input_height, self.input_width, self.n_classes))
            try:
                img = cv2.imread(label, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.input_width, self.input_height))
                for c in range(self.n_classes):
                    np_label[:, :, c] = (img == c).astype(int)
            except Exception as e:
                print(e)
            np_label = np.reshape(np_label, (self.input_width * self.input_height, self.n_classes))
            _labels.append(np_label)
        return _labels  

class Dataset:
    '''
    '''
    def __init__(self, x, y, batch_size):
        '''
        '''
        self.len = len(x)
        self.data = itertools.cycle(zip(x, y))
        self.batch_size = batch_size

    def __len__(self):
        '''
        '''
        return self.len


class Generator:
    '''
    '''
    def __init__(self, dataset, batch_size):
        '''
        '''
        self.dataset = dataset
        self.batch_size = batch_size
    
    def generate(self, util):
        '''
        '''
        while True:
            imgs = list()
            labels = list()
            for _ in range(self.batch_size):
                im, gt = next(self.dataset.data)
                img = cv2.imread(im, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (util.input_width, util.input_height), interpolation=cv2.INTER_AREA)
                img_label = cv2.imread(gt, cv2.IMREAD_COLOR)
                img_label = cv2.resize(img_label, (util.input_width, util.input_height), interpolation=cv2.INTER_AREA)
                np_label = np.zeros((util.input_height, util.input_width, util.n_classes))
                for c in range(util.n_classes):
                    np_label[:, :, c] = (img_label == util.label_colors[c])[:,:,0].astype(int)
                np_label = np.reshape(np_label, (util.input_width * util.input_height, util.n_classes))
                imgs.append(img)
                labels.append(np_label)
            yield np.array(imgs), np.array(labels)
        
    def __len__(self):
        '''
        '''
        return int( np.ceil(len(self.dataset.data) / float(self.batch_size)) )



