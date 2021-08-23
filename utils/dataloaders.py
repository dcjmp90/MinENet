from collections import namedtuple
import tensorflow as tf
from PIL import Image
import numpy as np


class TensorLoader:
    """this will be a helper class object 'Loader' which will do the actual
    processing of the file IO
 
    Notes: You will not actually use this class object directly
 
    Argurments:
 
        utils: this is the named dict of all the args parser information
    """

    def __init__(self, 
                 utils,
                 labels,
                 test,
                 ):
        """
            This constructor will be loading in the pickled label data for binary/onehot masking
        """
        self.test = test
        self.utils = utils
        self.labels = labels

    def __call__(self, 
                 img, 
                 gt,
                 ):
        """The __call__ method will be handling any direct call to this class object
        this will take in two paths: one to the image, and one to the ground truth.
        """
        im = Image.open(self.utils.ROOT_FOLDER+img.numpy().decode('utf-8')).convert("L")
        im = im.resize((self.utils.INPUT_SHAPE[1], self.utils.INPUT_SHAPE[0]))
        im = np.expand_dims(im, axis=-1)
        gt = np.load(self.utils.ROOT_FOLDER+gt.numpy().decode('utf-8'))
        gt = np.array(gt, dtype=np.uint8)
        mask = np.zeros((gt.shape[0], gt.shape[1], self.utils.NUM_CLASSES), dtype=np.uint8)
        tmp = gt.copy()
        for k in range(len(self.labels)):
            mask[:,:,k] = (tmp == k).astype(np.uint8)
        im = tf.convert_to_tensor(im, dtype=tf.float32)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32) 
        name = img.numpy().decode('utf-8').split('/')[-1].split('.')[0]
        if self.test:
            return im, mask, name
        else:
            return im, mask

class TFData:
    """the dataset object will process text line data mapped as text file x  and text file y or data sample file and label text file
 
    needs to be in string format data that maps to an image and a label that is in the format of a numpy pickled object
 
    Notes: Please see the if condition for running this as main for an example of how to run this module
    
    To Use:
 
        >>> from lib.dataloader import Dataset
        >>> dataset = Dataset(x = 'relative/path/to/train_data_input.txt', y = 'relative/path/to/train_labels.txt', btach_size = 32, utils = kargs)
        >>> dataset = dataset.prepare()
        >>> for batch in dataset:
        ...     print(batch) # will be a tuple (number of sets per iteration, train image, train label) 
    """
    def __init__(self, 
                 x, 
                 y, 
                 utils,
                 labels,
                 test=False,
                ):
        """

        Argurments:
 
        x: this will be the text file path/to/file_name.txt that will be a path per line/sample
        y: this will be the text file path/to/file_name.txt that will be a path per line/label
        batch_size: this will be an intger to represent how much the training batch size will be
            or how many samples per iteration of training, must be an integer
        """
        self.x = tf.data.TextLineDataset(x)
        self.y = tf.data.TextLineDataset(y)
        self.dataset = tf.data.Dataset.zip((self.x, self.y))
        self.lenx = len(open(x,'r').readlines())
        self.leny = len(open(y,'r').readlines())
        assert self.lenx == self.leny
        self.len = self.lenx
        self.batch_size = utils.BATCH_SIZE
        self.utils = utils
        self.test = test
        self.loader = TensorLoader(self.utils, labels, test)
    
    def prepare(self) -> tf.data.Dataset:
        """ the prepare method will be convert the TextLineDataset to a batched Tensor data output
 
        Notes: This may need to be optimized by changing the lambda py_funciton call to being a tf function
 
        Arguments: None
        """
        if not self.test:
            self.dataset = self.dataset.map( lambda img, gt: tf.py_function(self.loader, [img,gt], (tf.float32, tf.float32)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            self.dataset = self.dataset.map( lambda img, gt: tf.py_function(self.loader, [img,gt], (tf.float32, tf.float32, tf.string)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.batch(self.utils.BATCH_SIZE, drop_remainder=True)
        if not self.test:
            self.dataset = self.dataset.shuffle(self.batch_size, reshuffle_each_iteration=False)
        return self.dataset
    
    def __len__(self):
        """
        """
        return self.len


if __name__ == '__main__':
    from command_line import command_line_args
    from labelList import openeds_labels

    args = command_line_args.parse_args()

    dataset = TFData(args.TRAIN_IMAGES, args.TRAIN_LABELS, args, openeds_labels)
    dataset = dataset.prepare()

    for elem in dataset.as_numpy_iterator():
        string = elem[-1]
        string = string[0]
        print(string.decode("utf-8"))
