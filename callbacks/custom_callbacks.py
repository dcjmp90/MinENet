import tensorflow as tf
from tensorflow.keras.callbacks import Callback\
import numpy as np
import os
from PIL import Image

class validation_saves:
    """
    """
    def __init__(self, 
                 utils, 
                 labels, 
                 epoch, 
                 step):
        """
        """
        self.epoch = epoch
        self.utils = utils
        self.labels = labels
        self.step = step
        self.color_preds = None
        self.color_gt = None
    
    def get_color_mapping(self, gt, pred):
        """
        """
        self.color_preds = np.zeros((self.utils.INPUT_SHAPE[0], self.utils.INPUT_SHAPE[1], self.utils.INPUT_SHAPE[-1]), dtype=np.uint8)
        self.color_gt = np.zeros((self.utils.INPUT_SHAPE[0], self.utils.INPUT_SHAPE[1], self.utils.INPUT_SHAPE[-1]), dtype=np.uint8)
        for label in self.labels:
            if not label.ignoreInEval:
                binary_mask_2d = (pred == label.trainId).astype(np.uint8)
                self.color_preds[:,:,0] += binary_mask_2d * label.color[0]
                self.color_preds[:,:,1] += binary_mask_2d * label.color[1]
                self.color_preds[:,:,2] += binary_mask_2d * label.color[2]
                binary_mask_2d_gt = (gt == label.trainId).astype(np.uint8)
                self.color_gt[:,:,0] += binary_mask_2d_gt * label.color[0]
                self.color_gt[:,:,1] += binary_mask_2d_gt * label.color[1]
                self.color_gt[:,:,2] += binary_mask_2d_gt * label.color[ 2]

    def save_stack(self,
                   img,
                   ground_truth,
                   prediciton,
                   ):
            """
            """
            self.get_color_mapping(ground_truth, prediciton)
            stack = (np.array(img, np.uint8), self.color_gt, self.color_preds)
            horizontal_combined = np.hstack(stack)
            horizontal_combined = Image.fromarray(horizontal_combined)
            os.makedirs(self.utils.SAVE_PATH+self.utils.MODEL_NAME+'imgs/epoch_'+str(self.epoch), exist_ok=True)
            horizontal_combined.save(self.utils.SAVE_FOLDER+self.utils.MODEL_NAME+'imgs/epoch_'+str(self.epoch)+'/'+name+'.png')

        
class Logger(Callback):
    """
    """
    def __init__(self, utils):
        """
        """
        self.utils = utils

    def on_epoch_end(self,
                     epoch,
                     logs=None,
                    ):
        """
        """
        with open(self.utils.SAVE_FOLDER+self.utils.MODEL_NAME+self.utils.LOG_FILE, 'a+') as f:
            msg = '{epoch:3d},{loss:1.3f},{iou_score:0.4f},{val_loss:1.3f},{val_iou_score:0.4f}\n'
            f.write(msg.format(epoch=epoch, 
                               loss=logs['loss'],
                               iou_score=logs['iou_score'],
                               val_loss=logs['val_loss'],
                               val_iou_score=logs['val_iou_score']
                               ))

class load_validation:
        """
        """
        def __init__(self,
                     utils,
                     labels,
                     ):
            """
            """
            self.utils = utils
            self.mapping = {x.id : x.trainId for x in self.utils.labels}

class Validation(Callback):
    """ 
    """
    def __init__(self, 
                 utils,
                 validation_steps,
                 labels,
                ):
        """
        """
        super(Validation, self).__init__()
        self.utils = utils
        self.validation_steps = validation_steps
        self.x = tf.data.TextLineDataset(utils.VAL_IMAGES)
        self.y = tf.data.TextLineDataset(utils.VAL_LABELS)
        self.validation_data = tf.data.Dataset.zip((self.x, self.y))
        self.reader = load_validation(self.utils, labels)


    def on_epoch_end(self,
                     epoch,
                     logs,
                    ):
        """
        """
        if epoch%5 == 0:
            self.step = int()
            visuals = validation_saves(self.utils,epoch+1, self.step+1)
            for sample_num, data in enumerate(self.validation_data.as_numpy_iterator()):
                val_img, ground_truth = data
                if sample_num%20 == 0:
                    val_img, ground_truth = self.reader(val_img, ground_truth)
                    prediction = self.model.predict(np.array([val_img]))
                    ground_truth = np.array(ground_truth, dtype = np.uint8)
                    pr = np.array(prediction[0]).reshape((self.utils.INPUT_SHAPE[0], self.utils.INPUT_SHAPE[1], self.utils.NUM_CLASSES)).argmax(axis = -1)
                    pr = np.array(pr, dtype = np.uint8)
                    gt = np.array(ground_truth, dtype= np.uint8).argmax(axis=-1)
                    visuals.save_stack(val_img,gt, pr)
                    self.step += 1