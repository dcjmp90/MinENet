Skip to content
Search or jump to…
Pulls
Issues
Marketplace
Explore
 
@dcjmp90 
dcjmp90
/
MinENet
Private
1
0
0
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
More
MinENet/train.py /
@dcjmp90
dcjmp90 Update train.py
Latest commit eadf223 1 hour ago
 History
 1 contributor
Executable File  61 lines (57 sloc)  2.48 KB
  
'''
'''
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from segmentation_models.losses import bce_dice_loss
from segmentation_models.metrics import IOUScore
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from utils import TFData
from model import MinENet
from utils import openeds_labels
from utils import args
import os


def keras_main():
    '''
    '''
    mirrored_strat = tf.distribute.MirroredStrategy()
    with mirrored_strat.scope():
        train = TFData(args.TRAIN_IMAGES, args.TRAIN_LABELS, args, openeds_labels)
        validate = TFData(args.VAL_IMAGES, args.VAL_LABELS, args, openeds_labels)
        validation_data = validate.prepare().repeat(args.EPOCHS)
        metric = IOUScore(class_indexes=[1,2,3])
        os.makedirs(args.SAVE_PATH+args.MODEL_NAME+'imgs/', exist_ok=True)
        os.makedirs(args.SAVE_PATH+args.MODEL_NAME+'hdf5/', exist_ok=True)
        save_path = args.SAVE_PATH+args.MODEL_NAME+'hdf5/epoch_{epoch:03d}_val-loss_{val_loss:0.3f}_val-iou_{val_iou_score:0.3f}.hdf5'
        checkpoint = ModelCheckpoint(save_path, 
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False,
                                     save_weights_only=False,
                                     mode='auto',
                                     save_freq='epoch',
                                    )
        learn_rate_on_plat = ReduceLROnPlateau(monitor='val_loss', verbose=1, min_lr = 1e-5, patience = 5)
        model = MinENet(args)
        model = model.create_model()
        print("model created")
        model.compile(loss=bce_dice_loss,
                      optimizer=Adam(lr=args.LEARN_RATE),
                      metrics=[metric],
                     )
        print("model compiled")
        model.fit(train.prepare().repeat(args.EPOCHS), 
                  steps_per_epoch=len(train)//args.BATCH_SIZE, 
                  validation_data=validation_data, 
                  validation_steps=len(validate)//args.BATCH_SIZE, 
                  epochs = args.EPOCHS, 
                  shuffle=False, 
                  use_multiprocessing = True, 
                  workers = 36, 
                  callbacks = [ checkpoint, 
                                learn_rate_on_plat,
                              ],
                 )


if __name__ == '__main__':
    keras_main()
