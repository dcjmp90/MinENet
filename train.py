'''
'''
import argparse
from MinENet import MinENet
from Utils import Utils, Generator, OpenEDS, CityScape
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from segmentation_models.losses import bce_dice_loss
from segmentation_models.metrics import IOUScore



def _main(train, validate, util):
    '''
    '''
    G_train = Generator(train, util.batch_size)
    G_validate = Generator(validate, util.batch_size)
    model = MinENet(util)
    model = model.createModel()
    metric = IOUScore()
    print(model.summary())
    with open(util.save_weights_path+'model_summary.txt','w') as f:
        model.summary(print_fn=lambda line: f.write(line + '\n'))
    model = multi_gpu_model(model, gpus=2)
    model.compile( loss = bce_dice_loss, optimizer = Adam(), metrics=[metric])
    for epoch in range(util.epochs):  
        model.fit_generator(  G_train.generate(util)
                            , util.steps_per_epoch
                            , validation_data=G_validate.generate(util)
                            , validation_steps=util.validation_steps
                            , epochs = 1
                            #, use_multiprocessing = True
                            )
        model.save_weights(util.save_weights_path + 'weights.'+str(epoch))
        model.save(util.save_weights_path+'model.'+str(epoch))


if __name__ == '__main__':
    cityscape = CityScape()
    train_data, validate_data = cityscape.create_dataset()
    _main(train_data, validate_data, cityscape)
