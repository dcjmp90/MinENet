import tensorflow as tensorflow
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.activations import softmax
from model import InputBlock
from model import DecoderBlock
from model import EncoderBlock

class MinENet:
    '''
    '''
    def __init__(self, utils):
        '''
        '''
        self.utils = utils

    def encoder(self, inputs, filters = 32, dropout = True):
        '''
        '''
        model = EncoderBlock(filters, 
                             downsample=True,
                             dropout=dropout,
                             )(inputs)
        model = EncoderBlock(filters, 
                             dropout=dropout,
                             asymmetric=True,
                             )(model)
        model = EncoderBlock(filters, 
                             dilated=True,
                             dilation_size=(2, 2),
                            )(model)    
        model = EncoderBlock(filters,
                             dilated=True,
                             dilation_size=(4, 4),
                             )(model)  
        model = EncoderBlock(filters,
                             dropout=dropout, 
                             downsample=True,
                             )(model)  
        model = EncoderBlock(filters, dropout=True)(model)   
        model = EncoderBlock(filters, downsample=True)(model)  
        return model

    def decoder(self, model, filters_out = 32):
        '''
        '''
        model = DecoderBlock(filters_out,
                             upsample=True,
                             )(model)
        model = DecoderBlock(filters_out)(model)
        model = DecoderBlock(filters_out)(model) 
        model = DecoderBlock(filters_out//2,
                             upsample=True,
                             )(model) 
        model = DecoderBlock(filters_out//2)(model)
        model = DecoderBlock(filters_out//2)(model)
        model = Conv2DTranspose(filters=self.utils.NUM_CLASSES, 
                                 kernel_size=(2, 2), 
                                 strides=(2, 2), 
                                 padding='same',
                               )(model)
        return model

    def input_layer(self):
        """
        """
        input_layer = Input(shape=(self.utils.INPUT_SHAPE[0], self.utils.INPUT_SHAPE[1], 1))
        return input_layer


    def create_model(self):
        """
        """
        
        input_layer = self.input_layer()
        initial_block = self.input_block = InputBlock()(input_layer)
        encode = self.encoder(initial_block)
        minenet_model = self.decoder(encode)
        output_shape = Model(input_layer, minenet_model).output_shape
        #minenet_model = Reshape((output_shape[1]*output_shape[2], self.utils.NUM_CLASSES))(minenet_model)
        minenet_model = Activation(softmax)(minenet_model)
        minenet_model = Model(input_layer, minenet_model)
        minenet_model.outputWidth = output_shape[2]
        minenet_model.outputHeight = output_shape[1]
        print("outputshape = ", output_shape)
        return minenet_model