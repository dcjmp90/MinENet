'''
'''
import tensorflow as tf
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.core import SpatialDropout2D, Permute, Activation, Reshape
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras import Input
from keras.models import Model
from keras.activations import softmax, relu


class MinENet:
    '''
    '''
    def __init__(self, utils):
        '''
        '''
        self.utils = utils
        self.min_enet = self.createModel()
    
    def init_block(self, input_layer, filters=32, kernel=(3,3), padding='same', strides=(2, 2)):
        input_layer = Conv2D(filters, kernel, padding=padding, strides=strides)(input_layer)
        input_layer = BatchNormalization(momentum=0.1)(input_layer)
        input_layer = PReLU(shared_axes=[1, 2])(input_layer)
        return input_layer

    def encoder(self, model, filters = 32, dropout = 0.05):
        '''
        '''
        model = self.encoder_block(model, filters, downsample= True, dropout = dropout)
        model = self.encoder_block(model, filters, dropout = dropout)
        model = self.encoder_block(model, filters, downsample=True)
        model = self.encoder_block(model, filters)
        model = self.encoder_block(model, filters, dilated=True, dilated_tuple = (2, 2))  
        model = self.encoder_block(model, filters, asymmetric=True)   
        model = self.encoder_block(model, filters)  
        model = self.encoder_block(model, filters, dilated=True, dilated_tuple = (4, 4))  
        model = self.encoder_block(model, filters, asymmetric=True) 
        model = self.encoder_block(model, filters)
        model = self.encoder_block(model, filters, dilated=True, dilated_tuple = (8, 8))  
        model = self.encoder_block(model, filters, asymmetric=True)
        model = self.encoder_block(model, filters)    
        model = self.encoder_block(model, filters, dilated=True, dilated_tuple = (16, 16)) 
        model = self.encoder_block(model, filters)  
        model = self.encoder_block(model, filters, dilated=True, dilated_tuple = (32, 32)) 
        model = self.encoder_block(model, filters)  
        return model

    def decoder(self, encoder, filters_out = 16):
        '''
        '''
        decode = self.decoder_block(encoder, filters_out, upsample=True, reverse =True)
        decode = self.decoder_block(decode, filters_out)
        decode = self.decoder_block(decode, filters_out) 
        decode = self.decoder_block(decode, filters_out//2, upsample=True, reverse =True) 
        decode = self.decoder_block(decode, filters_out//2) 
        decode = self.decoder_block(decode, filters_out//2, reverse =True)
        decode = self.decoder_block(decode, filters_out//2)  
        decode = Conv2DTranspose(filters=self.utils.n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(decode)
        return decode

    def decoder_block(self, minenet_model, filters_out, upsample = False, reverse = False, padding = 'same'):
        '''
        '''
        decode = Conv2D(filters_out, (1, 1), use_bias=False)(minenet_model)
        decode = BatchNormalization(momentum=0.1)(decode)
        decode = Activation(relu)(decode)
        if not upsample:
            decode = Conv2D(filters_out, (3, 3), padding=padding, use_bias=True)(decode)
        else:
            decode = Conv2DTranspose(filters_out, (3, 3), strides=(2, 2), padding=padding)(decode)
        decode = BatchNormalization(momentum=0.1)(decode)
        decode = Activation(relu)(decode)
        decode = Conv2D(filters_out, (1, 1), padding=padding, use_bias=False)(decode)
        other = minenet_model

        if minenet_model.get_shape()[-1] != filters_out*4 or upsample:
            other = Conv2D(filters_out, (1, 1), padding=padding, use_bias=False)(other)
            other = BatchNormalization(momentum=0.1)(other)
            if upsample and reverse is not False:
                other = UpSampling2D(size=(2, 2))(other)

        if upsample and reverse is False:
            _decoder = decode
        else:
            decode = BatchNormalization(momentum=0.1)(decode)
            _decoder = add([decode, other])
            _decoder = Activation(relu)(_decoder)
        return _decoder

    def encoder_block(self, encoder, filters, asymmetric = False, dilated = False, dilated_tuple = (0, 0), downsample = False, padding = 'same', dropout = 0.1):
        '''
        '''
        other = encoder
        inp = encoder
        input_stride = (2,2) if downsample else (1,1)
        encoder = Conv2D(filters, input_stride, strides=input_stride, use_bias=False)(encoder)
        encoder = BatchNormalization(momentum=0.1)(encoder)
        
        encoder = PReLU(shared_axes=[1, 2])(encoder)

        if not asymmetric and not dilated:
            encoder = Conv2D(filters, (3, 3), padding=padding)(encoder)
        elif asymmetric:
            encoder = Conv2D(filters, (1, 5), padding=padding, use_bias=False)(encoder)
            encoder = Conv2D(filters, (5, 1), padding=padding)(encoder)
        elif dilated:
            encoder = Conv2D(filters, (3, 3), dilation_rate=dilated_tuple, padding=padding)(encoder)
        encoder = BatchNormalization(momentum=0.1)(encoder)
        encoder = PReLU(shared_axes=[1, 2])(encoder)
        encoder = Conv2D(filters*2, (1, 1), use_bias=False)(encoder)
        encoder = BatchNormalization(momentum=0.1)(encoder)

        encoder = SpatialDropout2D(dropout)(encoder)

        if downsample:
            other = MaxPooling2D()(other)

            other = Permute((1, 3, 2))(other)
            pad_feature_maps = filters*2 - inp.get_shape().as_list()[3]
            tb_pad = (0, 0)
            lr_pad = (0, pad_feature_maps)
            other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
            other = Permute((1, 3, 2))(other)

        encoder = add([encoder, other])
        encoder = PReLU(shared_axes=[1, 2])(encoder)

        return encoder

    def createModel(self):
        '''
        '''
        assert self.utils.input_height % 32 == 0
        assert self.utils.input_width % 32 == 0
        input_layer = Input(shape=(self.utils.input_height, self.utils.input_width, self.utils.input_ch))
        initial_block = self.init_block(input_layer)
        encode = self.encoder(initial_block)
        minenet_model = self.decoder(encode)
        output_shape = Model(input_layer, minenet_model).output_shape
        minenet_model = Reshape((output_shape[1]*output_shape[2], self.utils.n_classes))(minenet_model)
        minenet_model = Activation(softmax)(minenet_model)
        minenet_model = Model(input_layer, minenet_model)
        minenet_model.outputWidth = output_shape[2]
        minenet_model.outputHeight = output_shape[1]
        print('Output H = ',output_shape[1],'\n','Output W =', output_shape[2],'\n')
        return minenet_model

