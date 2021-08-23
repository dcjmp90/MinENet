import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import relu

class DecoderBlock:
    """
    """
    def __init__(self,
                 filters,
                 dilated = False,
                 dilation_size = (1,1),
                 padding = 'same',
                 dropout = False,
                 dropout_rate = 0.2,
                 momentum_rate=0.1,
                 reduction_kernel=(3,3),
                 full_connection_kernel=(1,1),
                 use_bias=False,
                 upsample_rate = (2,2),
                 reserve = False,
                 upsample = False,
                 ):
        """
        """
        self.filters = filters
        self.dilated = dilated
        self.dilation_size = dilation_size
        self.upsample = upsample
        self.upsample_rate = upsample_rate
        self.padding = padding
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.momentum_rate = momentum_rate
        self.reduction_kernel = reduction_kernel
        self.full_connection_kernel = full_connection_kernel
        self.use_bias = use_bias
        self.reverse = reserve

    def __call__(self, inputs):
        """
        """
        decode = Conv2D(self.filters, 
                        self.full_connection_kernel, 
                        use_bias=self.use_bias,
                        )(inputs)
        decode = BatchNormalization(momentum=self.momentum_rate)(decode)
        decode = Activation(relu)(decode)
        if not self.upsample:
            decode = Conv2D(self.filters, 
                            self.reduction_kernel,  
                            padding=self.padding, 
                            use_bias=self.use_bias,
                            )(decode)
        else:
            decode = Conv2DTranspose(self.filters,
                                     self.reduction_kernel,
                                     strides=self.upsample_rate,
                                     padding=self.padding,
                                     )(decode)
        decode = BatchNormalization(momentum=self.momentum_rate)(decode)
        decode = Activation(relu)(decode)
        decode = Conv2D(self.filters,
                        self.full_connection_kernel,
                        padding=self.padding, 
                        use_bias=self.use_bias,
                        )(decode)
        other = inputs

        if inputs.get_shape()[-1] != self.filters*4 or self.upsample:
            other = Conv2D(self.filters,
                           self.full_connection_kernel, 
                           padding=self.padding, 
                           use_bias=self.use_bias,
                           )(other)
            other = BatchNormalization(momentum=self.momentum_rate)(other)
            if self.upsample and self.reverse is not False:
                other = UpSampling2D(size=self.upsample_rate)(other)

        if self.upsample and self.reverse is False:
            _decoder = decode
        else:
            decode = BatchNormalization(momentum=self.momentum_rate)(decode)
            _decoder = add([decode, other])
            _decoder = Activation(relu)(_decoder)
        return _decoder