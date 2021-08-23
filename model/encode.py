import tensorflow as tf
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Conv2D, ZeroPadding2D
from tensorflow.keras.layers import Dropout, Permute
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D

class EncoderBlock:
    """
    """
    def __init__(self,
                 filters,
                 asymmetric = False,
                 dilated = False,
                 dilation_size = (1,1),
                 downsample = False,
                 padding = 'same',
                 dropout = True,
                 dropout_rate = 0.2,
                 momentum_rate=0.1,
                 asymmetric_tuples=[(1,5),(5,1)],
                 shared_axes=[1, 2],
                 reduction_kernel=(3,3),
                 full_connection_kernel=(1,1),
                 use_bias=False,
                 permute = (1,3,2),
                 ):
        """
        """
        self.filters = filters
        self.asymmetric = asymmetric
        self.dilated = dilated
        self.dilation_size = dilation_size
        self.downsample = downsample
        self.padding = padding
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.momentum_rate = momentum_rate
        self.asymmetric_tuples = asymmetric_tuples
        self.shared_axes = shared_axes
        self.reduction_kernel = reduction_kernel
        self.full_connection_kernel = full_connection_kernel
        self.use_bias = use_bias
        self.permute = permute

    def __call__(self, inputs):
        """
        """
        kernel = self.reduction_kernel if self.downsample else self.full_connection_kernel
        encoder = Conv2D(self.filters, 
                         kernel,
                         padding=self.padding, 
                         use_bias=self.use_bias,
                         )(inputs)
        encoder = BatchNormalization(momentum=self.momentum_rate)(encoder)
        encoder = PReLU(shared_axes=self.shared_axes)(encoder)

        if self.downsample:
            other = MaxPooling2D()(inputs)
            encoder = MaxPooling2D()(encoder)
        else:
            other = encoder
        if not self.asymmetric and not self.dilated:
            encoder = Conv2D(self.filters,
                             self.reduction_kernel, 
                             padding=self.padding,
                             use_bias=self.use_bias,
                             )(encoder)
        elif self.asymmetric:
            encoder = Conv2D(self.filters, 
                            self.asymmetric_tuples[0],
                            padding=self.padding,
                            use_bias=self.use_bias,
                            )(encoder)
            encoder = Conv2D(self.filters, 
                             self.asymmetric_tuples[1],
                             padding=self.padding,
                             use_bias=self.use_bias,
                             )(encoder)
        elif self.dilated:
            encoder = Conv2D(self.filters, 
                             self.reduction_kernel, 
                             dilation_rate=self.dilation_size, 
                             padding=self.padding,
                             use_bias=self.use_bias,
                             )(encoder)
        encoder = BatchNormalization(momentum=self.momentum_rate)(encoder)
        encoder = PReLU(shared_axes=self.shared_axes)(encoder) 
        encoder = Conv2D(self.filters, 
                         self.reduction_kernel, 
                         use_bias=self.use_bias,
                         padding=self.padding,
                         )(encoder)
        encoder = BatchNormalization(momentum=self.momentum_rate)(encoder)
        encoder = Dropout(self.dropout_rate)(encoder)
        encoder = add([encoder, other])
        encoder = PReLU(shared_axes=self.shared_axes)(encoder)

        return encoder