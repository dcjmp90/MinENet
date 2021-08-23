import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import BatchNormalization

class InputBlock:
    """
    """
    def __init__(self,
                 filters = 32,
                 kernel = (3,3),
                 padding = 'same',
                 strides = (2,2),
                 name = 'Input Block',
                 use_bias=False,
                 momentum_rate=0.1,
                 shared_axes=[1,2],
                 ):
        """
        """
        self.filters = filters
        self.kernel_size = kernel
        self.padding = padding
        self.stride_size = strides
        self.name = name
        self.use_bias = use_bias
        self.momentum_rate = momentum_rate
        self.shared_axes = shared_axes

    def __call__(self, inputs):
        """
        """
        input_layer = Conv2D(self.filters,
                             self.kernel_size, 
                             padding=self.padding, 
                             use_bias=self.use_bias,
                             )(inputs)
        input_layer = BatchNormalization(momentum=self.momentum_rate)(input_layer)
        input_layer = PReLU(shared_axes=self.shared_axes)(input_layer)
        return input_layer