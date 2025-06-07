"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        '''
        torch.nn.functional.pad(input, pad, mode='constant', value=0), 沿着张量的边界以指定的值或模式进行填充。
            input (Tensor): 需要填充的输入张量。
            pad (tuple): 一个元组，指定张量每个维度的填充大小。元组的长度必须是偶数，格式为 (pad_left, pad_right, pad_top, pad_bottom, ...)，依次对应每个维度。
            mode (str, 可选): 填充模式。支持的填充模式有：
                'constant': 使用常数填充（默认模式）。
                'reflect': 使用反射填充，不重复边界值。
                'replicate': 使用边界值复制填充。
                'circular': 使用循环填充（张量内容循环重复）。
            value (float, 可选): 在 constant 模式下使用的填充值，默认为 0。
        '''
        pad = conv_param['pad']
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        _H = 1 + (H + 2 * pad - HH) // stride
        _W = 1 + (W + 2 * pad - WW) // stride

        out = torch.zeros(N, F, _H, _W, dtype=x.dtype, device='cuda')
        
        for n in range(N):
            for f in range(F):
                for i in range(_H):
                    for j in range(_W):
                        out[n, f, i, j] =  torch.sum(x_pad[n, :, stride * i:stride * i + HH, stride * j:stride * j + WW] * w[f]) + b[f] 
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        x, w, b, conv_param = cache
        pad = conv_param['pad']
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        dx_pad = torch.zeros_like(x_pad)
        dw = torch.zeros_like(w)
        db = torch.zeros_like(b)

        db = torch.sum(dout, dim=(0, 2, 3))
        
        for n in range(N):
            for f in range(F):
                for i in range(0, H + 2 * pad - HH + 1, stride):
                    for j in range(0, W + 2 * pad - WW + 1, stride):
                        dw[f] += x_pad[n, :, i:i+HH, j:j+WW] * dout[n, f, i // stride, j // stride]
                        dx_pad[n, :, i:i+HH, j:j+WW] += w[f] * dout[n, f, i // stride, j // stride]

        if pad == 0:
            dx = dx_pad
        else:
            dx = dx_pad[:, :, pad:-pad, pad:-pad]
        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        N, C, H, W = x.shape
        p_H = pool_param['pool_height']
        p_W = pool_param['pool_width']
        stride = pool_param['stride']

        _H = 1 + (H - p_H) // stride
        _W = 1 + (W - p_W) // stride

        out = torch.zeros(N, C, _H, _W, dtype=x.dtype, device='cuda')

        for n in range(N):
            for c in range(C):
                for h in range(_H):
                    for w in range(_W):
                        out[n, c, h, w] = torch.max(x[n, c, h * stride:h * stride + p_H, w * stride:w * stride + p_W])
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        x, pool_param = cache
        N, C, H, W = x.shape
        p_H = pool_param['pool_height']
        p_W = pool_param['pool_width']
        stride = pool_param['stride']

        dx = torch.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for h in range(0, H - p_H + 1, stride):
                    for w in range(0, W - p_W + 1, stride):
                        pool_region = x[n, c, h:h+p_H, w:w+p_W]
                        mask = (pool_region == torch.max(pool_region))
                        
                        dx[n, c, h:h+p_H, w:w+p_W] += mask * dout[n, c, h // stride, w // stride] 
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        self.params['W1'] = weight_scale * torch.randn(num_filters, input_dims[0], filter_size, filter_size, dtype=self.dtype, device=device)
        self.params['b1'] = torch.zeros(num_filters, dtype=self.dtype, device=device)

        num_conv_relu_pool = num_filters * input_dims[1] * input_dims[2] // 4
        self.params['W2'] = weight_scale * torch.randn(num_conv_relu_pool,  hidden_dim, dtype=self.dtype, device=device)
        self.params['b2'] = torch.zeros(hidden_dim, dtype=self.dtype, device=device)

        self.params['W3'] = weight_scale * torch.randn(hidden_dim, num_classes, dtype=self.dtype, device=device)
        self.params['b3'] = torch.zeros(num_classes, dtype=self.dtype, device=device)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        # Replace "pass" statement with your code
        conv_out, conv_cache = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        linear_relu_out, linear_relu_cache = Linear_ReLU.forward(conv_out, W2, b2)
        scores, linear_cache = Linear.forward(linear_relu_out, W3, b3)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # Replace "pass" statement with your code
        loss, dscores = softmax_loss(scores, y)

        dl, grads['W3'], grads['b3'] = Linear.backward(dscores, linear_cache)
        dh, grads['W2'], grads['b2'] = Linear_ReLU.backward(dl, linear_relu_cache)
        dx, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dh, conv_cache)

        loss += self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2) + torch.sum(W3 * W3))
        grads['W1'] += 2 * self.reg * W1
        grads['W2'] += 2 * self.reg * W2
        grads['W3'] += 2 * self.reg * W3
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        # Replace "pass" statement with your code
        if weight_scale == 'kaiming':
            self.params['W1'] = kaiming_initializer(num_filters[0], input_dims[0], K=3, relu=True, dtype=self.dtype, device=device)
            self.params['b1'] = torch.zeros(num_filters[0], dtype=self.dtype, device=device)

            if self.batchnorm:
                self.params['gamma1'] = torch.ones(num_filters[0], dtype=self.dtype, device=device)
                self.params['beta1'] = torch.zeros(num_filters[0], dtype=self.dtype, device=device)
        
            for i in range(self.num_layers - 2):
                ind = i + 2
                self.params[f'W{ind}'] = kaiming_initializer(num_filters[i + 1], num_filters[i], K=3, relu=True, dtype=self.dtype, device=device)
                self.params[f'b{ind}'] = torch.zeros(num_filters[i + 1], dtype=self.dtype, device=device)

                if self.batchnorm:
                    self.params[f'gamma{ind}'] = torch.ones(num_filters[i + 1], dtype=self.dtype, device=device)
                    self.params[f'beta{ind}'] = torch.zeros(num_filters[i + 1], dtype=self.dtype, device=device)

            num_full = num_filters[-1] * input_dims[1] * input_dims[2] // ((2 ** len(self.max_pools)) ** 2)
            self.params[f'W{self.num_layers}'] = kaiming_initializer(num_full, num_classes, K=None, relu=False, dtype=self.dtype, device=device)
            self.params[f'b{self.num_layers}'] = torch.zeros(num_classes, dtype=self.dtype, device=device)
        else:
            self.params['W1'] = weight_scale * torch.randn(num_filters[0], input_dims[0], 3, 3, dtype=self.dtype, device=device)
            self.params['b1'] = torch.zeros(num_filters[0], dtype=self.dtype, device=device)

            if self.batchnorm:
                self.params['gamma1'] = torch.ones(num_filters[0], dtype=self.dtype, device=device)
                self.params['beta1'] = torch.zeros(num_filters[0], dtype=self.dtype, device=device)
        
            for i in range(self.num_layers - 2):
                ind = i + 2
                self.params[f'W{ind}'] = weight_scale * torch.randn(num_filters[i + 1], num_filters[i], 3, 3, dtype=self.dtype, device=device)
                self.params[f'b{ind}'] = torch.zeros(num_filters[i + 1], dtype=self.dtype, device=device)

                if self.batchnorm:
                    self.params[f'gamma{ind}'] = torch.ones(num_filters[i + 1], dtype=self.dtype, device=device)
                    self.params[f'beta{ind}'] = torch.zeros(num_filters[i + 1], dtype=self.dtype, device=device)

            num_full = num_filters[-1] * input_dims[1] * input_dims[2] // ((2 ** len(self.max_pools)) ** 2)
            self.params[f'W{self.num_layers}'] = weight_scale * torch.randn(num_full, num_classes, dtype=self.dtype, device=device)
            self.params[f'b{self.num_layers}'] = torch.zeros(num_classes, dtype=self.dtype, device=device)
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for i in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code
        conv_cache = {}
        pool_cache = {}
        batchnorm_cache = {}

        if self.batchnorm:
            out, batchnorm_cache['1'] = Conv_BatchNorm_ReLU.forward(X, self.params['W1'], self.params['b1'], 
                                                                    self.params['gamma1'], self.params['beta1'], conv_param, self.bn_params[0])
        else:
            out, conv_cache['1'] = Conv_ReLU.forward(X, self.params['W1'], self.params['b1'], conv_param)
        
        if 0 in self.max_pools:
            out, pool_cache['1'] = FastMaxPool.forward(out, pool_param)
        
        for i in range(self.num_layers - 2):
            ind = i + 2

            if self.batchnorm:
                out, batchnorm_cache[f'{ind}'] = Conv_BatchNorm_ReLU.forward(out, self.params[f'W{ind}'], self.params[f'b{ind}'], 
                                                                             self.params[f'gamma{ind}'], self.params[f'beta{ind}'],
                                                                            conv_param, self.bn_params[ind - 1])
            else:
                out, conv_cache[f'{ind}'] = Conv_ReLU.forward(out, self.params[f'W{ind}'], self.params[f'b{ind}'], conv_param)
            
            if (ind - 1) not in self.max_pools:
                continue
    
            out, pool_cache[f'{ind}'] = FastMaxPool.forward(out, pool_param)

        scores, linear_cache = Linear.forward(out, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code
        loss, dscores = softmax_loss(scores, y)

        dout, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = Linear.backward(dscores, linear_cache)

        for i in range(self.num_layers - 2):
            ind = self.num_layers - i - 1
            
            if (ind - 1) in self.max_pools:
                dout = FastMaxPool.backward(dout, pool_cache[f'{ind}'])

            if self.batchnorm:
                dout, grads[f'W{ind}'], grads[f'b{ind}'], grads[f'gamma{ind}'], grads[f'beta{ind}'] = Conv_BatchNorm_ReLU.backward(dout, 
                                                                                                                                   batchnorm_cache[f'{ind}'])
            else:
                dout, grads[f'W{ind}'], grads[f'b{ind}'] = Conv_ReLU.backward(dout, conv_cache[f'{ind}'])

        if 0 in self.max_pools:
            dout = FastMaxPool.backward(dout, pool_cache['1'])

        if self.batchnorm:
            dx, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = Conv_BatchNorm_ReLU.backward(dout, batchnorm_cache['1'])
        else:
            dx, grads['W1'], grads['b1'] = Conv_ReLU.backward(dout, conv_cache['1'])

        for i in range(self.num_layers):
            ind = i + 1
            loss += self.reg * torch.sum(self.params[f'W{ind}'] * self.params[f'W{ind}'])
            grads[f'W{ind}'] += 2 * self.reg * self.params[f'W{ind}']
        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    weight_scale = 1e-1
    learning_rate = 1e-3
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code
    model = DeepConvNet(input_dims=data_dict['X_train'].shape[1:], num_classes=10,
                       num_filters=[8, 32, 128, 512], 
                       max_pools=[0, 1, 2, 3],
                       weight_scale='kaiming',
                       reg=2e-4,
                       dtype=dtype,
                       device=device)

    solver = Solver(model, data_dict,
                   num_epochs=5, batch_size=128,
                   update_rule=adam,
                   optim_config={
                       'learning_rate': 3e-3
                   },
                   lr_decay=0.93, 
                   print_every=100, device=device)
    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        weight = torch.sqrt(torch.tensor(gain / Din)) * torch.randn(Din, Dout, dtype=dtype, device=device)
        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        weight = torch.sqrt(torch.tensor(gain / (Din * K * K))) * torch.randn(Din, Dout, K, K, dtype=dtype, device=device)
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code
            sample_mean = torch.sum(x, dim=0) / N
            sample_var = torch.sum((x - sample_mean) ** 2, dim=0) / N
            
            x_ = (x - sample_mean) / torch.sqrt(sample_var + eps)
            out = gamma * x_ + beta

            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var

            cache = (mode, x, running_var, eps, x_, gamma)
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code
            x_ = (x - running_mean) / torch.sqrt(running_var + eps)
            
            out = gamma * x_ + beta
            cache = (mode, x, running_var, eps, x_, gamma)
            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your code
        mode, x, running_var, eps, x_, gamma = cache
        N = x.shape[0]

        if mode == 'train':
            # 导数公式详见paper
            sample_mean = torch.sum(x, dim=0) / N
            sample_var = torch.sum((x - sample_mean) ** 2, dim=0) / N
            
            dx_ = dout * gamma
            dvar = torch.sum(dx_ * (x - sample_mean), dim=0) * ((- 1 / 2) / torch.pow(torch.sqrt(sample_var + eps), 3))
            dmean = -torch.sum(dx_, dim=0) / torch.sqrt(sample_var + eps) + dvar * torch.sum(2 * (sample_mean - x), dim=0) / N
            
            dx = dx_ / torch.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean) / N + dmean / N
            dgamma = torch.sum(dout * x_, dim=0)
            dbeta = torch.sum(dout, dim=0)
        elif mode == 'test':
            dx = gamma * dout / torch.sqrt(running_var + eps)
            dgamma = torch.sum(dout * x_, dim=0)
            dbeta = torch.sum(dout, dim=0)
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###################################################################
        # TODO: Implement the backward pass for batch normalization.      #
        # Store the results in the dx, dgamma, and dbeta variables.       #
        #                                                                 #
        # After computing the gradient with respect to the centered       #
        # inputs, you should be able to compute gradients with respect to #
        # the inputs in a single statement; our implementation fits on a  #
        # single 80-character line.                                       #
        ###################################################################
        # Replace "pass" statement with your code
        # backward中实现的就是这个版本？
        pass
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code
        N, C, H, W = x.shape
        
        x = x.reshape(-1, C)
        
        out, cache = BatchNorm.forward(x, gamma, beta, bn_param)
        
        out = out.reshape(N, C, H, W)
        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code
        N, C, H, W = dout.shape
        
        dout = dout.reshape(-1, C)
        
        dx, dgamma, dbeta = BatchNorm.backward(dout, cache)

        dx = dx.reshape(N, C, H, W)
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        '''
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            in_channels (int): 输入数据的通道数。
            out_channels (int): 输出数据的通道数。它决定了卷积核的数量。
            kernel_size (int or tuple): 卷积核的大小。可以是一个整数（表示正方形卷积核）或一个元组 (height, width)。
            stride (int or tuple, 可选): 卷积操作的步幅。默认值为 1。可以是一个整数（表示水平和垂直方向步幅相同）或一个元组 (stride_height, stride_width)。
            padding (int or tuple, 可选): 输入数据的填充大小。默认值为 0（不填充）。可以是一个整数或一个元组 (padding_height, padding_width)。
            dilation (int or tuple, 可选): 空洞卷积的膨胀率。默认值为 1（普通卷积）。可以是一个整数或一个元组 (dilation_height, dilation_width)。
            groups (int, 可选): 分组卷积的组数。默认值为 1（普通卷积）。如果 groups > 1，则输入和输出通道会被分成 groups 组，每组独立进行卷积。
            bias (bool, 可选): 是否使用偏置项。默认值为 True。
            padding_mode (str, 可选): 填充模式。默认值为 'zeros'（零填充）。其他可选模式包括 'reflect'、'replicate' 和 'circular'。
        '''
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        '''
        torch.nn.Parameter(data, requires_grad=True), 用于将张量标记为模型参数。是torch.Tensor的子类，作用是告诉PyTorch该张量需要在训练过程中更新。
        自动注册为模型参数：当torch.nn.Parameter被赋值给torch.nn.Module的属性时，会自动注册为模型参数。意味着会被包含在model.parameters()中，在训练过程中被优化器更新。
        支持梯度计算：默认情况下，torch.nn.Parameter 的 requires_grad 为 True，因此会自动计算梯度。
        与普通张量的区别：torch.nn.Parameter 是 torch.Tensor 的子类，因此支持所有张量操作。与普通张量不同，torch.nn.Parameter 会被 PyTorch 自动识别为模型参数
        '''
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        '''
        detach(), PyTorch中的一个方法，用于从计算图中分离张量。返回一个新的张量，该张量与原始张量共享数据，但不会参与梯度计算。会切断反向传播时梯度流向该张量的路径。
        使用场景
            阻止梯度计算：当你希望某些张量不参与反向传播时，可以使用 detach()。例如，在生成对抗网络（GAN）中，生成器的输出在训练判别器时不需要梯度。
            获取张量的值：当你需要获取张量的值（而不是计算图的一部分）时，可以使用 detach()。例如，将张量从 GPU 移动到 CPU 或转换为 NumPy 数组时。
            避免内存泄漏：在某些情况下，使用detach()可以避免不必要的计算图保存，从而减少内存占用。
        注意事项
            共享数据：detach()返回的张量与原始张量共享数据，修改其中一个张量会影响另一个张量。如果需要完全独立的张量，可以使用detach().clone()。
            与with torch.no_grad()的区别：detach()是张量级别的操作，仅对特定张量生效。with torch.no_grad()是上下文管理器，会禁用整个代码块中的梯度计算。
            与requires_grad=False的区别：detach()返回的张量仍然可以手动设置 requires_grad=True。直接设置 requires_grad=False 会完全禁用梯度计算，且无法重新启用。
        '''
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            '''
            tensor.backward(gradient=None, retain_graph=None, create_graph=False), 自动求导的核心方法。通过反向传播计算张量的梯度，并将梯度存储在张量的.grad属性中。
            参数
                gradient (Tensor or None, 可选): 用于链式求导的梯度。通常用于计算非标量张量的梯度。默认值为 None，表示标量张量的梯度为 1。
                retain_graph (bool, 可选): 是否保留计算图。为 True则计算图不会被释放，可以多次调用 backward()。为 False则计算图会在调用backward()后被释放。
                create_graph (bool, 可选): 是否创建计算图以支持高阶导数。默认值为 False。如果为 True，则可以计算二阶或更高阶的导数。
            使用场景
                计算标量张量的梯度：当损失函数是一个标量时，可以直接调用 backward()。
                计算非标量张量的梯度：当损失函数是一个非标量张量时，需要传入 gradient 参数。
                多次反向传播：如果需要多次调用 backward()，需要设置 retain_graph=True。
                高阶导数：如果需要计算高阶导数，需要设置 create_graph=True。
            注意事项
                梯度累加：每次调用 backward() 时，梯度会累加到 .grad 属性中。如果需要清零梯度，可以调用 x.grad.zero_()。
                释放计算图：默认情况下，调用 backward() 后会释放计算图。如果需要多次调用 backward()，需要设置 retain_graph=True。
                非标量张量的梯度：对于非标量张量，必须传入 gradient 参数，否则会抛出错误。
            '''
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        '''
        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            kernel_size (int or tuple): 池化窗口的大小。可以是一个整数（表示正方形窗口）或一个元组 (height, width)。
            stride (int or tuple, 可选): 池化操作的步幅。默认值为 kernel_size。可以是一个整数或一个元组 (stride_height, stride_width)。
            padding (int or tuple, 可选): 输入数据的填充大小。默认值为 0（不填充）。可以是一个整数或一个元组 (padding_height, padding_width)。
            dilation (int or tuple, 可选): 空洞池化的膨胀率。默认值为 1（普通池化）。可以是一个整数或一个元组 (dilation_height, dilation_width)。
            return_indices (bool, 可选): 是否返回最大值的位置索引。默认值为 False。如果为 True，则返回一个元组 (output, indices)，其中 indices 是最大值的位置索引。
            ceil_mode (bool, 可选): 是否使用 ceil 模式计算输出大小。默认值为 False。如果为 True，则使用 ceil 函数计算输出大小；否则使用 floor 函数
        '''
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
