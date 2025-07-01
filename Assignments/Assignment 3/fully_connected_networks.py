"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from eecs598 import Solver


def hello_fully_connected_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from fully_connected_networks.py!')


class Linear(object):
    #################################NOTES################################
    # @staticmethod 是 Python 中的一个装饰器，用于定义静态方法。
    # 静态方法是类中的一种特殊方法，它不需要访问类的实例（self）或类本身（cls），
    # 因此它既不能修改实例状态，也不能修改类状态。
    # 静态方法通常用于实现与类相关但不需要访问类或实例数据的工具函数。
    # 特点：
    #     不需要 self 或 cls 参数：静态方法不需要传递类的实例或类本身作为参数。
    #     通过类或实例调用：静态方法可以通过类直接调用，也可以通过类的实例调用。
    #     独立于类和实例：静态方法与类和实例无关，仅仅是逻辑上属于类的函数。
    # 静态方法与普通方法的区别：
    #     普通方法：
    #         需要 self 参数来访问实例属性或方法。
    #         必须通过实例调用。
    #     类方法（@classmethod）：
    #         需要 cls 参数来访问类属性或方法。
    #         可以通过类或实例调用。
    #     静态方法（@staticmethod）：
    #         不需要 self 或 cls 参数。
    #         可以通过类或实例调用。
    # 使用场景：
    #     当方法逻辑上与类相关，但不需要访问类或实例数据时。
    #     用于工具函数或辅助函数，例如数学计算、格式转换等。
    # 总结：
    #     @staticmethod 用于定义静态方法，静态方法不需要访问类或实例数据。
    #     静态方法可以通过类或实例调用。
    #     静态方法适合实现与类相关但独立于类和实例的工具函数。
    #################################NOTES################################
    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ######################################################################
        # TODO: Implement the linear forward pass. Store the result in out.  #
        # You will need to reshape the input into rows.                      #
        ######################################################################
        # Replace "pass" statement with your code
        out = x.reshape(x.shape[0], -1).mm(w) + b
        ######################################################################
        #                        END OF YOUR CODE                            #
        ######################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ##################################################
        # TODO: Implement the linear backward pass.      #
        ##################################################
        # Replace "pass" statement with your code
        dx = dout.mm(w.t()).reshape(x.shape)
        dw = x.reshape(x.shape[0], -1).t().mm(dout)
        db = dout.sum(dim=0)
        ##################################################
        #                END OF YOUR CODE                #
        ##################################################
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        ###################################################
        # TODO: Implement the ReLU forward pass.          #
        # You should not change the input tensor with an  #
        # in-place operation.                             #
        ###################################################
        # Replace "pass" statement with your code
        out = torch.relu(x)
        ###################################################
        #                 END OF YOUR CODE                #
        ###################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        #####################################################
        # TODO: Implement the ReLU backward pass.           #
        # You should not change the input tensor with an    #
        # in-place operation.                               #
        #####################################################
        # Replace "pass" statement with your code
        dx = dout.clone()
        dx[x < 0] = 0
        #####################################################
        #                  END OF YOUR CODE                 #
        #####################################################
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        ###################################################################
        # TODO: Initialize the weights and biases of the two-layer net.   #
        # Weights should be initialized from a Gaussian centered at       #
        # 0.0 with standard deviation equal to weight_scale, and biases   #
        # should be initialized to zero. All weights and biases should    #
        # be stored in the dictionary self.params, with first layer       #
        # weights and biases using the keys 'W1' and 'b1' and second layer#
        # weights and biases using the keys 'W2' and 'b2'.                #
        ###################################################################
        # Replace "pass" statement with your code
        self.params['W1'] = weight_scale * torch.randn(input_dim, hidden_dim, dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)

        self.params['W2'] = weight_scale * torch.randn(hidden_dim, num_classes, dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(num_classes, dtype=dtype, device=device)
        ###############################################################
        #                            END OF YOUR CODE                 #
        ###############################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        #############################################################
        # TODO: Implement the forward pass for the two-layer net,   #
        # computing the class scores for X and storing them in the  #
        # scores variable.                                          #
        #############################################################
        # Replace "pass" statement with your code
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        h, linear_relu_cache = Linear_ReLU.forward(X, W1, b1)
        scores, linear_cache = Linear.forward(h, W2, b2)
        ##############################################################
        #                     END OF YOUR CODE                       #
        ##############################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the two-layer net.        #
        # Store the loss in the loss variable and gradients in the grads  #
        # dictionary. Compute data loss using softmax, and make sure that #
        # grads[k] holds the gradients for self.params[k]. Don't forget   #
        # to add L2 regularization!                                       #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and       #
        # you pass the automated tests, make sure that your L2            #
        # regularization does not include a factor of 0.5.                #
        ###################################################################
        # Replace "pass" statement with your code
        loss, dscores= softmax_loss(scores, y)
        
        dh, grads['W2'], grads['b2'] = Linear.backward(dscores, linear_cache)
        dx, grads['W1'], grads['b1'] = Linear_ReLU.backward(dh, linear_relu_cache)

        loss += self.reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))
        grads['W2'] += 2 * self.reg * W2
        grads['W1'] += 2 * self.reg * W1
        ###################################################################
        #                     END OF YOUR CODE                            #
        ###################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability
          for networks with dropout. If dropout=0 then the network
          should not use dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deteriminstic so we
          can gradient check the model.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        #######################################################################
        # TODO: Initialize the parameters of the network, storing all         #
        # values in the self.params dictionary. Store weights and biases      #
        # for the first layer in W1 and b1; for the second layer use W2 and   #
        # b2, etc. Weights should be initialized from a normal distribution   #
        # centered at 0 with standard deviation equal to weight_scale. Biases #
        # should be initialized to zero.                                      #
        #######################################################################
        # Replace "pass" statement with your code
        ##################################NOTES################################
        # Python 中可以通过多种方式将多个列表有序拼接起来:
        # 方法	                适用场景	                      特点
        # +运算符	           简单列表拼接	           直观，适合少量列表拼接
        # extend()方法	       动态扩展列表	             适合逐步扩展列表
        # itertools.chain()	 高效拼接大量列表	      返回迭代器，适合处理大数据
        # 列表推导式	            灵活拼接	               适合需要额外逻辑的场景
        # sum() 函数	        简洁拼接	             需要提供初始空列表，性能较低
        # * 解包操作符   Python 3.5+ 的简洁拼接方式  语法简洁，适合现代 Python代码
        ##################################NOTES################################
        net_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(1, self.num_layers + 1):
            self.params[f'W{i}'] = weight_scale * torch.randn(net_dims[i - 1], net_dims[i], 
                                                              dtype=self.dtype, device=device)
            self.params[f'b{i}'] = torch.zeros(net_dims[i], dtype=self.dtype, device=device)
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        # When using dropout we need to pass a dropout_param dictionary
        # to each dropout layer so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'use_dropout': self.use_dropout,
          'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'
        # Set train/test mode for batchnorm params and dropout param
        # since they behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        ##################################################################
        # TODO: Implement the forward pass for the fully-connected net,  #
        # computing the class scores for X and storing them in the       #
        # scores variable.                                               #
        #                                                                #
        # When using dropout, you'll need to pass self.dropout_param     #
        # to each dropout forward pass.                                  #
        ##################################################################
        # Replace "pass" statement with your code
        back_cache = {}
        drop_cache = {}
        
        scores = X
        for i in range(1, self.num_layers):
            scores, back_cache[f'{i}'] = Linear_ReLU.forward(scores, self.params[f'W{i}'], self.params[f'b{i}'])

            if self.use_dropout and self.dropout_param['p'] != 0:
                scores, drop_cache[f'{i}'] = Dropout.forward(scores, self.dropout_param)

        scores, back_cache[f'{self.num_layers}'] = Linear.forward(scores, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #####################################################################
        # TODO: Implement the backward pass for the fully-connected net.    #
        # Store the loss in the loss variable and gradients in the grads    #
        # dictionary. Compute data loss using softmax, and make sure that   #
        # grads[k] holds the gradients for self.params[k]. Don't forget to  #
        # add L2 regularization!                                            #
        # NOTE: To ensure that your implementation matches ours and you     #
        # pass the automated tests, make sure that your L2 regularization   #
        # includes a factor of 0.5 to simplify the expression for           #
        # the gradient.                                                     #
        #####################################################################
        # Replace "pass" statement with your code
        loss, dscores = softmax_loss(scores, y)
        
        dh, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = Linear.backward(dscores, back_cache[f'{self.num_layers}'])

        for i in range(self.num_layers - 1, 0, -1):
            if self.use_dropout and self.dropout_param['p'] != 0:
                dh = Dropout.backward(dh, drop_cache[f'{i}'])

            dh, grads[f'W{i}'], grads[f'b{i}'] = Linear_ReLU.backward(dh, back_cache[f'{i}'])
            
        for i in range(1, self.num_layers + 1):
            loss += self.reg * torch.sum(self.params[f'W{i}'] * self.params[f'W{i}'])
            grads[f'W{i}'] += 2 * self.reg * self.params[f'W{i}']
        ###########################################################
        #                   END OF YOUR CODE                      #
        ###########################################################

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    #############################################################
    # TODO: Use a Solver instance to train a TwoLayerNet that   #
    # achieves at least 50% accuracy on the validation set.     #
    #############################################################
    solver = None
    # Replace "pass" statement with your code
    solver = Solver(model, data_dict, optim_config={'learning_rate': 1e-1}, 
                    lr_decay=0.95, num_epochs=20, batch_size=200, device=device)
    ##############################################################
    #                    END OF YOUR CODE                        #
    ##############################################################
    return solver


def get_three_layer_network_params():
    ###############################################################
    # TODO: Change weight_scale and learning_rate so your         #
    # model achieves 100% training accuracy within 20 epochs.     #
    ###############################################################
    weight_scale = 1e-2   # Experiment with this!
    learning_rate = 1e-4  # Experiment with this!
    # Replace "pass" statement with your code
    weight_scale = 8e-2
    learning_rate = 4e-1
    ################################################################
    #                             END OF YOUR CODE                 #
    ################################################################
    return weight_scale, learning_rate


def get_five_layer_network_params():
    ################################################################
    # TODO: Change weight_scale and learning_rate so your          #
    # model achieves 100% training accuracy within 20 epochs.      #
    ################################################################
    learning_rate = 2e-3  # Experiment with this!
    weight_scale = 1e-5   # Experiment with this!
    # Replace "pass" statement with your code
    learning_rate = 5e-4
    weight_scale = 5e-1 
    #############################NOTES##############################
    # 较大的初始化权重可以加速模型的训练过程:
    #     打破对称性，使神经元学习到不同的特征。
    #     提供更强的梯度信号，加速参数更新。
    #     缓解梯度消失问题，使梯度能够传播到更深的层。
    #     使激活函数的输入落在敏感区间，产生更强的梯度。
    # 然而，较大的初始化权重也可能导致梯度爆炸或过拟合。
    #############################NOTES##############################
    ################################################################
    #                       END OF YOUR CODE                       #
    ################################################################
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))

    next_w = None
    ##################################################################
    # TODO: Implement the momentum update formula. Store the         #
    # updated value in the next_w variable. You should also use and  #
    # update the velocity v.                                         #
    ##################################################################
    # Replace "pass" statement with your code
    v = config['momentum'] * v - config['learning_rate'] * dw
    w += v
    
    next_w = w
    ###################################################################
    #                           END OF YOUR CODE                      #
    ###################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # Replace "pass" statement with your code
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw * dw
    w -= config['learning_rate'] * dw / (config['cache'].sqrt() + config['epsilon'])

    next_w = w
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    ##########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in#
    # the next_w variable. Don't forget to update the m, v, and t variables  #
    # stored in config.                                                      #
    #                                                                        #
    # NOTE: In order to match the reference output, please modify t _before_ #
    # using it in any calculations.                                          #
    ##########################################################################
    # Replace "pass" statement with your code
    config['t'] += 1

    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dw * dw
    
    m_unbias = config['m'] / (1 - config['beta1'] ** config['t']) 
    v_unbias = config['v'] / (1 - config['beta2'] ** config['t'])

    w -= config['learning_rate'] * m_unbias / (v_unbias.sqrt() + config['epsilon'])

    next_w = w
    #########################################################################
    #                              END OF YOUR CODE                         #
    #########################################################################

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])

        mask = None
        out = None

        if mode == 'train':
            ##############################################################
            # TODO: Implement training phase forward pass for            #
            # inverted dropout.                                          #
            # Store the dropout mask in the mask variable.               #
            ##############################################################
            # Replace "pass" statement with your code
            ############################NOTES#############################
            # torch.rand()生成的是均匀分布的随机数（区间 [0, 1)）。
            # torch.randn()生成的是标准正态分布的随机数（均值为 0，标准差为 1）
            # 如果需要生成均值为 mu、标准差为 sigma 的正态分布随机数，可以通过
            # random_tensor = mu + sigma * torch.randn()转换
            ############################NOTES#############################
            # p = torch.tensor(p, dtype=torch.float64)
            mask = (torch.rand(x.shape, dtype=torch.float64, device='cuda') > p) / p
            out = x * mask
            ##############################################################
            #                   END OF YOUR CODE                         #
            ##############################################################
        elif mode == 'test':
            ##############################################################
            # TODO: Implement the test phase forward pass for            #
            # inverted dropout.                                          #
            ##############################################################
            # Replace "pass" statement with your code
            out = x
            ##############################################################
            #                      END OF YOUR CODE                      #
            ##############################################################

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            ###########################################################
            # TODO: Implement training phase backward pass for        #
            # inverted dropout                                        #
            ###########################################################
            # Replace "pass" statement with your code
            dx = mask * dout
            ###########################################################
            #                     END OF YOUR CODE                    #
            ###########################################################
        elif mode == 'test':
            dx = dout
        return dx
