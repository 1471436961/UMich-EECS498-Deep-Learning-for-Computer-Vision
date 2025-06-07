from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


def hello_vae():
    print("Hello from vae.py!")


class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = None  # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ###########################################################################
        # TODO: Implement the fully-connected encoder architecture described in   #
        # the notebook. Specifically, self.encoder should be a network that       #
        # inputs a batch of input images of shape (N, 1, H, W) into a batch of    #
        # hidden features of shape (N, H_d). Set up self.mu_layer and             #
        # self.logvar_layer to be a pair of linear layers that map the hidden     #
        # features into estimates of the mean and log-variance of the posterior   #
        # over the latent vectors; the mean and log-variance estimates will both  #
        # be tensors of shape (N, Z).                                             #
        ###########################################################################
        # Replace "pass" statement with your code
        self.hidden_dim = 400
        self.encoder = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.input_size, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU()
        )
        self.mu_layer = torch.nn.Linear(self.hidden_dim, self.latent_size)
        self.logvar_layer = torch.nn.Linear(self.hidden_dim, self.latent_size)
        ###########################################################################
        # TODO: Implement the fully-connected decoder architecture described in   #
        # the notebook. Specifically, self.decoder should be a network that inputs#
        # a batch of latent vectors of shape (N, Z) and outputs a tensor of       #
        # estimated images of shape (N, 1, H, W).                                 #
        ###########################################################################
        # Replace "pass" statement with your code
        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_size, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.input_size),
            torch.nn.Sigmoid(),
            torch.nn.Unflatten(1, (1, 28, 28))
        )
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################

    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through
        encoder, reparametrize trick, and decoder models

        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)

        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent
          space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z),
          with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ###########################################################################
        # TODO: Implement the forward pass by following these steps               #
        # (1) Pass the input batch through the encoder model to get posterior     #
        #     mu and logvariance                                                  #
        # (2) Reparametrize to compute  the latent vector z                       #
        # (3) Pass z through the decoder to resconstruct x                        #
        ###########################################################################
        # Replace "pass" statement with your code
        h = self.encoder(x)
        mu, logvar = self.mu_layer(h), self.logvar_layer(h)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################
        return x_hat, mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.num_classes = num_classes  # C
        self.hidden_dim = None  # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ###########################################################################
        # TODO: Define a FC encoder as described in the notebook that transforms  #
        # the image--after flattening and now adding our one-hot class vector (N, #
        # H*W + C)--into a hidden_dimension (N, H_d) feature space, and a final   #
        # two layers that project that feature space to posterior mu and posterior#
        # log-variance estimates of the latent space (N, Z)                       #
        ###########################################################################
        # Replace "pass" statement with your code
        self.hidden_dim = 400
        self.encoder = nn.Sequential(
            torch.nn.Linear(self.input_size + self.num_classes, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU()
        )
        self.mu_layer = torch.nn.Linear(self.hidden_dim, self.latent_size)
        self.logvar_layer = torch.nn.Linear(self.hidden_dim, self.latent_size)
        ###########################################################################
        # TODO: Define a fully-connected decoder as described in the notebook that#
        # transforms the latent space (N, Z + C) to the estimated images of shape #
        # (N, 1, H, W).                                                           #
        ###########################################################################
        # Replace "pass" statement with your code
        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_size + self.num_classes, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.input_size),
            torch.nn.Sigmoid(),
            torch.nn.Unflatten(1, (1, 28, 28))
        )
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################

    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through
        encoder, reparametrize trick, and decoder models

        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)

        Returns:
        - x_hat: Reconstructed input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent
          space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with
          Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ###########################################################################
        # TODO: Implement the forward pass by following these steps               #
        # (1) Pass the concatenation of input batch and one hot vectors through   #
        #     the encoder model to get posterior mu and logvariance               #
        # (2) Reparametrize to compute the latent vector z                        #
        # (3) Pass concatenation of z and one hot vectors through the decoder to  #
        #     resconstruct x                                                      #
        ###########################################################################
        # Replace "pass" statement with your code
        '''
        torch.cat(
            tensors,    # 要拼接的张量序列（列表或元组）
            dim=0,      # 沿哪个维度拼接（默认为0）
            out=None    # 可选输出张量
        )
        拼接条件：所有张量在非拼接维度上的形状必须相同
        '''
        conditional_x = torch.cat((x.view(x.shape[0], -1), c), dim=1)
        
        h = self.encoder(conditional_x)
        
        mu, logvar = self.mu_layer(h), self.logvar_layer(h)
        z = reparametrize(mu, logvar)
    
        conditional_z = torch.cat((z, c), dim=1)
        
        x_hat = self.decoder(conditional_z)
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################
        return x_hat, mu, logvar


def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance
    using the reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with
    mean mu and standard deviation sigma, such that we can backpropagate from the
    z back to mu and sigma. We can achieve this by first sampling a random value
    epsilon from a standard Gaussian distribution with zero mean and unit variance,
    then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network,
    it helps to pass this function the log of the variance of the distribution from
    which to sample, rather than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns:
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a
      Gaussian with mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    ###############################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and    #
    # scaling by posterior mu and sigma to estimate z                             #
    ###############################################################################
    # Replace "pass" statement with your code
    '''
    为什么需要重参数化？
        打破随机采样与梯度的矛盾
            直接采样（如torch.normal(mu, sigma)）会阻断梯度传播
            重参数化将随机性转移到外部噪声eps，保持计算路径可导
            mu和logvar是直接参与计算的参数，梯度可通过z反向传播
        训练稳定性
            对数方差logvar（即log(sigma^2)）比直接优化sigma更稳定（梯度更平滑）,可避免：
                方差计算中出现负值（logvar无约束）
                数值不稳定（如sigma接近零时）
            可通过约束logvar的范围（如clamp）控制方差的下限
    '''
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return z


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to
    formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space
      dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z
      latent space dimension

    Returns:
    - loss: Tensor containing the scalar loss for the negative variational
      lowerbound
    """
    loss = None
    ###############################################################################
    # TODO: Compute negative variational lowerbound loss as described in the      #
    # notebook                                                                    #
    ###############################################################################
    # Replace "pass" statement with your code
    '''
    torch.nn.functional.binary_cross_entropy(
        input,          # 模型预测值（需经过Sigmoid，范围[0,1]）
        target,         # 真实标签（0或1，或[0,1]间的概率值）
        weight=None,    # 每个样本的权重（可选）
        reduction='mean' # 损失聚合方式：'mean'|'sum'|'none'
    )
    reduction='mean'：返回批量的平均损失（默认）
    reduction='sum'：返回批量的总损失
    reduction='none'：返回每个样本的独立损失（形状与 input 相同）
    '''
    reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KL_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = reconstruction_loss + KL_divergence_loss
    ###############################################################################
    #                            END OF YOUR CODE                                 #
    ###############################################################################
    return loss
