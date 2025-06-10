"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
from typing import Dict, List


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses a naive set of nested loops over the training and
    test data.

    The input data may have any number of dimensions -- for example this
    function should be able to compute nearest neighbor between vectors, in
    which case the inputs will have shape (num_{train, test}, D); it should
    also be able to compute nearest neighbors between images, where the inputs
    will have shape (num_{train, test}, C, H, W). More generally, the inputs
    will have shape (num_{train, test}, D1, D2, ..., Dn); you should flatten
    each element of shape (D1, D2, ..., Dn) into a vector of shape
    (D1 * D2 * ... * Dn) before computing distances.

    The input tensors should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using a pair of nested loops over the    #
    # training data and the test data.                                       #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # Replace "pass" statement with your code
    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat = x_test.reshape(num_test, -1)
    
    for i in range(num_train):
        for j in range(num_test):
            dists[i, j] = (x_train_flat[i] - x_test_flat[j]).pow(2).sum()
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation uses only a single loop over the training data.

    Similar to `compute_distances_two_loops`, this should be able to handle
    inputs with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, D1, D2, ...)
        x_test: Tensor of shape (num_test, D1, D2, ...)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j]
            is the squared Euclidean distance between the i-th training point
            and the j-th test point. It should have the same dtype as x_train.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function using only a single loop over x_train.   #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    ##########################################################################
    # Replace "pass" statement with your code
    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat = x_test.reshape(num_test, -1)
    
    for i in range(num_train):
        dists[i] = (x_test_flat - x_train_flat[i]).pow(2).sum(dim=1)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    """
    Computes the squared Euclidean distance between each element of training
    set and each element of test set. Images should be flattened and treated
    as vectors.

    This implementation should not use any Python loops. For memory-efficiency,
    it also should not create any large intermediate tensors; in particular you
    should not create any intermediate tensors with O(num_train * num_test)
    elements.

    Similar to `compute_distances_two_loops`, this should be able to handle
    inputs with any number of dimensions. The inputs should not be modified.

    NOTE: Your implementation may not use `torch.norm`, `torch.dist`,
    `torch.cdist`, or their instance method variants (`x.norm`, `x.dist`,
    `x.cdist`, etc.). You may not use any functions from `torch.nn` or
    `torch.nn.functional` modules.

    Args:
        x_train: Tensor of shape (num_train, C, H, W)
        x_test: Tensor of shape (num_test, C, H, W)

    Returns:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is
            the squared Euclidean distance between the i-th training point and
            the j-th test point.
    """
    # Initialize dists to be a tensor of shape (num_train, num_test) with the
    # same datatype and device as x_train
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    ##########################################################################
    # TODO: Implement this function without using any explicit loops and     #
    # without creating any intermediate tensors with O(num_train * num_test) #
    # elements.                                                              #
    #                                                                        #
    # You may not use torch.norm (or its instance method variant), nor any   #
    # functions from torch.nn or torch.nn.functional.                        #
    #                                                                        #
    # HINT: Try to formulate the Euclidean distance using two broadcast sums #
    #       and a matrix multiply.                                           #
    ##########################################################################
    # Replace "pass" statement with your code
    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat = x_test.reshape(num_test, -1)
    ################################NOTES#####################################
    # dists[i, j] = ||x_train[i] - x_test[j]||^2 
    #        = ||x_train[i]||^2 + ||x_test[j]||^2 - 2 * x_train[i] @ x_test[j]
    ################################NOTES#####################################
    train_norm = torch.sum(x_train_flat ** 2, dim=1, keepdim=True)  # 为了广播
    test_norm = torch.sum(x_test_flat ** 2, dim=1)                  
    dot_product = torch.mm(x_train_flat, x_test_flat.t())           

    dists = train_norm + test_norm - 2 * dot_product
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return dists


def predict_labels_me(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    """
    Given distances between all pairs of training and test samples, predict a
    label for each test sample by taking a MAJORITY VOTE among its `k` nearest
    neighbors in the training set.

    In the event of a tie, this function SHOULD return the smallest label. For
    example, if k=5 and the 5 nearest neighbors to a test example have labels
    [1, 2, 1, 2, 3] then there is a tie between 1 and 2 (each have 2 votes),
    so we should return 1 since it is the smallest label.

    This function should not modify any of its inputs.

    Args:
        dists: Tensor of shape (num_train, num_test) where dists[i, j] is the
            squared Euclidean distance between the i-th training point and the
            j-th test point.
        y_train: Tensor of shape (num_train,) giving labels for all training
            samples. Each label is an integer in the range [0, num_classes - 1]
        k: The number of nearest neighbors to use for classification.

    Returns:
        y_pred: int64 Tensor of shape (num_test,) giving predicted labels for
            the test data, where y_pred[j] is the predicted label for the j-th
            test example. Each label should be an integer in the range
            [0, num_classes - 1].
    """
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64)
    ##########################################################################
    # TODO: Implement this function. You may use an explicit loop over the   #
    # test samples.                                                          #
    #                                                                        #
    # HINT: Look up the function torch.topk                                  #
    ##########################################################################
    # Replace "pass" statement with your code
    ################################NOTES#####################################
    # torch.topk(input, k, dim=None, largest=True, sorted=True)
    # 用于返回张量中沿着指定维度的前k个最值及其对应的索引。
    # 参数说明：
    #    input (Tensor): 输入张量。
    #    k (int): 需要返回的最大值（或最小值）的数量。
    #    dim (int, optional): 沿着哪个维度进行操作。默认沿最后一个维度进行操作。
    #    largest (bool, optional): True，最大值；False，最小值。
    #    sorted (bool, optional): 为 True时返回的结果会按照顺序排列。
    ################################NOTES#####################################
    _, indices = torch.topk(dists, k, dim=0, largest=False)
    
    for i in range(num_test):
        mask = torch.zeros(num_test)
        
        for j in range(num_test):
            mask[y_train[indices[:, i]][j]] += 1
        _, ind = torch.topk(mask, 1)
        y_pred[i] = ind
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return y_pred


def predict_labels_deepseek(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64)

    # 找到每个测试样本的 k 个最近邻的索引
    _, indices = torch.topk(dists, k=k, dim=0, largest=False)  # 形状: (k, num_test)

    # 获取这 k 个最近邻的标签
    k_nearest_labels = y_train[indices]  # 形状: (k, num_test)
    
    # 对每个测试样本的 k 个最近邻标签进行投票
    for i in range(num_test):
        # 获取第 i 个测试样本的 k 个最近邻标签
        labels = k_nearest_labels[:, i]

        # 统计每个标签的出现次数
        ################################NOTES#####################################
        # torch.bincount(input, weights=None, minlength=0), 用于统计非负整数张量中
        # 每个值的出现次数。是一个非常高效的函数(O(n))，特别适合处理分类任务中的标签统计问题。
        # 参数说明：
        #    input (Tensor): 输入张量，必须是一维的且包含非负整数。
        #    weights (Tensor, optional): 可选参数，与 input 形状相同的权重张量。
        #        如果提供，bincount() 会返回加权和而不是简单的计数。
        #    minlength (int, optional): 输出张量的最小长度。如果 input 中的最大值加 1 
        #        小于 minlength，则输出张量会被填充到 minlength 的长度。
        # 返回值：
        #    output (Tensor): 一个一维张量，表示每个非负整数的出现次数（或加权和）。
        #        张量的长度为 max(input) + 1 或 minlength，取两者中的较大值。
        ################################NOTES#####################################
        counts = torch.bincount(labels)

        # 找到票数最多的标签
        ################################NOTES#####################################
        # torch.where(condition, x, y), 用于根据条件从两个张量中选择元素。
        # 可以根据条件返回满足条件的元素的位置，或者从两个张量中选择元素。
        # 参数说明：
        #    condition (Tensor): 条件张量，布尔类型（True 或 False）。
        #    x (Tensor): 当 condition 为 True 时选择的元素。
        #    y (Tensor): 当 condition 为 False 时选择的元素。
        # 返回值：
        #    output (Tensor): 一个张量，形状与 x 和 y 广播后的形状相同。
        #    output 中的每个元素根据 condition 的值从 x 或 y 中选择。
        ################################NOTES#####################################
        max_count = torch.max(counts)
        candidate_labels = torch.where(counts == max_count)[0]

        # 如果出现平局，选择最小的标签
        y_pred[i] = torch.min(candidate_labels)
        
    return y_pred


class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """
        Create a new K-Nearest Neighbor classifier with the specified training
        data. In the initializer we simply memorize the provided training data.

        Args:
            x_train: Tensor of shape (num_train, C, H, W) giving training data
            y_train: int64 Tensor of shape (num_train, ) giving training labels
        """
        ######################################################################
        # TODO: Implement the initializer for this class. It should perform  #
        # no computation and simply memorize the training data in            #
        # `self.x_train` and `self.y_train`, accordingly.                    #
        ######################################################################
        # Replace "pass" statement with your code
        self.x_train = x_train
        self.y_train = y_train
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################

    def predict(self, x_test: torch.Tensor, k: int = 1):
        """
        Make predictions using the classifier.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            k: The number of neighbors to use for predictions.

        Returns:
            y_test_pred: Tensor of shape (num_test,) giving predicted labels
                for the test samples.
        """
        y_test_pred = None
        ######################################################################
        # TODO: Implement this method. You should use the functions you      #
        # wrote above for computing distances (use the no-loop variant) and  #
        # to predict output labels.                                          #
        ######################################################################
        # Replace "pass" statement with your code
        dists = compute_distances_no_loops(self.x_train, x_test)
        
        y_test_pred = predict_labels_deepseek(dists, self.y_train, k)
        ######################################################################
        #                         END OF YOUR CODE                           #
        ######################################################################
        return y_test_pred

    def check_accuracy(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        k: int = 1,
        quiet: bool = False
    ):
        """
        Utility method for checking the accuracy of this classifier on test
        data. Returns the accuracy of the classifier on the test data, and
        also prints a message giving the accuracy.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            y_test: int64 Tensor of shape (num_test,) giving test labels.
            k: The number of neighbors to use for prediction.
            quiet: If True, don't print a message.

        Returns:
            accuracy: Accuracy of this classifier on the test data, as a
                percent. Python float in the range [0, 100]
        """
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy


def knn_cross_validate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_folds: int = 5,
    k_choices: List[int] = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100],
):
    """
    Perform cross-validation for `KnnClassifier`.

    Args:
        x_train: Tensor of shape (num_train, C, H, W) giving all training data.
        y_train: int64 Tensor of shape (num_train,) giving labels for training
            data.
        num_folds: Integer giving the number of folds to use.
        k_choices: List of integers giving the values of k to try.

    Returns:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.
    """

    # First we divide the training data into num_folds equally-sized folds.
    x_train_folds = []
    y_train_folds = []
    ##########################################################################
    # TODO: Split the training data and images into folds. After splitting,  #
    # x_train_folds and y_train_folds should be lists of length num_folds,   #
    # where y_train_folds[i] is label vector for images inx_train_folds[i].  #
    #                                                                        #
    # HINT: torch.chunk                                                      #
    ##########################################################################
    # Replace "pass" statement with your code
    ################################NOTES#####################################
    # torch.chunk(input, chunks, dim=0), 用于将张量沿着指定维度分割成多个块。
    # 每个块的大小尽可能相等，但如果张量的大小不能被块数整除，最后一个块可能会较小。
    # 参数说明：
    #     input (Tensor): 输入张量。
    #     chunks (int): 需要将张量分割成的块数。
    #     dim (int, optional): 沿着哪个维度进行分割。默认为 0（第 0 维）。
    # 返回值：
    #     output (tuple of Tensors): 一个包含分割后张量的元组。元组的长度为 chunks。
    ################################NOTES#####################################
    x_train_folds = torch.chunk(x_train, chunks=num_folds)
    y_train_folds = torch.chunk(y_train, chunks=num_folds)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    # A dictionary holding the accuracies for different values of k that we
    # find when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the
    # different accuracies we found trying `KnnClassifier`s using k neighbors.
    k_to_accuracies = {}

    ##########################################################################
    # TODO: Perform cross-validation to find the best value of k. For each   #
    # value of k in k_choices, run the k-NN algorithm `num_folds` times; in  #
    # each case you'll use all but one fold as training data, and use the    #
    # last fold as a validation set. Store the accuracies for all folds and  #
    # all values in k in k_to_accuracies.                                    #
    #                                                                        #
    # HINT: torch.cat                                                        #
    ##########################################################################
    # Replace "pass" statement with your code
    k_to_accuracies = {k: [] for k in k_choices}
    for k in k_choices:
        for i in range(num_folds):
            data_val = x_train_folds[i]
            label_val = y_train_folds[i]

            data_train = torch.cat([x_train_folds[j] for j in range(num_folds) if j != i]) 
            label_train = torch.cat([y_train_folds[j] for j in range(num_folds) if j != i])
            
            classifier = KnnClassifier(data_train, label_train)
            accuracy = classifier.check_accuracy(data_val, label_val, k=k, quiet=True)
            k_to_accuracies[k].append(accuracy)
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################

    return k_to_accuracies


def knn_get_best_k(k_to_accuracies: Dict[int, List]):
    """
    Select the best value for k, from the cross-validation result from
    knn_cross_validate. If there are multiple k's available, then you SHOULD
    choose the smallest k among all possible answer.

    Args:
        k_to_accuracies: Dictionary mapping values of k to lists, where
            k_to_accuracies[k][i] is the accuracy on the i-th fold of a
            `KnnClassifier` that uses k nearest neighbors.

    Returns:
        best_k: best (and smallest if there is a conflict) k value based on
            the k_to_accuracies info.
    """
    best_k = 0
    best_accuracy = -1
    ##########################################################################
    # TODO: Use the results of cross-validation stored in k_to_accuracies to #
    # choose the value of k, and store result in `best_k`. You should choose #
    # the value of k that has the highest mean accuracy accross all folds.   #
    ##########################################################################
    # Replace "pass" statement with your code
    for k, accs in k_to_accuracies.items():
        mean_accuracy = torch.tensor(accs).mean().item()

        if mean_accuracy > best_accuracy:
            best_k = k
            best_accuracy = mean_accuracy
        elif mean_accuracy == best_accuracy and k < best_k:
            best_k = k
    ##########################################################################
    #                           END OF YOUR CODE                             #
    ##########################################################################
    return best_k