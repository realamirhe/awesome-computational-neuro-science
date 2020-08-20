import itertools

import numpy as np
import torch

import matplotlib.pyplot as plt
from .helpers import rdm


# Show 2D the tensor.
def show_tensor(aTensor, _vmin=None, _vmax=None):
    r"""Plots a 2D tensor in gray color map and shows it in a window.
    Args:
            aTensor (Tensor): The input tensor.
            _vmin (float, optional): Minimum value. Default: None
            _vmax (float, optional): Maximum value. Default: None
    .. note::
            :attr:`None` for :attr:`_vmin` or :attr:`_vmin` causes an auto-scale mode for each.
    """
    if aTensor.is_cuda:
        aTensor = aTensor.cpu()
    plt.figure()
    plt.imshow(aTensor.numpy(), cmap='gray', vmin=_vmin, vmax=_vmax)
    plt.colorbar()
    plt.show()


def plot_tensor_in_image(fname, aTensor, _vmin=None, _vmax=None):
    r"""Plots a 2D tensor in gray color map in an image file.
    Args:
            fname (str): The file name.
            aTensor (Tensor): The input tensor.
            _vmin (float, optional): Minimum value. Default: None
            _vmax (float, optional): Maximum value. Default: None
    .. note::
            :attr:`None` for :attr:`_vmin` or :attr:`_vmin` causes an auto-scale mode for each.
    """
    if aTensor.is_cuda:
        aTensor = aTensor.cpu()
    plt.imsave(fname, aTensor.numpy(), cmap='gray', vmin=_vmin, vmax=_vmax)

# Computes window size of a neuron on specific previous layer
# layer_details must be a sequence of quaraples containing (height, width, row_stride, col_stride)
# of each layer


def get_deep_receptive_field(*layers_details):
    h, w = 1, 1
    for height, width, r_stride, c_stride in reversed(layers_details):
        h = height + (h-1) * r_stride
        w = width + (w-1) * c_stride
    return h, w

# Computes the feature that a neuron is selective to given the feature of the neurons underneath
# The cumulative stride (which is cumulative product of previous layers' strides) must be given
# The stride of the previous layer must be given
# pre_feature is the 3D tensor of the features for underlying neurons
# feature_stride is the cumulative stride (tuple) = (height, width)
# stride is the stride of previous layer (tuple) = (height, width)
# weights is the 4D weight tensor of current layer (None if it is a pooling layer)
# retruns features and the new cumulative stride


def get_deep_feature(pre_feature, feature_stride, window_size, stride, weights=None):
    new_cstride = (
        feature_stride[0] * stride[0],
        feature_stride[1] * stride[1]
    )
    new_size = (
        pre_feature.size(-2) + (window_size[0]-1) * feature_stride[0],
        pre_feature.size(-1) + (window_size[1]-1) * feature_stride[1]
    )

    depth = weights.size(0) if weights is not None else pre_feature.size(-3)

    new_feature = torch.zeros(depth, *new_size, device=pre_feature.device)
    if weights is None:  # place the feature in the middle of the field
        sh = new_size[0]//2 - pre_feature.size(-2)//2
        sw = new_size[1]//2 - pre_feature.size(-1)//2
        new_feature[:,
                    sh:sh + pre_feature.size(-2),
                    sw:sw + pre_feature.size(-1)] = pre_feature
    else:
        for r in range(weights.size(-2)):  # rows
            for c in range(weights.size(-1)):  # cols
                temp_features = pre_feature * weights[:, :, r:r+1, c:c+1]
                temp_features = temp_features.max(dim=1)[0]
                s0 = r*feature_stride[0]
                s1 = c*feature_stride[1]
                new_feature[:,
                            s0:s0+pre_feature.size(-2),
                            s1:s1+pre_feature.size(-1)] += temp_features
        new_feature.clamp_(min=0)  # removing negatives

    return new_feature, new_cstride


# confusion matrix
# given a sklearn confusion matrix (cm), make a nice plot
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()

# from torchvision.utils import save_image


def feature_selection_plot(features, ROW, COL):
    for r in range(ROW):
        for c in range(COL):
            idx = r*COL + c
            ax = plt.subplot(ROW, COL, idx + 1)
            plt.xticks([])
            plt.yticks([])
            plt.setp(ax, xticklabels=[])
            plt.setp(ax, yticklabels=[])
            plt.imshow(features[idx].numpy(), cmap='gray')
    plt.show()


def rdm_plot(matrix):
    plt.matshow(rdm(matrix))
    plt.xticks([])
    plt.yticks([])
    plt.show()
