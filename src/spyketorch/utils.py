import torch
import torch.nn.functional as fn
import numpy as np
import math
import os


def to_pair(data):
    r"""Converts a single or a tuple of data into a pair. If the data is a tuple with more than two elements, it selects
    the first two of them. In case of single data, it duplicates that data into a pair.
    Args:
            data (object or tuple): The input data.
    Returns:
            Tuple: A pair of data.
    """
    if isinstance(data, tuple):
        return data[0:2]
    return (data, data)


def generate_inhibition_kernel(inhibition_percents):
    r"""Generates an inhibition kernel suitable to be used by :func:`~functional.intensity_lateral_inhibition`.
    Args:
            inhibition_percents (sequence): The sequence of inhibition factors (in range [0,1]).
    Returns:
            Tensor: Inhibition kernel.
    """
    inhibition_kernel = torch.zeros(
        2*len(inhibition_percents)+1, 2*len(inhibition_percents)+1).float()
    center = len(inhibition_percents)
    for i in range(2*len(inhibition_percents)+1):
        for j in range(2*len(inhibition_percents)+1):
            dist = int(max(math.fabs(i - center), math.fabs(j - center)))
            if dist != 0:
                inhibition_kernel[i, j] = inhibition_percents[dist - 1]
    return inhibition_kernel


def tensor_to_text(data, address):
    r"""Saves a tensor into a text file in row-major format. The first line of the file contains comma-separated integers denoting
    the size of each dimension. The second line contains comma-separated values indicating all the tensor's data.
    Args:
            data (Tensor): The tensor to be saved.
            address (str): The saving address.
    """
    f = open(address, "w")
    data_cpu = data.cpu()
    shape = data.shape
    print(",".join(map(str, shape)), file=f)
    data_flat = data_cpu.view(-1).numpy()
    print(",".join(data_flat.astype(np.str)), file=f)
    f.close()


def text_to_tensor(address, type='float'):
    r"""Loads a tensor from a text file. Format of the text file is as follows: The first line of the file contains comma-separated integers denoting
    the size of each dimension. The second line contains comma-separated values indicating all the tensor's data.
    Args:
            address (str): Address of the text file.
            type (float or int, optional): The type of the tensor's data ('float' or 'int'). Default: 'float'
    Returns:
            Tensor: The loaded tensor.
    """
    f = open(address, "r")
    shape = tuple(map(int, f.readline().split(",")))
    data = np.array(f.readline().split(","))
    if type == 'float':
        data = data.astype(np.float32)
    elif type == 'int':
        data = data.astype(np.int32)
    else:
        raise ValueError("type must be 'int' or 'float'")
    data = torch.from_numpy(data)
    data = data.reshape(shape)
    f.close()
    return data


class LateralIntencityInhibition:
    r"""Applies lateral inhibition on intensities. For each location, this inhibition decreases the intensity of the
    surrounding cells that has lower intensities by a specific factor. This factor is relative to the distance of the
    neighbors and are put in the :attr:`inhibition_percents`.
    Args:
            inhibition_percents (sequence): The sequence of inhibition factors (in range [0,1]).
    """

    def __init__(self, inhibition_percents):
        self.inhibition_kernel = generate_inhibition_kernel(
            inhibition_percents)
        self.inhibition_kernel.unsqueeze_(0).unsqueeze_(0)

    # decrease lateral intencities by factors given in the inhibition_kernel
    def intensity_lateral_inhibition(self, intencities):
        intencities.squeeze_(0)
        intencities.unsqueeze_(1)

        inh_win_size = self.inhibition_kernel.size(-1)
        rad = inh_win_size//2
        # repeat each value
        values = intencities.reshape(
            intencities.size(0), intencities.size(1), -1, 1)
        values = values.repeat(1, 1, 1, inh_win_size)
        values = values.reshape(intencities.size(0), intencities.size(
            1), -1, intencities.size(-1)*inh_win_size)
        values = values.repeat(1, 1, 1, inh_win_size)
        values = values.reshape(intencities.size(0), intencities.size(
            1), -1, intencities.size(-1)*inh_win_size)
        # extend patches
        padded = fn.pad(intencities, (rad, rad, rad, rad))
        # column-wise
        patches = padded.unfold(-1, inh_win_size, 1)
        patches = patches.reshape(patches.size(0), patches.size(
            1), patches.size(2), -1, patches.size(3)*patches.size(4))
        patches.squeeze_(-2)
        # row-wise
        patches = patches.unfold(-2, inh_win_size, 1).transpose(-1, -2)
        patches = patches.reshape(patches.size(
            0), patches.size(1), 1, -1, patches.size(-1))
        patches.squeeze_(-3)
        # compare each element by its neighbors
        coef = values - patches
        coef.clamp_(min=0).sign_()  # "ones" are neighbors greater than center
        # convolution with full stride to get accumulative inhibiiton factor
        factors = fn.conv2d(coef, self.inhibition_kernel, stride=inh_win_size)
        result = intencities + intencities * factors

        intencities.squeeze_(1)
        intencities.unsqueeze_(0)
        result.squeeze_(1)
        result.unsqueeze_(0)
        return result

    def __call__(self, input):
        return self.intensity_lateral_inhibition(input)


class FilterKernel:
    r"""Base class for generating image filter kernels such as Gabor, DoG, etc. Each subclass should override :attr:`__call__` function.
    """

    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self):
        pass


class DoGKernel(FilterKernel):
    r"""Generates DoG filter kernel.
    Args:
            window_size (int): The size of the window (square window).
            sigma1 (float): The sigma for the first Gaussian function.
            sigma2 (float): The sigma for the second Gaussian function.
    """

    def __init__(self, window_size, sigma1, sigma2):
        super(DoGKernel, self).__init__(window_size)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    # TODO: Refactor
    # returns a 2d tensor corresponding to the requested DoG filter
    def __call__(self):
        w = self.window_size // 2
        x, y = np.mgrid[-w:w+1:1, -w:w+1:1]

        prod = x*x + y*y

        f1 = np.exp(-0.5 * prod / (self.sigma1**2)) / (self.sigma1**2)
        f2 = np.exp(-0.5 * prod / (self.sigma2**2)) / (self.sigma2**2)

        dog = (f1-f2) / (2 * math.pi)
        dog = (dog - np.mean(dog)) / np.max(dog)
        dog_tensor = torch.from_numpy(dog)
        return dog_tensor.float()


class GaborKernel(FilterKernel):
    r"""Generates Gabor filter kernel.
    Args:
            window_size (int): The size of the window (square window).
            orientation (float): The orientation of the Gabor filter (in degrees).
            div (float, optional): The divisor of the lambda equation. Default: 4.0
    """

    def __init__(self, window_size, orientation, div=4.0):
        super(GaborKernel, self).__init__(window_size)
        self.orientation = orientation
        self.div = div

    # returns a 2d tensor corresponding to the requested Gabor filter
    def __call__(self):
        w = self.window_size//2
        x, y = np.mgrid[-w:w+1:1, -w:w+1:1]

        wave_length = self.window_size * 2 / self.div
        sigmaSquare = (wave_length * 0.8) ** 2
        aspect_ratioSquare = 0.3 ** 2
        theta = (self.orientation * np.pi) / 180

        Y = y*np.cos(theta) - x*np.sin(theta)
        X = y*np.sin(theta) + x*np.cos(theta)

        gabor = np.exp(-0.5 * (X**2 + aspect_ratioSquare * Y**2) /
                       sigmaSquare) * np.cos(2*np.pi/wave_length * X)
        # normalize
        gabor = (gabor - np.mean(gabor)) / np.max(gabor)

        return torch.from_numpy(gabor).float()


class Filter:
    r"""Applies a filter transform. Each filter contains a sequence of :attr:`FilterKernel` objects.
    The result of each filter kernel will be passed through a given threshold (if not :attr:`None`).
    Args:
            filter_kernels (sequence of FilterKernels): The sequence of filter kernels.
            padding (int, optional): The size of the padding for the convolution of filter kernels. Default: 0
            thresholds (sequence of floats, optional): The threshold for each filter kernel. Default: None
            use_abs (boolean, optional): To compute the absolute value of the outputs or not. Default: False
    .. note::
            The size of the compund filter kernel tensor (stack of individual filter kernels) will be equal to the 
            greatest window size among kernels. All other smaller kernels will be zero-padded with an appropriate 
            amount.
    """
    # filter_kernels must be a list of filter kernels
    # thresholds must be a list of thresholds for each kernel

    def __init__(self, filter_kernels, padding=0, thresholds=None, use_abs=False):
        tensor_list = []
        self.max_window_size = 0
        for kernel in filter_kernels:
            if isinstance(kernel, torch.Tensor):
                tensor_list.append(kernel)
                self.max_window_size = max(
                    self.max_window_size, kernel.size(-1))
            else:
                tensor_list.append(kernel().unsqueeze(0))
                self.max_window_size = max(
                    self.max_window_size, kernel.window_size)
        for i in range(len(tensor_list)):
            p = (self.max_window_size - filter_kernels[i].window_size)//2
            tensor_list[i] = fn.pad(tensor_list[i], (p, p, p, p))

        self.kernels = torch.stack(tensor_list)
        self.number_of_kernels = len(filter_kernels)
        self.padding = padding
        if isinstance(thresholds, list):
            self.thresholds = thresholds.clone().detach()
            self.thresholds.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
        else:
            self.thresholds = thresholds
        self.use_abs = use_abs

        # without kernel
        # # ------------------------------------------
        #         self.max_size = max(0, *(kernel.size for kernel in kernels))
        #         tensors = [kernel().unsqueeze(0) for kernel in kernels]
        #         # zero padding to filters
        #         for i in range(len(tensors)):
        #             padding = (self.max_size - kernels[i].size) // 2
        #             tensors[i] = fn.pad(tensors[i], pad=(padding,)*4, value=0)

        #         self.kernels = torch.stack(tensors)
        #         self.only_positive = only_positive
        # # ------------------------------------------

    # returns a 4d tensor containing the flitered versions of the input image
    # input is a 4d tensor. dim: (minibatch=1, filter_kernels, height, width)
    def __call__(self, input):
        output = fn.conv2d(input, self.kernels, padding=self.padding).float()
        if not(self.thresholds is None):
            output = torch.where(output < self.thresholds, torch.tensor(
                0.0, device=output.device), output)
        if self.use_abs:
            torch.abs_(output)
        return output


class IntensityToLatency:
    def __init__(self, number_of_spike_bins, to_spike=False):
        self.time_steps = number_of_spike_bins
        self.to_spike = to_spike

    # intensities is a tensor of input intensities (batchsize, kernels, height, width)
    # returns a tensor of tensors containing spikes in each timestep (considers minibatch for timesteps)
    # spikes are accumulative, i.e. spikes in timestep i are also presented in i+1, i+2, ...
    def __call__(self, batches):
        bins_intensities = []
        for image in batches:
            images_intensities = []
            # check for empty bins (nonzero indies from intensities)
            bin_size = torch.nonzero(image).size()[0]
            bin_size = bin_size // self.time_steps

            # flatten
            intensities = torch.reshape(image, (-1,))
            # sort
            intensities = torch.sort(intensities, descending=True)
            # bin packing for sorted intensities
            bins_value = torch.split(intensities[0], bin_size)
            bins_idx = torch.split(intensities[1], bin_size)
            # add to the list of timesteps
            spike_map = torch.zeros_like(intensities[0])

            for i in range(self.time_steps):
                spike_map.scatter_(0, bins_idx[i], bins_value[i])
                spike_map_copy = spike_map.clone().detach()
                spike_map_copy = spike_map_copy.reshape(tuple(image.shape))
                images_intensities.append(spike_map_copy.squeeze(0).float())
            bins_intensities.append(torch.stack(images_intensities))
        if self.to_spike:
            return torch.stack(bins_intensities).sign()
        return torch.stack(bins_intensities)


class CacheDataset(torch.utils.data.Dataset):
    r"""A wrapper dataset to cache pre-processed data. It can cache data on RAM or a secondary memory.
    .. note::
            Since converting image into spike-wave can be time consuming, we recommend to wrap your dataset into a :attr:`CacheDataset`
            object.
    Args:
            dataset (torch.utils.data.Dataset): The reference dataset object.
            cache_address (str, optional): The location of cache in the secondary memory. Use :attr:`None` to cache on RAM. Default: None
    """

    def __init__(self, dataset, cache_address=None):
        self.dataset = dataset
        self.cache_address = cache_address
        self.cache = [None] * len(self.dataset)

    def __getitem__(self, index):
        if self.cache[index] is None:
            # cache it
            sample, target = self.dataset[index]
            if self.cache_address is None:
                self.cache[index] = sample, target
            else:
                save_path = os.path.join(self.cache_address, str(index))
                torch.save(sample, save_path + ".cd")
                torch.save(target, save_path + ".cl")
                self.cache[index] = save_path
        else:
            if self.cache_address is None:
                sample, target = self.cache[index]
            else:
                sample = torch.load(self.cache[index] + ".cd")
                target = torch.load(self.cache[index] + ".cl")
        return sample, target

    def reset_cache(self):
        r"""Clears the cached data. It is useful when you want to change a pre-processing parameter during
        the training process.
        """
        if self.cache_address is not None:
            for add in self.cache:
                os.remove(add + ".cd")
                os.remove(add + ".cl")
        self.cache = [None] * len(self)

    def __len__(self):
        return len(self.dataset)
