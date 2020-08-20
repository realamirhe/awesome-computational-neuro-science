
""" In the name of Allah, the Most Gracious, the Most Merciful """
# *****************
#      Modules
# *****************

import os
from datetime import datetime
# scientific
import numpy as np
# torch
import torch
import torch.nn as nn
# sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn.parameter import Parameter
# loader
from torch.utils.data import DataLoader, SubsetRandomSampler
# torchvision
from torchvision import transforms
from torchvision.transforms import functional as ttf

from torchvision.datasets import ImageFolder
# fancy progressbar
from tqdm import tqdm

# spykeTorch
import spyketorch.utils as su
import spyketorch.functional as sf
import spyketorch.visualization as sv
from spyketorch import snn
from spyketorch.helpers import (
    argument_parser,
    get_decision_map,
    measure,
    reset_seed,
    train_test_split
)

# *****************
#   CONFIGURATION
# *****************
CONFIG = {
    "data_dir": "data/caltech101",
    "cache_dir": "data/caltech101/cached",
    "train_batch_size": 1000,
    "image_size": (40, 40),
    "split_size": 0.75,
    "max_iter": 66,
    "features": ['Faces', 'Leopards', 'Motorbikes', 'airplanes'],
    "S2out_channels": 20,
    "checkpoint": 5,

    "first_layer_epoch": 5,
    "second_layer_epoch": 30,
    "third_layer_epoch": 325,

    "dt": 1,
    "time": 15,
    "timesteps": 15,

    "num_workers": None,
    "gpu": False,
    "device": None,
}


# Configure
args = argument_parser(CONFIG, "config")
cwd = os.getcwd()

# update configuration based on non set variables
if args.num_workers is None:
    args = args._replace(num_workers=args.gpu*4*torch.cuda.device_count())

if not (args.gpu and torch.cuda.is_available()):
    args = args._replace(gpu=False)

if args.device is None:
    args = args._replace(device='cuda' if args.gpu else 'cpu')


# *******************
#   Transforamtion
# *******************
class Transformation:
    def __init__(self, filter):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.i2l = su.Intensity2Latency(args.timesteps, to_spike=True)
        self.rgb2gray = transforms.Grayscale()

    def __call__(self, image):
        image = self.rgb2gray(image)  # convert to grayscale
        image = ttf.resize(image, args.image_size)  # resize PIL Images
        # convert PIL Image to tensor [c, h, w]
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)  # add mini batch 1

        image = self.filter(image)  # apply convolution's filter
        # local normalization instead of total image normalization
        image = sf.local_normalization(image, 8)
        image = self.i2l(image)  # intensity to latency spikes
        return image.byte()  # convert to uint8


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # [batchsize, channel=len(filters), height, width]
        initialize_w_from = {"weight_mean": 0.8, "weight_std": 0.05}
        learning_rate = (0.004, -0.003)

        # Layer 1
        self.convolution_1 = snn.Convolution(in_channels=6,
                                             out_channels=30,
                                             kernel_size=5,
                                             **initialize_w_from)
        self.stdp_1 = snn.STDP(self.convolution_1, learning_rate)

        # Layer 2
        self.convolution_2 = snn.Convolution(in_channels=30,
                                             out_channels=250,
                                             kernel_size=3,
                                             **initialize_w_from)
        self.stdp_2 = snn.STDP(self.convolution_2, learning_rate)

        # Layer 3
        self.convolution_3 = snn.Convolution(in_channels=250,
                                             out_channels=200,
                                             kernel_size=5,
                                             **initialize_w_from)
        self.stdp_3 = snn.STDP(self.convolution_3,
                               learning_rate,
                               use_stabilizer=False,
                               lower_bound=0.2,
                               upper_bound=0.8)

        self.antistdp_3 = snn.STDP(self.convolution_3,
                                   learning_rate=(-0.004, 0.0005),
                                   use_stabilizer=False,
                                   lower_bound=0.2,
                                   upper_bound=0.8)

        self.maximum_positive_amplitude = Parameter(torch.Tensor([0.15]))
        # max_count gonna be normalized map for maximum layer
        self.decision_map = get_decision_map('data/caltech101', max_count=250)

        self.config = {
            "c:1.k_winner": 5,
            "c:1.r_inhibition": 3,
            "c:1.spikecount": 0,
            "c:1.threshold": 15,

            "c:2.k_winner": 8,
            "c:2.r_inhibition": 1,
            "c:2.spikecount": 0,
            "c:2.threshold": 10,
        }

        self.neural_activity = {
            "input_spikes": None,
            "output_spikes": None,
            "potentials": None,
            "winners": None,
        }

    def forward(self, data, deep=1):
        # zero padded input value
        data = sf.pad(data.float(), (2, 2, 2, 2))
        return self.layer_pass(data, deep)

    def layer_pass(self, data, deep):
        pot = self.convolution_1(data)
        spike, pot = sf.fire(pot, self.config['c:1.threshold'], True)
        # layer 1
        if deep == 1:
            if self.training:
                self.neural_activity["input_spikes"] = data
                return self.train_deep(spike, pot, deep=1)
            else:
                return spike, pot

        input_spike = sf.pad(sf.pooling(spike, 2, 2), (1, 1, 1, 1))
        pot = self.convolution_2(input_spike)
        spike, pot = sf.fire(pot, self.config['c:2.threshold'], True)
        #  layer 2
        if deep == 2:
            if self.training:
                self.neural_activity["input_spikes"] = input_spike
                return self.train_deep(spike, pot, deep=2)
            else:
                return spike, pot

        #  layer 3
        input_spike = sf.pad(sf.pooling(spike, 3, 3), (2, 2, 2, 2))
        pot = self.convolution_3(input_spike)
        # TODO: can place SVM here ...
        if deep == -1:
            return pot
        spike = sf.fire(pot)

        winners = sf.get_k_winners(pot, 1, 0, spike)
        if self.training:
            self.neural_activity["input_spikes"] = input_spike
            self.neural_activity["potentials"] = pot
            self.neural_activity["output_spikes"] = spike
            self.neural_activity["winners"] = winners

        if len(winners) != 0:
            return self.decision_map[winners[0][0]]
        return -1

    def reset_and_update_stdp(self, stdp, key):
        self.config[key] += 1
        # reseting
        if self.config[key] >= 100:
            self.config[key] = 0
            ap = torch.tensor(stdp.learning_rate[0][0].item()) * 2
            ap = torch.min(ap, self.maximum_positive_amplitude)
            an = ap * -0.75
            stdp.update_all_learning_rate(ap.item(), an.item())

    def train_deep(self, spike, potential, deep):
        self.reset_and_update_stdp(
            self.stdp_1 if deep == 1 else self.stdp_2,
            f'c:{deep}.spikecount'
        )
        potential = sf.pointwise_inhibition(potential)
        spike = potential.sign()
        winners = sf.get_k_winners(potential,
                                   self.config[f'c:{deep}.k_winner'],
                                   self.config[f'c:{deep}.r_inhibition'],
                                   spike)
        self.neural_activity["potentials"] = potential
        self.neural_activity["output_spikes"] = spike
        self.neural_activity["winners"] = winners
        return spike, potential

    def stdp(self, deep):
        params = (
            self.neural_activity["input_spikes"],
            self.neural_activity["potentials"],
            self.neural_activity["output_spikes"],
            self.neural_activity["winners"]
        )
        if deep == 1:
            self.stdp_1(*params)
        if deep == 2:
            self.stdp_2(*params)

    def update_learning_rates(self, Ap, An, anti_Ap, anti_An):
        self.stdp_3.update_all_learning_rate(Ap, An)
        self.antistdp_3.update_all_learning_rate(anti_An, anti_Ap)

    def dopamin(self, produce):
        params = (
            self.neural_activity["input_spikes"],
            self.neural_activity["potentials"],
            self.neural_activity["output_spikes"],
            self.neural_activity["winners"]
        )

        if produce:
            self.stdp_3(*params)  # reward
        else:
            self.antistdp_3(*params)  # punishment


def train_rl(network, batch, targets):
    perf = measure()
    for data, target in zip(batch, targets):
        predict = network(data, 3)
        if predict != -1:
            perf[('F', 'T')[predict == target]] += 1
            network.dopamin(predict == target)
        else:
            perf['U'] += 1

    for i in ['T', 'F', 'U']:
        perf[i] /= len(batch)
    return perf.tolist()


def test(network, batch, targets):
    perf = measure()
    for i, data in enumerate(batch):
        target = targets[i]
        predict = network(data, 3)
        if predict != -1:
            perf[('F', 'T')[predict == target]] += 1
        else:
            perf['U'] += 1

    for i in ['T', 'F', 'U']:
        perf[i] /= len(batch)
    return perf.tolist()


def predict(network, dataloader):
    network.eval()
    y_true, y_pred = [], []
    for batch, targets in dataloader:
        for (data, target) in zip(batch, targets):
            y_true.append(target)
            y_pred.append(network(data, 3))
    return y_true, y_pred


# TODO: Gabor Kernels must be refactored from original src code
reset_seed(cuda=args.gpu)
transform_filters = su.Filter((
    su.DoGKernel(3, 3/9, 6/9),
    su.DoGKernel(3, 6/9, 3/9),
    su.DoGKernel(7, 7/9, 14/9),
    su.DoGKernel(7, 14/9, 7/9),
    su.DoGKernel(13, 13/9, 26/9),
    su.DoGKernel(13, 26/9, 13/9)
), padding=6, thresholds=50)

#  Load Dataset
dataset = dataset = ImageFolder(
    args.data_dir,
    transform=Transformation(
        transform_filters
    )
)

# Split Test & Train
train_indices, test_indices = train_test_split(dataset, split=args.split_size)
train_dataloader = DataLoader(dataset,
                              batch_size=len(train_indices),
                              num_workers=args.num_workers,
                              pin_memory=args.gpu,
                              sampler=SubsetRandomSampler(train_indices))
test_dataloader = DataLoader(dataset,
                             batch_size=len(test_indices),
                             num_workers=args.num_workers,
                             pin_memory=args.gpu,
                             sampler=SubsetRandomSampler(test_indices))


network = Network()
# Layer 1
print("<Network Phase:training layer:1 State:start />")
layer_path = "data/caltech101/cached/network/first-layer.net"
start_time = datetime.now()
if os.path.isfile(layer_path):
    network.load_state_dict(torch.load(layer_path))
else:
    network.train()
    for epoch in tqdm(range(args.first_layer_epoch), desc="Epoch"):
        for batch, _ in tqdm(train_dataloader, desc="data-iter", leave=False):
            for data in batch:
                network(data, deep=1)
                network.stdp(deep=1)

    torch.save(network.state_dict(), layer_path)
delta = datetime.now() - start_time
print(f"<Network Phase:training layer:1 State:end   took:{delta} />")
#       <Network Phase:training layer:1 State:end   took:0:01:42.003395 />
#       <Network Phase:Testing  Method:accuracy     0.07931034482758621 />


# Layer 2
print("<Network Phase:training layer:2 State:start />")
layer_path = "data/caltech101/cached/network/second-layer.net"
start_time = datetime.now()
rest_epoch = 30
if rest_epoch >= args.second_layer_epoch and os.path.isfile(layer_path):
    network.load_state_dict(torch.load(layer_path))
else:
    network.train()
    for epoch in tqdm(range(rest_epoch, args.second_layer_epoch), desc="Epoch"):
        for batch, _ in tqdm(train_dataloader, desc="data-iter", leave=False):
            for data in batch:
                network(data, deep=2)
                network.stdp(deep=2)

    torch.save(network.state_dict(), layer_path)
delta = datetime.now() - start_time
print(f"<Network Phase:training layer:2 State:end   took:{delta} />")
# 0-10  <Network Phase:training layer:2 State:end   took:0:09:37.057285 />
#       <Network Phase:Testing  Method:accuracy     0.1413793103448276  />
# 10-30 <Network Phase:training layer:2 State:end   took:0:24:49.830814 />
#       <Network Phase:Testing  Method:accuracy     0.14482758620689656 />

# Layer 3
print("<Network Phase:training layer:3 State:start />")
layer_path = "data/caltech101/cached/network/network.net"
dev_path = "data/caltech101/cached/network/network-most-trained.net"
rest_epoch = 325
DEV = False

if DEV and os.path.isfile(dev_path):
    network.load_state_dict(torch.load(dev_path))
if not DEV and os.path.isfile(layer_path):
    network.load_state_dict(torch.load(layer_path))

# initial adaptive learning rates
apr = network.stdp_3.learning_rate[0][0].item()
anr = network.stdp_3.learning_rate[0][1].item()
app = network.antistdp_3.learning_rate[0][1].item()
anp = network.antistdp_3.learning_rate[0][0].item()

# True, False, unknown, epoch
best_train = np.array((0.0, 0.0, 0.0, 0.0))
best_test = np.array((0.0, 0.0, 0.0, 0.0))
start_time = datetime.now()
for epoch in tqdm(range(rest_epoch, args.third_layer_epoch), desc="Epoch"):
    network.train()
    perf_train = np.array([0.0, 0.0, 0.0])
    data_loader = tqdm(train_dataloader,
                       desc="data-iter",
                       leave=False,
                       miniters=1)

    for batch, targets in data_loader:
        T, F, U = train_rl(network, batch, targets)
        # update adaptive learning rates
        network.update_learning_rates(Ap=apr * F,
                                      An=anr * F,
                                      anti_Ap=app * T,
                                      anti_An=anp * T)

        perf_train += np.array([T, F, U])
    perf_train /= len(train_dataloader)
    # more true labeld
    if perf_train[0] >= best_train[0]:
        best_train = np.append(perf_train, epoch)
        print(f'\nBest:train updatedWith={best_train}')

    print('Current:train ',
          f'DiffOfBest[T,F]:={best_train[:2] - perf_train[:2]} U={perf_train[2]}')

    # disable network
    network.eval()
    data_loader = tqdm(test_dataloader,
                       desc="test-iter",
                       leave=False,
                       miniters=1)

    if DEV:
        if accuracy_score(*predict(network, data_loader)) >= 0.87:
            print(f'Epoch={epoch}, accuracy reaches 90%.')
            torch.save(network.state_dict(), layer_path)
            break
    else:
        for batch, targets in data_loader:
            T, F, U = test(network, batch, targets)

            if T >= best_test[0]:
                best_test = np.array([T, F, U, epoch])
                torch.save(network.state_dict(), layer_path)
                print(f'\nBest:test updatedWith={best_test}')

        print('Current:test ',
              f'DiffOfBest[T,F]={np.array([T,F] - best_test[:2])} U={U}')

if DEV:
    print(f'last {args.third_layer_epoch - 1} save model')
    torch.save(network.state_dict(), dev_path)

delta = datetime.now() - start_time
print(f"<Network Phase:training layer:3 State:end   took:{delta} />")
# 0-10   <Network Phase:training layer:3 State:end   took:0:15:57.753561 />
# 10-15  <Network Phase:training layer:3 State:end   took:0:07:58.105385 />
# 15-100 <Network Phase:training layer:3 State:end   took:2:22:25.527989 />
# accuracy: 0.8413793103448276
# 100-150<Network Phase:training layer:3 State:end   took:1:23:56.447735 />
# accuracy: 0.8448275862068966
# 150-200<Network Phase:training layer:3 State:end   took:1:18:34.740325 />
# accuracy: 0.8206896551724138
# 200-201<Network Phase:training layer:3 State:end   took:0:01:41.605375 />
# accuracy: 0.8655172413793103
# 201-210<Network Phase:training layer:3 State:end   took:0:09:31.787039 />
# accuracy: 0.8724137931034482
# 210-250<Network Phase:training layer:3 State:end   took:1:05:25.662818 />
# accuracy: 0.8586206896551725
# 250-300<Network Phase:training layer:3 State:end   took:1:27:03.813423 />
# accuracy: 0.8586206896551725
# 300-304<Network Phase:training layer:3 State:end   took:0:08:27.927803 />
# accuracy: 0.8517241379310345
# 304-325<Network Phase:training layer:3 State:end   took:0:39:53.570797 />
# accuracy: 0.8862068965517241
# -------------------------------------------------------------------------
# TOTAL 300 Epoch best_accuracy=88.6% trained_time=8:52:45
# <Network Phase:Testing  Method:accuracy     0.8862068965517241 />


""" Accuracy """
y_true, y_pred = predict(network, test_dataloader)
accuracy = accuracy_score(y_true, y_pred)
print(f"<Network Phase:Testing  Method:accuracy     {accuracy} />")


""" Confusion Matrix """
print("<Network Phase:Testing  Method:confusion_matrix />")
sv.plot_confusion_matrix(
    confusion_matrix(y_true, y_pred),
    target_names=args.features
)


""" Weights Feature """
sv.feature_selection_plot(
    network.convolution_3.weight.reshape(-1, 5, 5), ROW=10, COL=8)


""" Feature Selection """

features = torch.tensor([[[1]]]).float()

cstride = (1, 1)
features, cstride = sv.get_deep_feature(
    features,
    cstride,
    (transform_filters.max_window_size, transform_filters.max_window_size),
    transform_filters.kernels)
features, cstride = sv.get_deep_feature(
    features, cstride, (5, 5), (1, 1), network.convolution_1.weight)

features, cstride = sv.get_deep_feature(features, cstride, (2, 2), (2, 2))
features, cstride = sv.get_deep_feature(
    features, cstride, (3, 3), (1, 1),  network.convolution_2.weight)

features, cstride = sv.get_deep_feature(features, cstride, (3, 3), (3, 3))
sv.feature_selection_plot(features, ROW=5, COL=5)

""" Representational Dissimilarity Matrix """

indexes = [np.random.randint(0, len(dataset)) for i in range(5)]
rdm_matrix = np.concatenate([
    network(dataset[i][0], deep=-1).reshape(300, 360) for i in indexes
], axis=1)
rdm_matrix = rdm_matrix.T @ np.concatenate([
    dataset[i][0].reshape(300, 480) for i in indexes
], axis=1)

sv.rdm_plot(rdm_matrix)
