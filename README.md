<h1 align="center">Computational Neuro Science</h1>

## Resources <img align="right" src="./assets/README/logo.svg" width="75px" >
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

### tools
* [BindsNET](https://github.com/BindsNET/bindsnet) is a Python package used for simulating spiking neural networks (SNNs) on CPUs or GPUs using PyTorch Tensor functionality. `PACKAGE`
* [cuSNN](https://github.com/tudelft/cuSNN) is a C++ library that enables GPU-accelerated simulations of large-scale Spiking Neural Networks (SNNs). `LIBRARY`
* [decolle](https://github.com/nmi-lab/decolle-public) implements an online learning algorithm described in the paper ["Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)"](https://arxiv.org/abs/1811.10766) by J. Kaiser, M. Mostafa and E. Neftci. `ALGORITHMS`, `UTILITY`
* [GeNN](http://genn-team.github.io/genn/) compiles SNN network models to NVIDIA CUDA to achieve high-performing SNN model simulations. `PERFORMANCE`
* [Long short-term memory Spiking Neural Networks (LSNN)](https://github.com/IGITUGraz/LSNN-official) 
provides a [Tensorflow](https://www.tensorflow.org/) 1.12 library and a tutorial to train a recurrent spiking neural networks (LSNN). The library focuses on a single neuron and gradient model. `LIBRARY`
* [Nengo](https://www.nengo.ai/nengo-dl/introduction.html) is a neuron simulator, and Nengo-DL is a deep learning network simulator that optimised spike-based neural networks based on an approximation method suggested by [Hunsberger and Eliasmith (2016)](https://arxiv.org/abs/1611.05141). `SIMULATOR`
* [Nengo PyTorch](https://github.com/nengo/pytorch-spiking) a thin wrapper for PyTorch that adds a single voltage-only spiking model. The approach is independent from the Nengo framework. `LIBRARY`
* [Neuron Simulation Toolkit (NEST)](https://nest-simulator.org) constructs and evaluates highly detailed simulations of spiking neural networks. This is useful in a medical/biological sense but maps poorly to large datasets and deep learning. `SIMULATOR`
* [PyNN](http://neuralensemble.org/docs/PyNN/) is a simulator-independent language for building neuronal network models. It does not currently provide mechanisms for optimisation or arbitrary synaptic plasticity. `SIMULATOR`
* [PySNN](https://github.com/BasBuller/PySNN/) is a Spiking neural network (SNN) framework written on top of PyTorch for efficient simulation of SNNs both on CPU and GPU. `FRAMEWORK`
* [Rockpool](https://gitlab.com/aiCTX/rockpool) is a Python package developed by SynSense for training, simulating and deploying spiking neural networks. It offers both JAX and PyTorch primitives. `PACKAGE`
* [Sinabs](https://gitlab.com/synsense/sinabs) is a python library for development and implementation of Spiking Convolutional Neural Networks (SCNNs). it provides support to import CNN models implemented in torch conveniently to test their spiking equivalent implementation. `LIBRARY`
* [SlayerPyTorch](https://github.com/bamsumit/slayerPytorch) is a **S**pike **LAY**er **E**rror **R**eassignment library, that focuses on solutions for the temporal credit problem of spiking neurons and a probabilistic approach to backpropagation errors. It includes support for the [Loihi chip](https://en.wikichip.org/wiki/intel/loihi). `PACKAGE`


---
### Tutorials

<p align="center"><img src="./assets/README/contribution.png"  /></p>
<p align="center"><small>Feel free to <a href="https://github.com/amirHossein-Ebrahimi/ComputationalNeuroScience/blob/master/CONTRIBUTING.md"><b>contribute</b></a> & <a href="https://github.com/amirHossein-Ebrahimi/ComputationalNeuroScience/issues/new"><b>open an issue</b></a> to help <code>the nervous system</code></small></p>

Project Charters

1. **Neuron Models**
2. **Neural Population**
3. **Unsupervised STDP Learning**
4. **Reward Modulated STDP Learning**
5. **DoG & Gabor in Convolution**
6. **Full object detection model based on spiking neural network**

## DoG & Gabor in Convolution

- [x] [Gabor Filter](https://en.wikipedia.org/wiki/Gabor_filter)
- [x] [Difference of Gaussian](https://en.wikipedia.org/wiki/Difference_of_Gaussians)
- [x] Feature map
- [x] Detect dominant lines and features, and show improvement and precision through time

## Unsupervised STDP Learning

- [x] Implementation Unsupervised STDP Learning
- [x] Plot delta weights caused by STDP learning for two neuron with random excitation
- [x] Generate 10x2 Spiking Networks & learn Two 10th-tuple by each output neuron
- [x] Add an inhibitory neuron to upper network

## Full object detection model based on spiking neural network

Model trained and tested on [caltech101](https://www.tensorflow.org/datasets/catalog/caltech101) dataset

> Caltech-101 consists of pictures of objects belonging to 101 classes, plus one background clutter class. Each image is labelled with a single object. Each class contains roughly 40 to 800 images, totalling around 9k images. Images are of variable sizes, with typical edge lengths of 200-300 pixels.

We presents an deep spiking neural model which consist of 3 layer. for first two layer only STDP learning is used, and for last layer dopamine releases followed by STDP and anti-STDP.

**For more documentation see code, documentation will be updated**

---

> [class videos and Lecture Notes](https://t.me/CNRLab)
>
> Computational Neuroscience Research Lab. (Department of Computer Science, University of Tehran) For more info, please visit [cnrl.ut.ac.ir](https://cnrl.ut.ac.ir/)




---
<small>Logo icon made by [Freepik](https://www.flaticon.com/authors/freepik) from [www.flaticon.com](https://www.flaticon.com/)</small>  
<small>List of tools are highly inspried by [Norse](https://github.com/norse/norse)</small>
