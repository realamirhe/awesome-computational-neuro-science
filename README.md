# Computational Neuro Science <img align="right" src="./assets/README/logo.svg" width="150px" >
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

### Tutorials
* <a href="/tutorials.md" title="tutorial Series on computational-neuroscience & spiking-neural-network"><img src="https://img.shields.io/badge/Our tutorial Series on CNS & SNN (🔥)-f7df1e" width="300px" alt="tutorial Series on computational-neuroscience & spiking-neural-network"/></a> 
* [Spiking-Neural-Network](https://github.com/Shikhargupta/Spiking-Neural-Network) is the pure python implementation of hardware efficient spiking neural network. It includes the modified learning and prediction rules which could be released on hardware and are energy efficient. Aim is to develop a network which could be used for on-chip learning as well as prediction.

### tools
* [Auryn](https://github.com/fzenke/auryn) is a C++ simulator for recurrent spiking neural networks with synaptic plasticity. It comes with the GPLv3. `SIMULATOR`
* [Bee](https://github.com/ricardodeazambuja/Bee) is an open source simulator for Spiking Neural Network (SNN) simulator, freely available, specialised in Liquid State Machine (LSM) systems with its core functions fully implemented in C. `SIMULATOR`
* [BindsNET](https://github.com/BindsNET/bindsnet) is a Python package used for simulating spiking neural networks (SNNs) on CPUs or GPUs using PyTorch Tensor functionality. `PACKAGE` ![promote-badge][promote]
* [BrainPy](https://github.com/PKU-NIP-Lab/BrainPy) is an integrative framework for computational neuroscience and brain-inspired computation based on Just-In-Time (JIT) compilation (built on the top of JAX and Numba). `FRAMEWORK`
* [Brian2](https://github.com/brian-team/brian2) is a clock-driven simulator for spiking neural networks. `SIMULATOR`
* [DRL with Population Coded Spiking Neural Network](https://github.com/combra-lab/pop-spiking-deep-rl) the PyTorch implementation of the **Pop**ulation-coded **S**piking **A**ctor **N**etwork (PopSAN) that integrates with both on-policy (PPO) and off-policy (DDPG, TD3, SAC) DRL algorithms for learning optimal and energy-efficient continuous control policies. `PACKAGE`
* [Encoders](https://github.com/iamsoroush/Encoders) is a Python utility package of encoding algorithms that encode real-valued data into spike trains for using in Spiking Neural Networks. `UTILITY`
* [GeNN](http://genn-team.github.io/genn/) compiles SNN network models to NVIDIA CUDA to achieve high-performing SNN model simulations. `PERFORMANCE`
* [Long short-term memory Spiking Neural Networks (LSNN)](https://github.com/IGITUGraz/LSNN-official) provides a [Tensorflow](https://www.tensorflow.org/) 1.12 library and a tutorial to train a recurrent spiking neural networks (LSNN). The library focuses on a single neuron and gradient model. `LIBRARY`
* [Nengo PyTorch](https://github.com/nengo/pytorch-spiking) a thin wrapper for PyTorch that adds a single voltage-only spiking model. The approach is independent from the Nengo framework. `LIBRARY`
* [Nengo](https://www.nengo.ai/nengo-dl/introduction.html) is a neuron simulator, and Nengo-DL is a deep learning network simulator that optimised spike-based neural networks based on an approximation method suggested by [Hunsberger and Eliasmith (2016)](https://arxiv.org/abs/1611.05141). `SIMULATOR`
* [Neurapse](https://github.com/udion/Neurapse) is a package in python which implements some of the fundamental blocks of SNN and is written in a manner so that it can easily be extended and customized for simulation purposes. `PACKAGE`
* [Neuron Simulation Toolkit (NEST)](https://nest-simulator.org) constructs and evaluates highly detailed simulations of spiking neural networks. This is useful in a medical/biological sense but maps poorly to large datasets and deep learning. `SIMULATOR`
* [Norse](https://github.com/norse/norse) is Pytorch expansion which aims to exploit the advantages of bio-inspired neural components, which are sparse and event-driven - a fundamental difference from artificial neural networks
* [PyNN](http://neuralensemble.org/docs/PyNN/) is a simulator-independent language for building neuronal network models. It does not currently provide mechanisms for optimisation or arbitrary synaptic plasticity. `SIMULATOR`
* [PySNN](https://github.com/BasBuller/PySNN/) is a Spiking neural network (SNN) framework written on top of PyTorch for efficient simulation of SNNs both on CPU and GPU. `FRAMEWORK`
* [PymoNNto](https://github.com/trieschlab/PymoNNto) The "**Py**thon **mo**dular **n**eural **n**etwork **to**olbox" allows you to create different Neuron-Groups, define their Behaviour and connect them with Synapse-Groups. `PACKAGE` ![promote-badge][promote]
* [Rockpool](https://gitlab.com/aiCTX/rockpool) is a Python package developed by SynSense for training, simulating and deploying spiking neural networks. It offers both JAX and PyTorch primitives. `PACKAGE`
* [SNN toolbox](https://snntoolbox.readthedocs.io/en/latest/guide/intro.html) is a framework to transform rate-based artificial neural networks into spiking neural networks, and to run them using various spike encodings. `FRAMEWORK`
* [Sinabs](https://gitlab.com/synsense/sinabs) is a python library for development and implementation of Spiking Convolutional Neural Networks (SCNNs). it provides support to import CNN models implemented in torch conveniently to test their spiking equivalent implementation. `LIBRARY`
* [SlayerPyTorch](https://github.com/bamsumit/slayerPytorch) is a **S**pike **LAY**er **E**rror **R**eassignment library, that focuses on solutions for the temporal credit problem of spiking neurons and a probabilistic approach to backpropagation errors. It includes support for the [Loihi chip](https://en.wikichip.org/wiki/intel/loihi). `PACKAGE`
* [SpikeTorch](https://github.com/djsaunde/spiketorch) Python package used for simulating spiking neural networks (SNNs) in PyTorch. [successor to this project](https://github.com/BINDS-LAB-UMASS/bindsnet). `PACKAGE`  
* [SpikingJelly](https://github.com/fangwei123456/spikingjelly) is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch. `FRAMEWORK`
* [SpykeTorch](https://github.com/miladmozafari/SpykeTorch) High-speed simulator of convolutional spiking neural networks with at most one spike per neuron. `PACKAGE`
* [Tonic](https://github.com/neuromorphs/tonic) is a tool to facilitate the download, manipulation and loading of event-based/spike-based data. It's like PyTorch Vision but for neuromorphic data! `LIBRARY`
* [WheatNNLeek](https://github.com/libgirlenterprise/WheatNNLeek) A Rust and common-lisp spiking neural network system. `LIBRARY`
* [cuSNN](https://github.com/tudelft/cuSNN) is a C++ library that enables GPU-accelerated simulations of large-scale Spiking Neural Networks (SNNs). `LIBRARY`
* [decolle](https://github.com/nmi-lab/decolle-public) implements an online learning algorithm described in the paper ["Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)"](https://arxiv.org/abs/1811.10766) by J. Kaiser, M. Mostafa and E. Neftci. `ALGORITHMS`, `UTILITY`
* [s2net](https://github.com/romainzimmer/s2net) is based on the implementation presented in [SpyTorch](https://github.com/fzenke/spytorch), but implements convolutional layers as well. It also contains a demonstration how to use those primitives to train a model on the [Google Speech Commands dataset](https://arxiv.org/abs/1804.03209). `LIBRARY`
* [snnTorch](https://github.com/jeshraghian/snntorch) is a Python package for performing gradient-based learning with spiking neural networks. It extends the capabilities of PyTorch, taking advantage of its GPU accelerated tensor computation and applying it to networks of spiking neurons. `PACKAGE`
* [spikeflow](https://github.com/colinator/spikeflow) Python library for easy creation and running of spiking neural networks in tensorflow. `LIBRARY`

### Papers
* "Convolutional spiking neural networks (SNN) for spatio-temporal feature extraction". [Github repository](https://github.com/aa-samad/conv_snn)
* "Enabling Deep Spiking Neural Networks with Hybrid Conversion and Spike Timing Dependent Backpropagation" published in [ICLR, 2020](https://openreview.net/forum?id=B1xSperKvH). [Github repository](https://github.com/nitin-rathi/hybrid-snn-conversion)

---
<small>Logo icon made by [Freepik](https://www.flaticon.com/authors/freepik) from [www.flaticon.com](https://www.flaticon.com/)</small>  
<small>List of tools are highly inspried by [Norse](https://github.com/norse/norse)</small>

[promote]: https://img.shields.io/badge/❤️promote-e95420
