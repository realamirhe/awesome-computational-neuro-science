# Computational Neuro Science

<p align="center"><img src="./assets/README/logo.svg" width="250px"  /></p>

---

<small>Icons made by [Freepik](https://www.flaticon.com/authors/freepik) from [www.flaticon.com](https://www.flaticon.com/)</small>

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

We presents an deep spiking nerual model which consist of 3 layer. for first two layer only STDP learning is used, and for last layer dopamin releases followed by STDP and anti-STDP.

**For more documentation see code, documentation will be updated**

---

> [class videos and Lecture Notes](https://t.me/CNRLab)
>
> Computational Neuroscience Research Lab. (Department of Computer Science, University of Tehran) For more info, please visit [cnrl.ut.ac.ir](https://cnrl.ut.ac.ir/)
