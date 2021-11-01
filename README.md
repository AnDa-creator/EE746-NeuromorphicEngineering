Project on Implementation of Liquid State Machines
=================================================
**Introduction and Motivation**
Liquid state machines are widely used architecture in the neuroscience community for solving problems related to time series prediction and also speech recognition. The main purpose of Stage 1 of our project is to replicate the results published by *Zhang et al* on implementation of such an LSM. In particular, we seek to classify the subset of digits from the TI-46 dataset.

**Literature Survey**
We mainly follow the implementation of *Zhang et al* for the ideation of preprocessing and reservoir. Learning rule used for the classification in readout neurons is taken from an amalgamation of *Zhang et al* and *Gorad et al*. The teacher current scheme that was implemented is inspired from *Gorad et al*. Besides these two main papers, SpilLinc model  although for image recognition, was of immense use in understanding the problem.

**Methods and Results**
Implementation is done in Python. The method for the work includes broadly three serial steps,
- Preprocessing was done by feeding in the sound signal files into *Lyon's Passive ear model* and subsequently the output was encoded into spikes with *Ben Spiker's Algorithm(BSA)*. Total 78 spike trains are generated for each input based on different bands.
- We set the initial reservoir into a cuboidal shape and use probabilistic connection between the LIF neurons to set up the network. The model given used here was the same as  *Zhang et al* and *Gorad et al*. The shape of cuboid chosen is 5\times5\times5$.
- Finally the classifier used is based on spiking neural networks. There are 10 output neurons(1 for each class) and the one with higher no.of spikes is chosen as the class for given input. During training we teach the neuron with high current values so as to tune to a particular input. We are still building up the final training and testing part to finally comment upon the recognition rate.

