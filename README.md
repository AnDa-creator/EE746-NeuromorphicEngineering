# EE746-NeuromorphicEngineering
**Methods and Results**<br>
Implementation is done in Python. The method for the work includes broadly three serial steps,
- Preprocessing was done by feeding in the sound signal files into *Lyon's Passive ear model* and subsequently the output was encoded into spikes with *Ben Spiker's Algorithm(BSA)*. Total 78 spike trains are generated for each input based on different bands.
- We set the initial reservoir into a cuboidal shape and use probabilistic connection between the LIF neurons to set up the network. The model given used here was the same as  *Zhang et al* and *Gorad et al*. The shape of cuboid chosen is 5 * 5 * 5.
- Finally the classifier used is based on spiking neural networks. There are 10 output neurons(1 for each class) and the one with higher no.of spikes is chosen as the class for given input. During training we teach the neuron with high current values so as to tune to a particular input. We are still building up the final training and testing part to finally comment upon the recognition rate.
