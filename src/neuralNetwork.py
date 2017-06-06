#numpy handles arrays and related operations
import numpy
#scipy does something
import scipy.special


class neuralNetwork:

    #constructor for our neural network
    #takes nodes and learning rate as parameters
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        #initialise and assign NN member variables
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #learning rate is how rapidly our network will abandon old beliefs for new ones
        self.learn_rate = learningrate

        #link weights are the heart of neural networks
        self.weight_input_hidden = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.weight_hidden_output = numpy.random.normal(0.0, pow(self.onodes, - 0.5), (self.onodes, self.hnodes))

        #A function which takes input signal and generates output signal while taking a threshold into account
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self):
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass


class initiator:

    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    network = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(network.query([0.5,1,-1.5]))


