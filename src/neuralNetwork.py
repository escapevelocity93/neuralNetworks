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
        #we calculate signal outputs using matrix multiplication
        #this is expressed concisely as X = W . I where W is the matrix of link weights and I is the matrix of inputs
        #and X is the resultant matrix of combined moderated signals

        #"So the thing to remember is, no matter how many layers we have, we can treat each layer like any other
        # - with incoming signals which we combine, link weights to moderate those incoming signals,
        # and an activation function to produce the output from that layer. We don’t care whether we’re working on the
        # 3rd or 53rd or even the 103rd layer - the approach is the same."

        #weight input hidden is the matrix of weights between the input and hidden layers
        self.weight_input_hidden = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        #weight hidden output is the matrix of weights between the hidden layers and the output layer
        self.weight_hidden_output = numpy.random.normal(0.0, pow(self.onodes, - 0.5), (self.onodes, self.hnodes))

        #A function which takes input signal and generates output signal while taking a threshold into account
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self):
        pass

    def query(self, inputs_list):
        #split inputs into a 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        #performing matrix multiplication on inputs using numpy dot product function
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass


class initiator:

    input_nodes = 4
    hidden_nodes = 4
    output_nodes = 4
    learning_rate = 0.3

    network = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(network.query([0.5,1,-1.5,9.5]))


