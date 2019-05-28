import numpy as np
import random
import math

class Network(object):

    def __init__( self, sizes):

        self.layers_count = len(sizes)

        #input and output don't have weights
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        #input layer dont have bias
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]

        print('w ', self.weights, '\nb ', self.biases)

    def backprop(self, x, y):
        """
        x has dims (inputs_n, 1)
        y has dims (output_n, 1)
        """
        #forward pass

        layerInput = x
        #calc  and store inputs for output layer
        inputs = []
        for i in range(self.layers_count-2):
            inputs.append(np.dot(self.weights[i], layerInput) + self.biases[i]) # (hidden, 1)
            layerInput = sigmoid(inputs[i]) # hidden x 1

        #calc output for output layer
        z2 = np.dot(self.weights[-1], layerInput) + self.biases[-1] # (output, 1)
        a2 = z2

        #backward pass
        #cost = 0.5*( ( x-y )**2 )

        nabla_b = [np.zeros(b.shape) for b in self.biases ]
        nabla_w = [np.zeros(w.shape) for w in self.weights ]

        #calculate delta for output layer
        delta = (a2-y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, layerInput.T) # (output, 1) . (1, hidden)
        #delta = np.dot(self.weights[-1].T, delta) * ReLUprime(z1)  # (hidden, 1)

        #calculate delta for hidden layers
        for i in reversed(range(self.layers_count-2)):
            delta = np.dot(self.weights[i+1].T, delta) * sigmoidprime(inputs[i])  # (hidden, 1)
            nabla_b[i] = delta
            if not i == 0:
                nabla_w[i] = np.dot(delta, inputs[i-1].T) #(hidden, 1) . (1, input_n)
            else:
                nabla_w[i] = np.dot(delta, x.T)




        return (nabla_b, nabla_w)

    def SGD(self, trainingData, miniBatchSize, lRate, epochs, testData):
        """Stochastic Gradient Descent implementation.
        trainingData is a list op pairs (x,y)
        where x is an input and y is a desired output.
        """
        n = trainingData.__len__()

        for i in range(epochs):
            print("epoch: ", i, "/", epochs)
            #auto reduce learning rate every x epochs
            if i > 0 and i % 10 == 0:
                lRate = lRate * 0.95
                #print('lrate', lRate)
            #print("epoch: ", i)
            # first we have to shuffle our training data for each epoch
            random.shuffle(trainingData)
            # then we split the data into equal-sized minibatches
            batch = [trainingData[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)]

            # then we run SGD for each mini-batch
            for miniBatch in batch:
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                for network_input, desired_output in miniBatch:
                    # by our convention, first element contain inputs, and last element is desired output
                    delta_nabla_b, delta_nabla_w = self.backprop(network_input, desired_output)
                    
                    # assign modifiers calculated in backpropagation to apropriate positions
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
               
                # update our weights by applying results from backpropagation with respect to learning rate and by
                # dividing result by length of minibatch (so we have nice average across whole minibatch)
                self.weights = [w - nw * (lRate/len(miniBatch)) for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - nb * (lRate/len(miniBatch)) for b, nb in zip(self.biases, nabla_b)]
            #total_error = 0
            #for row in testData:
            #    #print(row[1], self.feedforward(row[0]))
            #    tmp = abs(row[1] - self.feedforward(row[0])) / row[1]
            #    total_error += tmp
            #print(total_error / len(testData))
        #print("w ", self.weights, "\nb ", self.biases)


    def feedforward(self, x):
        layerInput = x
        inputs = []
        for i in range(self.layers_count-2):
            inputs.append(np.dot(self.weights[i], layerInput) + self.biases[i]) # (hidden, 1)
            layerInput = sigmoid(inputs[i]) # hidden x 1

        #calc output for output layer
        z2 = np.dot(self.weights[-1], layerInput) + self.biases[-1] # (output, 1)
        a2 = z2
        return a2
        
def sigmoid(x):
    """ReLU function for hidden layer"""
    return 1.0/(1.0+np.exp(-x))

def sigmoidprime(x):
    """ReLU derivative"""

    return sigmoid(x) * (1 - sigmoid(x))
