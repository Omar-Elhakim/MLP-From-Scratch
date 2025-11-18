import numpy as np


class MLP:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return (x) * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_d(self, x):
        return (1 - x) * (1 + x)

    def UseTanh(self):
        self.act = self.tanh

    def UseBias(self):
        self.useBias = True

    def __init__(self, noOfInputs: int, layerSizes: [int], noOfOutputs: int):
        self.noOfInputs = noOfInputs
        self.layerSizes = layerSizes
        self.noOfOutputs = noOfOutputs
        self.initRandWeightMin = -0.1
        self.initRandWeightMax = 0.1
        self.useBias = False
        self.lr = 0.01
        self.act = self.sigmoid

    def Init(self):
        self.weights = self.initWeights(self.layerSizes)
        # print("Weights: \n" + str(self.weights))

        self.neuronsOutput = self.initNeuronsOutput()
        # print("Outputs: \n" + str(self.neuronsOutput))

        self.errors = self.initErrors()
        # print("Errors: \n" + str(self.errors))

        if self.useBias:
            self.bias = self.initBias()
            # print("Bias: \n" + str(self.bias))

        self.act_d = self.sigmoid_d if self.act == self.sigmoid else self.tanh_d

    def gRandomNumbers(self, size):
        return np.random.uniform(
            low=self.initRandWeightMin, high=self.initRandWeightMax, size=size
        )

    def initWeights(self, layerSizes):
        weights = []
        weights.append(self.gRandomNumbers((self.noOfInputs, layerSizes[0])))
        for i, l in enumerate(layerSizes[:-1]):
            weights.append(
                self.gRandomNumbers((l, layerSizes[i + 1]))
            )  # weights between last layer and the final output neurons
        weights.append(
            self.gRandomNumbers((layerSizes[-1], self.noOfOutputs))
        )  # weights between last layer and the final output neurons
        return weights

    def initBias(self):
        bias = []
        for i, l in enumerate(self.layerSizes):
            bias.append(
                self.gRandomNumbers((l))
            )  # bias between last layer and the final output neurons
        bias.append(
            self.gRandomNumbers((self.noOfOutputs))
        )  # bias between last layer and the final output neurons
        return bias

    def initNeuronsOutput(self):
        neuronsOutput = []
        neuronsOutput.append(np.random.random(self.noOfInputs))
        for l in self.layerSizes:
            neuronsOutput.append(np.random.random(l))
        neuronsOutput.append(np.random.random(self.noOfOutputs))
        return neuronsOutput

    def initErrors(self):
        errors = []
        for l in self.layerSizes:
            errors.append(np.random.random(l))
        errors.append(np.random.random(self.noOfOutputs))
        return errors

    """
    - A 'layer' is a layer of neurons
    - A 'connection' is the set of weights between two layers
    """

    def feedForward(self):
        # instead of multiplying each neuron weights and inputs
        # we can just multiply weights[i].T x inputs[j]
        # and produce the next neuronsOutput directly
        # for example (5, 2).T @ (5,) = (2,) -> activation((2,)) -> (2,)
        for j, connection in enumerate(self.weights):
            z = (connection.T @ self.neuronsOutput[j]) + (
                self.bias[j] if self.useBias else 0
            )
            self.neuronsOutput[j + 1] = self.act(z)

    def backPropagation(self, label):
        # Output layer error
        k = [i == label for i in range(self.noOfOutputs)]
        self.errors[-1] = (k - self.neuronsOutput[-1]) * self.act_d(
            self.neuronsOutput[-1]
        )

        for j in range(len(self.weights) - 1, 0, -1):  # number of hidden layers
            for i in range(
                self.errors[j - 1].shape[0]
            ):  # number of neurons in that layer
                s = (self.errors[j].dot(self.weights[j][i, :])) + (
                    self.bias[j - 1][i] if self.useBias else 0
                )  # s means sigma
                # don't know why but it errors out when i use bias[j][i] with [5,2,4(>2),3(<4)] or similar arch.
                self.errors[j - 1][i] = s * self.act_d(self.neuronsOutput[j][i])

    # I LOVE obfuscated code

    def updateWeights(self):
        for i in range(len(self.weights)):
            layer_input = self.neuronsOutput[i].reshape(-1, 1)
            layer_error = self.errors[i].reshape(1, -1)

            self.weights[i] += self.lr * (layer_input @ layer_error)

            if self.useBias:
                self.bias[i] += self.lr * self.errors[i]

    def fit(self, xTrain, yTrain):
        acc = 0
        for i, row in xTrain.iterrows():
            self.neuronsOutput[0] = row.values
            self.feedForward()

            if self.neuronsOutput[-1].argmax() == yTrain[i]:
                acc += 1

            self.backPropagation(yTrain[i])
            self.updateWeights()
        return acc

    def predict(self, xTest):
        y = []
        for i, row in xTest.iterrows():
            self.neuronsOutput[0] = row.values
            self.feedForward()
            y.append(self.neuronsOutput[-1].argmax())
        return y

        # acc += 1
        # return acc
        # accurecies.append(acc / len(xTrain))
