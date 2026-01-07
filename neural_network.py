import torch.nn.functional
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_intputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential( # sequential means, pass the output of the layer directly into the next
            # 1st hidden layer
            torch.nn.Linear(num_intputs, 30), # the linear layer takes the number of input and output nodes as arguments
            torch.nn.ReLU(), # Nonlinear activation fucntions are placed between the hidden layers

            # 2nd hidden layer
            torch.nn.Linear(30, 20), # the number of output nodes(neurons) of one hidden layer has to match the number of inputs of the next layer
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )
    
    def forward(self, x):
        logits = self.layers(x)
        return logits # the outputs of the last layer are called logits
    
# instantiate a new neural network object as follows:
"""
This means that this network expects 50 numbers as inputs and will
output 3 numbers which are the logits( raw numbers before applying sigmoid or softmax)
"""
model = NeuralNetwork(50, 3) # input size = 50, output size = 3
print(model)

"""
check the number of trainable parameters of this model:
Layer 1

Weights: 50 x 30 = 1500

Biases: 30

Total: 1530

Layer 2

Weights: 30 x 20 = 600

Biases: 20

Total: 620

Layer 3

Weights: 20 x 3 = 60

Biases: 3

Total: 63

Grand total = 2213
"""
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total numbers of trainable model parameters:", num_params)

print(model.layers[0].weight) # print the weight matrix of the first linear layer 30x50

# print the shape/dimentions of the matrix 
print(model.layers[0].weight.shape) # torch.Size([30, 50])

# print the bias vector
print(model.layers[0].bias) # prints a vector of 30 values

torch.manual_seed(123) # stop randomness, mainly for debugging, in real training we want randomness otherwise the model will not learn new patterns
model = NeuralNetwork(50, 3)
print(model.layers[0].weight)

torch.manual_seed(123) # stop randomness
X = torch.rand((1, 50)) # tensor of 50 random numbers
print(X)
out = model(X) # train on those 50 
print(out) # outputs the 3 logits using the forward pass of the model

