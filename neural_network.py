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