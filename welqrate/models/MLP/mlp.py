from torch.nn import Module, Linear, Sequential, Sigmoid, Dropout, ReLU
import torch
from torch_scatter import scatter_mean, scatter_add


from torch.nn import Module, Linear, ReLU, Dropout, Sequential

class MLP(Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, one_hot=False):
        super().__init__()
        self.one_hot = one_hot

        # Setup the MLP layers using ReLU activation
        layers = []
        for _ in range(num_layers):
            layers.extend([
                Dropout(0.05),
                Linear(input_dim, hidden_dim),
                ReLU(),
                Dropout(0.35),
            ])
            input_dim = hidden_dim  

        layers.append(Linear(hidden_dim, 1))  # Final output layer

        self.mlp = Sequential(*layers)


    def forward(self, batch_data):
        if self.one_hot:
            x = batch_data.one_hot_atom.float()  
        else:
            x = batch_data.x
          
        x = scatter_add(x, batch_data.batch, dim=0)
        x = self.mlp(x)

        return x

class bcl_MLP(Module):
    def __init__(self, hidden_dim=32, num_layers=2, input_dim=391):
        super().__init__()

        # Setup the MLP layers using ReLU activation
        layers = []
        for _ in range(num_layers):
            layers.extend([
                Dropout(0.05),
                Linear(input_dim, hidden_dim),
                ReLU(),
                Dropout(0.35),
            ])
            input_dim = hidden_dim  # Update input dimension for the next layer

        layers.append(Linear(hidden_dim, 1))  # Final output layer

        self.mlp = Sequential(*layers)

    def forward(self, batch_data):
        x = batch_data.bcl.view(-1, 391)
        x = self.mlp(x)

        return x



