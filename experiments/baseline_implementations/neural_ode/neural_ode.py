import torch



class NeuralGradient(torch.nn.Module):
    def __init__(self, config):
        super(NeuralGradient, self).__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.n_features = config.n_features
        self.hidden_sizes = config.encoder.hidden_sizes
        self.dropout_p = config.encoder.dropout_p

        assert len(self.hidden_sizes) > 0

        self.layers = []
        activation = torch.nn.Softplus()

        self.layers.append(torch.nn.Linear(self.n_features + 1,self.hidden_sizes[0])) # +1 for y
        self.layers.append(torch.nn.BatchNorm1d(self.hidden_sizes[0]))
        self.layers.append(activation)
        self.layers.append(torch.nn.Dropout(self.dropout_p))

        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(self.hidden_sizes[i],self.hidden_sizes[i+1]))
            self.layers.append(torch.nn.BatchNorm1d(self.hidden_sizes[i+1]))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(self.dropout_p))
        
        self.layers.append(torch.nn.Linear(self.hidden_sizes[-1],1))

        self.nn = torch.nn.Sequential(*self.layers)


    def forward(self, t, y, cov):
        """
        Computes the gradient of the neural ODE at a given point (t, y)
        Args:
            t: the time point, a scalar
            y: the values of the function at time t, a vector of shape (n_features)
            cov: the covariates at time t, a vector of shape (n_covariates)
        """
        # print(f"Forward call with t = {t}")
        y_cov = torch.cat([y, cov], dim=-1) # Should have shape (n_features + n_covariates) = (1 + M)
        y_cov = y_cov.unsqueeze(0) # Add batch dimension
        dydt = self.nn(y_cov) # Should have shape (1, n_features) = (1, 1)

        # The rest of the gradient is equal to 0, pad the gradient with zeros
        # grad = torch.zeros_like(y)
        # grad[:,0] = dydt[:,0]
        return dydt[0,:] # Remove batch dimension
    

class NeuralGradient2(torch.nn.Module):
    def __init__(self, config):
        super(NeuralGradient2, self).__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.n_features = config.n_features
        self.hidden_sizes = config.encoder.hidden_sizes
        self.dropout_p = config.encoder.dropout_p

        assert len(self.hidden_sizes) > 0

        self.layers = []
        activation = torch.nn.Softplus()

        self.layers.append(torch.nn.Linear(self.n_features + 1,self.hidden_sizes[0])) # +1 for y
        # self.layers.append(torch.nn.BatchNorm1d(self.hidden_sizes[0]))
        self.layers.append(activation)
        self.layers.append(torch.nn.Dropout(self.dropout_p))

        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(self.hidden_sizes[i],self.hidden_sizes[i+1]))
            # self.layers.append(torch.nn.BatchNorm1d(self.hidden_sizes[i+1]))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(self.dropout_p))
        
        self.layers.append(torch.nn.Linear(self.hidden_sizes[-1],1))

        self.nn = torch.nn.Sequential(*self.layers)


    def forward(self, t, y):
        """
        Computes the gradient of the neural ODE at a given point (t, y)
        Args:
            t: the time point, a scalar
            y: the values of the function at time t, a vector of shape (n_features), i.e., (1 + n_covariates)
        """
        # print(f"Forward call with t = {t}")
        dydt = self.nn(y.unsqueeze(0)) # Should have shape (1, n_features) = (1, 1)

        dydt = dydt.squeeze(0) # Remove batch dimension

        # The rest of the gradient is equal to 0, pad the gradient with zeros
        grad = torch.zeros_like(y)
        grad[0] = dydt[0]
        return grad

