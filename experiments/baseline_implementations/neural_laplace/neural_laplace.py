import torch

class LaplaceEncoder(torch.nn.Module):

    def __init__(self, config):
        super(LaplaceEncoder, self).__init__()
        torch.manual_seed(config.seed)
        print(config.encoder)
        self.config = config
        self.n_features = config.n_features
        self.hidden_sizes = config.encoder.hidden_sizes
        self.dropout_p = config.encoder.dropout_p
        self.latent_dim = config.encoder.latent_dim

        assert len(self.hidden_sizes) > 0

        self.layers = []
        activation = torch.nn.ReLU()

        self.layers.append(torch.nn.Linear(self.n_features,self.hidden_sizes[0]))
        # self.layers.append(torch.nn.BatchNorm1d(self.hidden_sizes[0]))
        self.layers.append(activation)
        self.layers.append(torch.nn.Dropout(self.dropout_p))

        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(self.hidden_sizes[i],self.hidden_sizes[i+1]))
            # self.layers.append(torch.nn.BatchNorm1d(self.hidden_sizes[i+1]))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(self.dropout_p))
        
        self.layers.append(torch.nn.Linear(self.hidden_sizes[-1],self.latent_dim))

        self.nn = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nn(x)

# from https://samholt.github.io/NeuralLaplace/notebooks/user_core.html
class LaplaceRepresentationFunc(torch.nn.Module):
    # SphereSurfaceModel : C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
        super(LaplaceRepresentationFunc, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.linear_tanh_stack = torch.nn.Sequential(
            torch.nn.Linear(s_dim * 2 + latent_dim, hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
        )

        for m in self.linear_tanh_stack.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

        phi_max = torch.pi / 2.0
        self.phi_scale = phi_max - -torch.pi / 2.0

    def forward(self, i):
        out = self.linear_tanh_stack(i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
            -1, 2 * self.output_dim, self.s_dim
        )
        theta = torch.nn.Tanh()(out[:, : self.output_dim, :]) * torch.pi  # From - pi to + pi
        phi = (
            torch.nn.Tanh()(out[:, self.output_dim :, :]) * self.phi_scale / 2.0
            - torch.pi / 2.0
            + self.phi_scale / 2.0
        )  # Form -pi / 2 to + pi / 2
        return theta, phi
