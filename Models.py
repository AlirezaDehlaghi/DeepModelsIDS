# model_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == "elu":
        return nn.ELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "gelu":
        return nn.GELU()
    elif name == "swish":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, dropout=0.0, activation="relu"):
        super().__init__()
        act = get_activation(activation)

        encoder = []
        prev_dim = input_dim
        for dim in hidden_layers:
            encoder.append(nn.Linear(prev_dim, dim))
            encoder.append(act)
            if dropout > 0:
                encoder.append(nn.Dropout(dropout))
            prev_dim = dim
        encoder.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        prev_dim = latent_dim
        for dim in reversed(hidden_layers):
            decoder.append(nn.Linear(prev_dim, dim))
            decoder.append(act)
            if dropout > 0:
                decoder.append(nn.Dropout(dropout))
            prev_dim = dim
        decoder.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# class LSTMAutoEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
#         super().__init__()
#         self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.to_latent = nn.Linear(hidden_dim, latent_dim)
#         self.from_latent = nn.Linear(latent_dim, hidden_dim)
#         self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
#
#         self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.to_latent = nn.Linear(hidden_dim, latent_dim)
#         self.from_latent = nn.Linear(latent_dim, hidden_dim)
#
#         self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.output_layer = nn.Linear(hidden_dim, input_dim)
#
#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         batch_size, seq_len, _ = x.size()
#         _, (h_n, _) = self.encoder(x)
#         latent = self.to_latent(h_n[-1])
#         hidden_dec = self.from_latent(latent).unsqueeze(0)
#         dec_input = hidden_dec.repeat(seq_len, 1, 1).permute(1, 0, 2)
#         out, _ = self.decoder(dec_input)
#         return out

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()

        # --- Encoding ---
        _, (h_n, _) = self.encoder(x)
        latent = self.to_latent(h_n[-1])  # [batch, latent_dim]

        # --- Decoding ---
        hidden_dec = self.from_latent(latent).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        cell_dec = torch.zeros_like(hidden_dec)  # LSTM needs both hidden and cell

        # Option 1: Feed zeros for all time steps (teacher forcing could be added)
        dec_input = torch.zeros(batch_size, seq_len, feature_dim, device=x.device)

        dec_output, _ = self.decoder(dec_input, (hidden_dec, cell_dec))
        reconstructed = self.output_layer(dec_output)

        return reconstructed


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, dropout=0.0, activation="relu"):
        super().__init__()
        act = get_activation(activation)

        encoder = []
        prev_dim = input_dim
        for dim in hidden_layers:
            encoder.append(nn.Linear(prev_dim, dim))
            encoder.append(act)
            if dropout > 0:
                encoder.append(nn.Dropout(dropout))
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        decoder = []
        prev_dim = latent_dim
        for dim in reversed(hidden_layers):
            decoder.append(nn.Linear(prev_dim, dim))
            decoder.append(act)
            if dropout > 0:
                decoder.append(nn.Dropout(dropout))
            prev_dim = dim
        decoder.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, n_samples=1):

        mu, logvar = self.encode(x)
        recons = [self.decoder(self.reparameterize(mu, logvar)) for _ in range(n_samples)]
        recon_avg = torch.stack(recons).mean(dim=0)
        return recon_avg, mu, logvar


class ConditionalVariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_layers, latent_dim, dropout=0.0, activation="relu"):
        super().__init__()
        act = get_activation(activation)
        total_input = input_dim + cond_dim

        encoder = []
        prev_dim = total_input
        for dim in hidden_layers:
            encoder.append(nn.Linear(prev_dim, dim))
            encoder.append(act)
            if dropout > 0:
                encoder.append(nn.Dropout(dropout))
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        decoder = []
        prev_dim = latent_dim + cond_dim
        for dim in reversed(hidden_layers):
            decoder.append(nn.Linear(prev_dim, dim))
            decoder.append(act)
            if dropout > 0:
                decoder.append(nn.Dropout(dropout))
            prev_dim = dim
        decoder.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x, c):
        xc = torch.cat([x, c], dim=1)
        h = self.encoder(xc)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c, n_samples=1):
        mu, logvar = self.encode(x, c)
        recons = [self.decoder(torch.cat([self.reparameterize(mu, logvar), c], dim=1)) for _ in range(n_samples)]
        recon_avg = torch.stack(recons).mean(dim=0)
        return recon_avg, mu, logvar


class ProbabilisticVariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, dropout=0.0, activation="relu"):
        super().__init__()
        act = get_activation(activation)

        # Shared encoder
        encoder = []
        prev_dim = input_dim
        for dim in hidden_layers:
            encoder.append(nn.Linear(prev_dim, dim))
            encoder.append(act)
            if dropout > 0:
                encoder.append(nn.Dropout(dropout))
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Shared decoder with same architecture as encoder (reversed)
        decoder = []
        prev_dim = latent_dim
        for dim in reversed(hidden_layers):
            decoder.append(nn.Linear(prev_dim, dim))
            decoder.append(act)
            if dropout > 0:
                decoder.append(nn.Dropout(dropout))
            prev_dim = dim
        self.decoder = nn.Sequential(*decoder)
        self.fc_mu_out = nn.Linear(prev_dim, input_dim)
        self.fc_logvar_out = nn.Linear(prev_dim, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-6.0, max=2.0)  # <-- Add this
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, n_samples=1):
        mu, logvar = self.encode(x)
        mus, logvars = [], []
        for _ in range(n_samples):
            z = self.reparameterize(mu, logvar)
            h_dec = self.decoder(z)
            mus.append(self.fc_mu_out(h_dec))
            logvars.append(torch.clamp(self.fc_logvar_out(h_dec), min=-6.0, max=2.0))
        mu_out = torch.stack(mus).mean(dim=0)
        logvar_out = torch.stack(logvars).mean(dim=0)
        return mu_out, logvar_out, mu, logvar


def compute_probabilistic_likelihood(x, mu, logvar):
    logvar = torch.clamp(logvar, min=-6.0, max=2.0)
    log_two_pi = torch.log(torch.tensor(2 * torch.pi, device=logvar.device))
    log_pdf = -0.5 * (log_two_pi + logvar + ((x - mu) ** 2) / torch.exp(logvar))
    return torch.sum(log_pdf, dim=1)


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# Training loop for LSTM AutoEncoder
def train_autoencoder(
    model,
    train_data,
    batch_size=32,
    lr=1e-3,
    optimizer_name="adam",
    num_epochs=20,
    device="cpu",
    verbose=True,
    n_samples=1
):
    model.to(device)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()

            if model.__class__.__name__ == "LSTMAutoEncoder":
                recon = model(x)
                loss = nn.MSELoss()(recon, x)
            elif model.__class__.__name__ == "VariationalAutoEncoder":
                recon, mu, logvar = model(x, n_samples=n_samples)
                recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld
            elif model.__class__.__name__ == "ConditionalVariationalAutoEncoder":
                c = batch[1].to(device)
                recon, mu, logvar = model(x, c, n_samples=n_samples)
                recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld
            elif model.__class__.__name__ == "ProbabilisticVariationalEncoder":
                mu_out, logvar_out, mu, logvar = model(x, n_samples=n_samples)
                logvar_out = torch.clamp(logvar_out, min=-6.0, max=2.0)
                neg_log_likelihood = -torch.mean(compute_probabilistic_likelihood(x, mu_out, logvar_out))
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = neg_log_likelihood + kld

            else:
                recon = model(x)
                loss = nn.MSELoss()(recon, x)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.6f}")

    return model


def compute_reconstruction_error(model, data_tensor, cond_tensor=None, device="cpu", n_samples=1):
    model.eval()
    with torch.no_grad():
        data_tensor = data_tensor.to(device)
        if model.__class__.__name__ == "ConditionalVariationalAutoEncoder":
            cond_tensor = cond_tensor.to(device)
            reconstructed, _, _ = model(data_tensor, cond_tensor, n_samples=n_samples)
            errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
        elif model.__class__.__name__ == "VariationalAutoEncoder":
            reconstructed, _, _ = model(data_tensor, n_samples=n_samples)
            errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
        elif model.__class__.__name__ == "ProbabilisticVariationalEncoder":
            mu_out, logvar_out, _, _ = model(data_tensor, n_samples=n_samples)
            errors = -compute_probabilistic_likelihood(data_tensor, mu_out, logvar_out)
        elif model.__class__.__name__ == "LSTMAutoEncoder":
            reconstructed = model(data_tensor)
            # Reduce over both time and feature dimensions: [B, T, D] → [B]
            # errors = torch.mean((data_tensor - reconstructed) ** 2, dim=[1, 2])
            errors = torch.mean((data_tensor[:, -1, :] - reconstructed[:, -1, :]) ** 2, dim=1)
        else:
            reconstructed = model(data_tensor)
            errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
    return errors.cpu().numpy()
