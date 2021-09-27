import torch.nn as nn
import torch
from typing import Tuple
import globals

class autoencoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = globals.device
        self.num_features = args.input_dim
        self.latent_dim = args.latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.num_features, self.latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_features),
        )
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(inputs)
        out = self.decoder(x)
        loss = self.criterion(out, inputs)
        return loss, out

    def fit(self, epoch: int, train_loader, verbose=True) -> None:

        for batch_idx, (data, _, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)

            # throughput measurement - start
            if self.args.time_throughput:
                starter_noDT, ender_noDT = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter_noDT.record()

            # train the network
            self.optimizer.zero_grad()
            loss, _ = self(data)
            loss.backward()
            self.optimizer.step()

            # throughput measurement - end
            if self.args.time_throughput:
                ender_noDT.record()
                torch.cuda.synchronize()
                globals.total_throughput_time_noDT += starter_noDT.elapsed_time(ender_noDT)/1000
            
            if verbose and batch_idx % self.args.log_interval == 0:
                msg = 'Train Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f}'
                print(msg.format(epoch, batch_idx, loss.detach().cpu().numpy()))
