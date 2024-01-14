#Chanels reducer mapps embedding tensor 256 channels * H 32 * W 32 to 20 class id * H 32 * W 32 
#Using 3*3 convolutions
import torch
from torch.utils.data import DataLoader
from pathlib import Path as _P
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EmbeddingConvReducer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, device=DEVICE, chkpt_path=None):
        super().__init__()
        self.device = device
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=True),
            #torch.nn.Upsample(scale_factor=2, mode='nearest') #Upsample to preview size
        )
        self.chkpt_path = chkpt_path or f"weight_dumps/EmbeddingConvReducer_{in_channels}_{out_channels}_numlayers_{len(self.layers)}.pth"

        try:
            self.load_state_dict(torch.load(self.chkpt_path))
            print(f"Success Loading EmbeddingConvReducer from {self.chkpt_path}")
        except:
            print(f"Failed to load EmbeddingConvReducer from {self.chkpt_path}")
        self.to(self.device)

    def save_to_checkpoint(self):
        if self.chkpt_path is None:
            print("chkpt_path is None, not saving")
            return
        _P(self.chkpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), self.chkpt_path)
        print(f"Saved EmbeddingConvReducer to {self.chkpt_path}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Assume x is B * C * H * W
        x = self.layers(x)
        return x

    def __repr__(self):
        #Print number or total params in the model, and of which are trainable:
        num_paramters = sum(p.numel() for p in self.parameters())
        num_trainable_paramters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        sup_retp = super().__repr__()
        return f"{sup_retp}" \
                f"num_paramters={num_paramters}, num_trainable_paramters={num_trainable_paramters}"

    def __str__(self):
        return self.__repr__()

    def __call__(self, x):
        return self.forward(x)
    
    def predict(self, x):
        return self.forward(x)
    
    def fit(self, dataloader, epochs=10):
        self.train()
        print(self)
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        loss_graph = []
        try:
            for epoch in range(epochs):
                for x, y in dataloader:
                    optimizer.zero_grad()
                    y_pred = self.forward(x.to(self.device))
                    #loss = (y_pred.softmax(dim=1).to(self.device) - y.softmax(dim=1).to(self.device)).abs().mean()
                    loss = (y_pred-y).abs().sum()
                    loss_graph.append(loss.detach().cpu().numpy())
                    loss.backward()
                    optimizer.step()
                    print(f"epoch: {epoch}, loss: {loss.detach().cpu().numpy()}", end='\r')
                print(f"\n")

        except KeyboardInterrupt:
            print("KeyboardInterrupt")

        return loss_graph



if __name__ == '__main__':
    self = EmbeddingConvReducer(256, 20).to(DEVICE)
    print(self)
    x = torch.rand(1, 256, 32, 32).to(DEVICE)
    y = self(x)
    print(y.shape)
