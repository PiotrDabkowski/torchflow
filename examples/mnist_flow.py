import pytorch_lightning as pl
from pl_bolts.datamodules import mnist_datamodule
from torchflow.flow import *
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import numpy as np
import pycat

class MnistFlow(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.glow_module = FlowGlowNetwork([8, 8], channels=1)
        # self.glow_module = FlowSequentialModule(FlowSqueeze2D(), FlowGlowStep(4), FlowTerminateGaussianPrior())

    def forward(self, imgs) -> Flow:
        return self.glow_module.encode(Flow(imgs))

    def training_step(self, batch, batch_idx):
        imgs = batch[0]
        levels = 32
        imgs = (0.9999*imgs*levels).floor() + torch.zeros_like(imgs).uniform_()
        imgs = imgs / levels
        flow = self(imgs)
        p_img = flow.get_logp().mean()
        num_pix = np.prod(imgs.shape[1:])
        p_pix = p_img / num_pix + np.log(1./levels)
        return -p_pix

    def validation_step(self, batch, batch_idx):
        imgs = batch[0]
        flow = self(imgs)
        p_img = flow.get_logp().mean()
        num_pix = np.prod(imgs.shape[1:])
        p_pix = p_img / num_pix + np.log(1. / 32)
        print()
        pycat.show(self.glow_module.decode(flow.sample_like()).data[0].cpu().numpy())
        print()
        return -p_pix


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        return optimizer


class MnistDTS(pl.LightningDataModule):
    def __init__(self, batch_size=512):
        super(MnistDTS, self).__init__()
        self.batch_size = batch_size


    def prepare_data(self):
        MNIST("/tmp/mnist", download=True, transform=ToTensor(), train=True)
        MNIST("/tmp/mnist", download=True, transform=ToTensor(), train=False)

    def train_dataloader(self) -> DataLoader:
        d = MNIST("/tmp/mnist", download=True, transform=ToTensor(), train=True)
        return DataLoader(d, batch_size=self.batch_size, num_workers=2, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        d = MNIST("/tmp/mnist", download=True, transform=ToTensor(), train=False)
        return DataLoader(d, batch_size=self.batch_size, num_workers=2, pin_memory=True)


if __name__ == '__main__':
    torch.manual_seed(11)
    np.random.seed(11)

    eye = MnistFlow()

    trainer = pl.Trainer()
    dts = MnistDTS()
    trainer.fit(eye, datamodule=dts)