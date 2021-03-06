import pytorch_lightning as pl
from pl_bolts.datamodules import mnist_datamodule
from .flow import *
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, LSUN, CelebA
import numpy as np
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
import os

NUM_LEVELS = 256


class MnistFlow(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.glow_module = FlowGlowNetwork([32, 32, 32, 32], channels=3)
        # self.glow_module = FlowSequentialModule(FlowSqueeze2D(), FlowGlowStep(4), FlowTerminateGaussianPrior())


    def training_step(self, batch, batch_idx):
        global VAL_STEP
        VAL_STEP = 0
        imgs = level_encode(batch[0], num_levels=NUM_LEVELS)
        flow = self.glow_module.encode(Flow(imgs))
        loss = flow.get_elem_bits(
            input_data_shape=imgs.shape, num_data_levels=NUM_LEVELS
        ).mean()
        self.log("step", self.global_step)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch[0]
        flow = self.glow_module.encode(Flow(imgs))
        loss = flow.get_elem_bits(
            input_data_shape=imgs.shape, num_data_levels=NUM_LEVELS
        ).mean()
        for temperature in [0.1, 0.3, 0.4, 0.6, 0.7, 0.8]:
            self.log(
                "samples@%f" % temperature,
                [
                    wandb.Image(
                        self.glow_module.decode(flow.sample_like(temperature))
                        .data[0]
                        .detach()
                    )
                ],
                reduce_fx=lambda x: sum(x, []),
            )
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer



class FMnistDTS(pl.LightningDataModule):
    def __init__(self, batch_size=2048):
        super(FMnistDTS, self).__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        FashionMNIST("/tmp/fmnist", download=True, train=True)
        FashionMNIST("/tmp/fmnist", download=True, train=False)

    def train_dataloader(self) -> DataLoader:
        d = FashionMNIST(
            "/tmp/fmnist",
            download=True,
            transform=Compose([Resize(32), ToTensor()]),
            train=True,
        )
        return DataLoader(d, batch_size=self.batch_size, num_workers=2, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        d = FashionMNIST(
            "/tmp/fmnist",
            download=True,
            transform=Compose([Resize(32), ToTensor()]),
            train=False,
        )
        return DataLoader(d, batch_size=self.batch_size, num_workers=2, pin_memory=True)


class MnistDTS(pl.LightningDataModule):
    def __init__(self, batch_size=512):
        super(MnistDTS, self).__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST("/tmp/mnist", download=True, train=True)
        MNIST("/tmp/mnist", download=True, train=False)

    def train_dataloader(self) -> DataLoader:
        d = MNIST(
            "/tmp/mnist",
            download=True,
            transform=Compose([Resize(32), ToTensor()]),
            train=True,
        )
        return DataLoader(d, batch_size=self.batch_size, num_workers=2, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        d = MNIST(
            "/tmp/mnist",
            download=True,
            transform=Compose([Resize(32), ToTensor()]),
            train=False,
        )
        return DataLoader(d, batch_size=self.batch_size, num_workers=2, pin_memory=True)


class CifarDTS(pl.LightningDataModule):
    def __init__(self, batch_size=2048):
        super(CifarDTS, self).__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        CIFAR10("/tmp/cifar10", download=True, train=True)
        CIFAR10("/tmp/cifar10", download=True, train=False)

    def train_dataloader(self) -> DataLoader:
        d = CIFAR10(
            "/tmp/cifar10",
            download=True,
            transform=Compose([Resize(32), ToTensor()]),
            train=True,
        )
        return DataLoader(d, batch_size=self.batch_size, num_workers=5, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        d = CIFAR10(
            "/tmp/cifar10",
            download=True,
            transform=Compose([Resize(32), ToTensor()]),
            train=False,
        )
        return DataLoader(d, batch_size=self.batch_size, num_workers=5, pin_memory=True)


class CelebADTS(pl.LightningDataModule):
    root = "/home/piter/datasets"

    def __init__(self, batch_size=256):
        super(CelebADTS, self).__init__()
        self.batch_size = batch_size
        self.train_size = None
        self.val_size = None

    def prepare_data(self):
        d = CelebA(self.root, download=True)
        self.train_size = int(0.9 * len(d))
        self.val_size = len(d) - self.train_size

    def _split(self, dts):
        return random_split(
            dts,
            [self.train_size, self.val_size],
            generator=torch.Generator().manual_seed(11),
        )

    def train_dataloader(self) -> DataLoader:
        dts = CelebA(
            self.root,
            download=True,
            transform=Compose([CenterCrop(178), Resize(64), ToTensor()]),
        )
        return DataLoader(
            self._split(dts)[0],
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        dts = CelebA(
            self.root,
            download=True,
            transform=Compose([CenterCrop(178), Resize(64), ToTensor()]),
        )
        return DataLoader(
            self._split(dts)[1],
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
        )


# mnist_dts = MnistDTS(1)
# batch = next(iter(mnist_dts.val_dataloader()))[0]
#
# model = MnistFlow.load_from_checkpoint("/home/piter/PycharmProjects/torchflow/flow_try/77b9d8mm/checkpoints/epoch=53-step=6371.ckpt")
# print(type(batch))
# flow = model.glow_module(Flow(batch-0.5))
# print(flow)
# print(flow.logpz)
# print(flow.logdet)
# print(flow)
# print(torch.slogdet(calculate_jacobian(Flow(batch-0.5), model.glow_module)))
# exit()



if __name__ == "__main__":
    torch.manual_seed(11)
    np.random.seed(11)

    root_dir = "bolt"
    project = "flow_try"
    name = "celeba_pls"
    version = "1"
    project_dir = os.path.join(root_dir, project)
    run_dir = os.path.join(project_dir, name, version)
    os.makedirs(run_dir, exist_ok=True)

    wandb_logger = WandbLogger(name=name, project=project, save_dir=run_dir, version=f"{name}-{version}")

    ckpts_dir = os.path.join(run_dir, "ckpts")
    ckpt_saver = pl.callbacks.ModelCheckpoint(dirpath=ckpts_dir, monitor="val_loss", save_top_k=3, mode="min", save_last=True)

    last_ckpt = os.path.join(ckpts_dir, "last.ckpt")
    if not os.path.exists(last_ckpt):
        print("Checkoint does not exist at %s, starting from scratch", last_ckpt)
        last_ckpt = None
    else:
        print("Restoring from: ", last_ckpt)
    eye = MnistFlow()
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        gpus=2,
        callbacks=[ckpt_saver],
        check_val_every_n_epoch=1,
        gradient_clip_val=5,
        log_every_n_steps=1,
        flush_logs_every_n_steps=50,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        resume_from_checkpoint=last_ckpt,
        enable_pl_optimizer=False,

    )
    dts = CelebADTS(55)
    trainer.fit(eye, datamodule=dts)
