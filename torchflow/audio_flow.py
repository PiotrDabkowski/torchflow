import pytorch_lightning as pl
from pl_bolts.datamodules import mnist_datamodule
# from .flow import *
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets.librispeech import LIBRISPEECH
import torchaudio
from torch.utils.data import random_split
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, LSUN, CelebA
import torch.nn.functional as F
import numpy as np
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
import random

torchaudio.backend.set_audio_backend("sox_io")
#  1 SEC ~= 8196
#  8196  SQ
#  4098  SQ
#  2048  4ch: N Quartz 32kernel, SQ
#  1024  8ch: N Quartz 32kernel, Separate 50%, SQ
#  512   8ch: N Quartz 32kernel, SQ
#  256   16ch: N Quartz 32kernel, Separate 50%, SQ
#  128   16ch: N Quartz 32kernel, SQ
#  64    32ch: N Quartz 32kernel, Separate 50%, SQ
#  32    32ch: N Quartz 16kernel, SQ
#  16    64ch: N Quartz 8kernel SQ
#  8     128ch: N Quartz 4kernel, Terminate




class LibiSpeechUnsupervisedDts(Dataset):
    NATIVE_SAMPLE_RATE = 16000
    def __init__(self, libri: LIBRISPEECH, waveform_transforms=None):
        super().__init__()
        self.libri = libri
        self.waveform_transforms = waveform_transforms

    def __len__(self):
        return len(self.libri)

    def __getitem__(self, item):
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = self.libri[item]
        assert sample_rate == self.NATIVE_SAMPLE_RATE
        if self.waveform_transforms is not None:
            waveform = self.waveform_transforms(waveform)
        return waveform


class WaveformSample:
    def __init__(self, sample_len, allowed_empty_frac):
        self.sample_len = sample_len
        self.allowed_empty_frac = allowed_empty_frac

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        b, s = waveform.shape
        assert b == 1
        min_padding = max((self.sample_len - s) // 2 + 2, 0)
        pad_size = max(int(self.allowed_empty_frac * self.sample_len), min_padding)
        waveform = F.pad(waveform, (pad_size, pad_size))
        b, s2 = waveform.shape
        assert s + 2*pad_size == s2
        assert s2 > self.sample_len
        start_pos = random.randrange(s2-self.sample_len)
        print(start_pos)
        return waveform[:, start_pos:start_pos+self.sample_len].contiguous()

class LibriSpeechDataModule(pl.LightningDataModule):
    def __init__(self, root="/Users/piter/datasets",  batch_size=256, train_fraction=0.9, sample_freq=8000, sample_len=12000, allowed_empty_frac=0.3):
        super().__init__()
        self.batch_size = batch_size
        self.train_size = None
        self.val_size = None
        self.root = root
        self.train_fraction = train_fraction

        self.transform_fn = Compose([
            torchaudio.transforms.Resample(LibiSpeechUnsupervisedDts.NATIVE_SAMPLE_RATE, sample_freq),
            WaveformSample(sample_len, allowed_empty_frac)
        ])
        self.prepare_data()

    def prepare_data(self):
        d = LIBRISPEECH(self.root, download=True)
        self.train_size = int(0.9 * len(d))
        self.val_size = len(d) - self.train_size

    def _split(self, dts):
        assert isinstance(dts, Dataset)
        return random_split(
            dts,
            [self.train_size, self.val_size],
            generator=torch.Generator().manual_seed(11),
        )

    def train_dataloader(self) -> DataLoader:
        train_dts = self._split(LIBRISPEECH(self.root, download=True))[0]
        return DataLoader(
            LibiSpeechUnsupervisedDts(train_dts, waveform_transforms=self.transform_fn),
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_dts = self._split(LIBRISPEECH(self.root, download=True))[1]
        return DataLoader(
            LibiSpeechUnsupervisedDts(val_dts, waveform_transforms=self.transform_fn),
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=True,
        )

if __name__ == '__main__':
    x = LibriSpeechDataModule(batch_size=10, sample_len=12000)
    for e in x.train_dataloader():
        print(e.shape)
        for i in range(e.shape[0]):
            torchaudio.save('dupa/chuj%d.flac' % i, e[i], 8000)

        break