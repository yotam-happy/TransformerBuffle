from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

N_POS = 3
N_VAL = 2

D_MODEL = 64
NHEAD = 4
NLAYERS = 4
LR = 0.0001
BATCH_SZ = 32


class TestTask(LightningModule):
    def __init__(self):
        super().__init__()
        self.pos_embd = nn.Embedding(N_POS, D_MODEL)
        self.val_embd = nn.Embedding(N_VAL, D_MODEL)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, dim_feedforward=D_MODEL * 4,
                                                            nhead=NHEAD, activation='relu')
        self.model = nn.TransformerEncoder(self.transformer_layer, num_layers=NLAYERS)
        self.cls = nn.Linear(D_MODEL, N_VAL)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x):
        pos = torch.range(0, x.size(1) - 1, dtype=torch.int64).repeat(x.size(0), 1)
        inp = self.val_embd(x) + self.pos_embd(pos)
        return self.cls(self.model(inp))

    def step(self, batch):
        # predict value at position k+1 using output at position k.
        # As the sequence is random, the model should learn to use the positional embeddings
        # to attend to the next position of the sequence and retrieve the value there.
        outputs = self.forward(batch[0])[:, :-1, :]
        y = batch[0][:, 1:]

        loss = self.ce_loss(outputs.flatten(end_dim=-2), y.flatten())
        acc = (outputs.flatten(end_dim=-2).argmax(axis=-1) == y.flatten()).sum() / y.flatten().size(0)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("acc", acc, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        return loss

    def configure_optimizers(self):
        return optim.Adam(list(self.parameters()), lr=LR)


class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_ds = TensorDataset(torch.randint(N_VAL, size=[1000000, N_POS]))
        self.val_ds = TensorDataset(torch.randint(N_VAL, size=[3000, N_POS]))
        self.test_ds = TensorDataset(torch.randint(N_VAL, size=[3000, N_POS]))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SZ)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=BATCH_SZ)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=BATCH_SZ)


def main():
    trainer = pl.Trainer(deterministic=True, gpus=0, max_epochs=1, val_check_interval=500)
    trainer.fit(TestTask(), datamodule=MyDataModule())


if __name__ == '__main__':
    main()
