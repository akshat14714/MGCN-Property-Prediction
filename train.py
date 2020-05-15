''' Please find the pre-processed dataset file here: https://drive.google.com/file/d/1-TqjK8t5GKJteSBOwjG0udTmBDegoREg/view?usp=sharing'''

import pickle
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from mgcn import MGCNModel
from alchemy import AlchemyDataset, batcher

BATCH_SIZE = 20
SHUFFLE_BOOL = False
NUM_WORKERS = 0
EPOCHS = 250
OUTPUT_DIM = 12
LEARNING_RATE = 0.0001
MODEL_PATH = './models/model_'
DEVICE = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

with open("./alchemy_dataset", "rb") as fp:
    alchemy_dataset1 = pickle.load(fp)

alchemy_loader1 = DataLoader(dataset=alchemy_dataset1, batch_size=BATCH_SIZE, collate_fn=batcher(),
                            shuffle=SHUFFLE_BOOL, num_workers=NUM_WORKERS)

model = MGCNModel(norm=True, output_dim=OUTPUT_DIM)
device = DEVICE
model.set_mean_std(alchemy_dataset1.mean, alchemy_dataset1.std, device)
model.to(device)
loss_fn, MAE_fn, optimizer = nn.MSELoss(), nn.L1Loss(), th.optim.Adam(model.parameters(), lr=LEARNING_RATE)
epochs = EPOCHS

for epoch in range(epochs):
    w_loss, w_mae = 0, 0
    model.train()
    for idx, batch in enumerate(alchemy_loader1):
        batch.graph.to(device)
        batch.label = batch.label.to(device)
        res = model(batch.graph)
        loss = loss_fn(res, batch.label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        w_mae = w_mae + MAE_fn(res, batch.label).detach().item()
        w_loss = w_loss + loss.detach().item()
    w_mae /= idx + 1

    if epoch%10==0:
        checkpoint = {'model': MGCNModel(norm=True, output_dim=12), 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        th.save(checkpoint, MODEL_PATH + str(epoch) + '.pth')

    print("Epoch {:2d}, loss: {:.7f}, mae: {:.7f}".format(
        epoch, w_loss, w_mae))
