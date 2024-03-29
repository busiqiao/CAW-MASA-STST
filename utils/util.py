import torch
import torch.nn as nn


def train(model, optimizer, criterion, x, x_spe, y):
    model.train()
    optimizer.zero_grad()
    y_ = model(x, x_spe)
    loss = criterion(y_, y)
    loss.backward()
    optimizer.step()
    return loss, y_


def test(model, criterion, x, x_spe, y):
    x = x.cuda()
    x_spe = x_spe.cuda()
    y = y.cuda()
    model.eval()
    y_ = model(x, x_spe)
    loss = criterion(y_, y)
    corrects = (torch.argmax(y_, dim=1).data == y.data)
    acc = corrects.cpu().int().sum().numpy()
    return loss, acc
