import torch


def train(model, optimizer, criterion, x, x_spe, y):
    model.train()
    optimizer.zero_grad()
    y_ = model(x=x, xcwt=x_spe)
    loss = criterion(y_, y)
    loss.backward()
    optimizer.step()
    corrects = (torch.argmax(y_, dim=1).data == y.data)
    acc = corrects.cpu().int().sum().numpy() / x.size(0)
    return loss, acc


def test(model, criterion, x, x_spe, y):
    x = x.cuda()
    x_spe = x_spe.cuda()
    y = y.cuda()
    model.eval()
    y_ = model(x=x, xcwt=x_spe)
    loss = criterion(y_, y)
    corrects = (torch.argmax(y_, dim=1).data == y.data)
    acc = corrects.cpu().int().sum().numpy() / x.size(0)
    return loss, acc
