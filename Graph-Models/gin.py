import os.path as osp
from typing import Callable, Optional, Union
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

import torch
import torch.nn.functional as F
from torch import Tensor

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv, Linear, Aggregation

import wandb

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

print(data.x.shape)
print(data.edge_index.shape)


class ModifiedGINConv(GINConv):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False, nn2: Callable = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(nn=nn, eps=eps, train_eps=train_eps, **kwargs)
        self.mlp = nn2


    def message(self, x_j: Tensor) -> Tensor:
        # print(f"message shape {x_j.shape}")
        if self.mlp:
            return self.mlp(x_j)
        return x_j


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            if self.mlp is not None:
                out = out + (1 + self.eps) * self.mlp(x_r)
            else:
                out = out + (1 + self.eps) * x_r

        return self.nn(out)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden):
        super().__init__()
        # inner1 = Linear(in_channels=in_channels, out_channels=64)
        inner1 = None
        mlp1 = Linear(in_channels=in_channels, out_channels=16)
        self.conv1 = ModifiedGINConv(mlp1, train_eps=True, nn2=inner1)
        
        # inner2 = Linear(in_channels=16, out_channels=16)
        inner2 = None
        mlp2 = Linear(in_channels=16, out_channels=out_channels)
        self.conv2 = ModifiedGINConv(mlp2, train_eps=True, nn2=inner2)


    def forward(self, x, edge_index, edge_weight):
        x = F.sigmoid(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv2(x, edge_index, edge_weight)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(dataset.num_features, dataset.num_classes, hidden=16).to(device)
data = data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.005)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index, data.edge_attr), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

wandb.init(project="hw3", name="gin")
max_val_acc = 0
records = None
for epoch in range(1, 301):
    loss = train(data)
    train_acc, val_acc, test_acc = test(data)
    if max_val_acc < val_acc:
        max_val_acc = val_acc
        records = epoch, loss, train_acc, val_acc, test_acc
    wandb.log({
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_loss": loss,
        "epoch": epoch
    })
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

print(f"Best epoch = {records[0]} loss = {records[1]}, train_acc = {records[2]}, val_acc = {records[3]}, test_acc = {records[4]}")