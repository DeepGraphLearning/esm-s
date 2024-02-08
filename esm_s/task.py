from collections.abc import Sequence

import time

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add

from torchdrug import core, layers, tasks, metrics, data
from torchdrug.core import Registry as R


class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)
    

@R.register("tasks.FunctionAnnotation")
class FunctionAnnotation(tasks.Task, core.Configurable):

    eps = 1e-10
    _option_members = {"metric"}

    def __init__(self, model, num_class=1, metric=('auprc@micro', 'f1_max'), weight=None, graph_construction_model=None, 
                 mlp_batch_norm=False, mlp_dropout=0, verbose=0):
        super(FunctionAnnotation, self).__init__()
        self.model = model
        if weight is None:
            weight = torch.ones((num_class,), dtype=torch.float)
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.metric = metric
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        self.mlp = MLP(in_channels=self.model.output_dim,
                       mid_channels=self.model.output_dim,
                       out_channels=num_class,
                       batch_norm=mlp_batch_norm,
                       dropout=mlp_dropout)

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        loss_fn = torch.nn.BCELoss(weight=torch.as_tensor(self.weight))
        loss = loss_fn(pred.sigmoid(), target)
        
        name = tasks._get_criterion_name("bce")
        metric[name] = loss
        all_loss += loss

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        graph_feature = output["graph_feature"]
        pred = self.mlp(graph_feature)
        return pred

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


@R.register("tasks.FoldClassification")
class FoldClassification(tasks.Task, core.Configurable):

    eps = 1e-10
    _option_members = {"metric"}

    def __init__(self, model, num_class=1, metric=('acc'), graph_construction_model=None, 
                 mlp_batch_norm=False, mlp_dropout=0, verbose=0):
        super(FoldClassification, self).__init__()
        self.model = model
        self.metric = metric
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        self.mlp = MLP(in_channels=self.model.output_dim,
                       mid_channels=self.model.output_dim,
                       out_channels=num_class,
                       batch_norm=mlp_batch_norm,
                       dropout=mlp_dropout)

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(pred, target)
        
        name = tasks._get_criterion_name("ce")
        metric[name] = loss
        all_loss += loss

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        graph_feature = output["graph_feature"]
        pred = self.mlp(graph_feature)
        return pred

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "acc":
                score = metrics.accuracy(pred.squeeze(-1), target.long())
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric