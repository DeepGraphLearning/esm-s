import os
import sys
import math
import pprint
import pickle
import random

from tqdm import tqdm

import numpy as np

import torch
from torch import nn

from torchdrug import core, tasks, datasets, utils, metrics, data
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from esm_s import dataset, task


@torch.no_grad()
def dump(cfg, dataset, task):
    dataloader = data.DataLoader(dataset, cfg.batch_size, shuffle=False, num_workers=0)
    device = torch.device(cfg.gpus[0])
    task = task.cuda(device)
    task.eval()
    preds = []
    target = []
    for batch in tqdm(dataloader):
        batch = utils.cuda(batch, device=device)
        graph = batch["graph"]
        if task.graph_construction_model:
            graph = task.graph_construction_model(graph)
        output = task.model(graph, graph.node_feature.float())
        preds.append(output["graph_feature"].detach())
        target.append(batch["targets"].detach())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(target, dim=0)
    return pred, target


@torch.no_grad()
def retrieve(cfg, train_keys, train_targets, test_keys, test_targets, task):
    cos_sim = nn.CosineSimilarity(dim=1)
    preds = []
    for i in tqdm(range(0, len(test_keys), cfg.batch_size)):
        test_key = test_keys[i:i+cfg.batch_size]
        sim = -cos_sim(test_key.unsqueeze(-1), train_keys.transpose(0, 1).unsqueeze(0))   # (batch_size, num_train)
        retrieval_items = sim.argsort(dim=1)[:, :cfg.knn]     # (num_test, k)
        pred = train_targets[retrieval_items]   # (num_test, k, num_tasks)
        if cfg.weighted == "linear":
            _sim = -torch.gather(sim, 1, retrieval_items).unsqueeze(-1)
        elif cfg.weighted == "exp":
            _sim = ((-torch.gather(sim, 1, retrieval_items).unsqueeze(-1)) / cfg.temp).exp()
        else:
            _sim = torch.ones((pred.shape[0], pred.shape[1], 1), device=pred.device, dtype=torch.float)
        pred = (pred * _sim).sum(dim=1) / _sim.sum(dim=1)
        preds.append(pred)
    pred = torch.cat(preds, dim=0)
    return task.evaluate(pred, test_targets)


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger(file=False)
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    assert cfg.dataset["class"] in ["EC", "GO"]
    cfg.dataset.split = "train"
    train_set = core.Configurable.load_config_dict(cfg.dataset)
    cfg.dataset.split = "valid"
    valid_set = core.Configurable.load_config_dict(cfg.dataset)
    cfg.dataset.split = "test"
    cfg.dataset.percent = 95
    test_set95 = core.Configurable.load_config_dict(cfg.dataset)
    print(test_set95)
    cfg.dataset.percent = 50
    test_set50 = core.Configurable.load_config_dict(cfg.dataset)
    print(test_set50)
    cfg.dataset.percent = 30
    test_set30 = core.Configurable.load_config_dict(cfg.dataset)
    print(test_set30)
    dataset = (train_set, valid_set, test_set95)

    cfg.task.num_class = valid_set.num_classes
    task = core.Configurable.load_config_dict(cfg.task)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        task.model.load_state_dict(model_dict)

    train_keys, train_targets = dump(cfg, train_set, task)
    valid_keys, valid_targets = dump(cfg, valid_set, task)
    test95_keys, test95_targets = dump(cfg, test_set95, task)
    test50_keys, test50_targets = dump(cfg, test_set50, task)
    test30_keys, test30_targets = dump(cfg, test_set30, task)

    valid_metric = retrieve(cfg, train_keys, train_targets, valid_keys, valid_targets, task)
    print("Metrics on valid set:", valid_metric)
    test95_metric = retrieve(cfg, train_keys, train_targets, test95_keys, test95_targets, task)
    print("Metrics on test set with 0.95 cutoff:", test95_metric)
    test50_metric = retrieve(cfg, train_keys, train_targets, test50_keys, test50_targets, task)
    print("Metrics on test set with 0.5 cutoff:", test50_metric)
    test30_metric = retrieve(cfg, train_keys, train_targets, test30_keys, test30_targets, task)
    print("Metrics on test set with 0.3 cutoff:", test30_metric)