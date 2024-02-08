import os
import csv
import copy
import random
import pickle
import glob
import lmdb
import zipfile
import warnings
import joblib

from tqdm import tqdm
import numpy as np
from collections import defaultdict

import torch
from torch.utils import data as torch_data
from torch.nn import functional as F

from torchdrug import datasets, data, utils, core
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("transforms.ProteinViewList")
class ProteinViewList(core.Configurable):

    def __init__(self, view, keys="graph"):
        self.view = view
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def __call__(self, item):
        item = item.copy()
        for key in self.keys:
            graphs = copy.copy(item[key])
            if isinstance(graphs, list):
                for graph in graphs:
                    graph.view = self.view
            else:
                graphs.view = self.view
            item[key] = graphs
        return item


def load_protein(seq, pos):
    residue_type = torch.as_tensor(seq)
    num_residue = len(seq)
    residue_feature = torch.zeros((num_residue, 1), dtype=torch.float)
    residue_number = torch.arange(num_residue)
    num_atom = num_residue
    atom2residue = torch.arange(num_residue)
    node_position = torch.as_tensor(pos)
    atom_type = torch.as_tensor([6 for _ in range(num_atom)])
    atom_name = torch.as_tensor([data.Protein.atom_name2id["CA"] for _ in range(num_atom)])

    edge_list = torch.as_tensor([[0, 0, 0]])
    bond_type = torch.as_tensor([0])

    protein = data.Protein(edge_list, atom_type, bond_type, num_node=num_atom, num_residue=num_residue,
                            node_position=node_position, atom_name=atom_name,
                            atom2residue=atom2residue, residue_feature=residue_feature, 
                            residue_type=residue_type, residue_number=residue_number)
    return protein


@R.register("datasets.MyFold")
class MyFold(data.ProteinDataset):

    def __init__(self, path, split="training", transform=None):
        path = os.path.expanduser(path)
        self.path = path
        self.split = split
        self.transform = transform
        npy_dir = os.path.join(path, 'coordinates', split)
        fasta_file = os.path.join(path, split+'.fasta')

        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(data.Protein.residue_symbol2id.get(amino, 0))
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        fold_classes = {}
        with open(os.path.join(path, 'class_map.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                fold_classes[arr[0]] = int(arr[1])

        protein_folds = {}
        with open(os.path.join(path, split+'.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                protein_folds[arr[0]] = fold_classes[arr[-1]]

        self.data = []
        self.labels = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            protein = load_protein(amino_ids.astype(int), pos)
            self.data.append((protein_name, protein))
            self.labels.append(protein_folds[protein_name])

        self.num_classes = max(self.labels) + 1

    @property
    def tasks(self,):
        return ["targets"]

    def get_item(self, idx):
        protein_name, protein = self.data[idx]
        label = torch.as_tensor(self.labels[idx])

        item = {"graph": protein, "targets": label}
        if self.transform:
            item = self.transform(item)

        return item
    

@R.register("datasets.EC")
class EC(data.ProteinDataset):

    def __init__(self, path, percent=30, split="train", transform=None, **kwargs):
        path = os.path.expanduser(path)
        self.path = path
        self.percent = percent
        self.split = split
        self.transform = transform
        npy_dir = os.path.join(path, 'coordinates')
        fasta_file = os.path.join(path, split+'.fasta')

        test_set = set()
        if split == "test":
            with open(os.path.join(path, "nrPDB-EC_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(data.Protein.residue_symbol2id.get(amino, 0))
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        self.data = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            protein = load_protein(amino_ids.astype(int), pos, **kwargs)
            self.data.append((protein_name, protein))

        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}

        with open(os.path.join(path, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1

                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_ec_list = arr[level_idx]
                        protein_ec_list = protein_ec_list.split(',')
                        for ec in protein_ec_list:
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(ec_annotations)
        self.weights = np.zeros((ec_cnt,), dtype=np.float32)
        for ec, idx in ec_annotations.items():
            self.weights[idx] = len(self.labels)/ec_num[ec]

    @property
    def tasks(self,):
        return ["targets"]

    def get_item(self, idx):
        protein_name, protein = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0
        label = torch.as_tensor(label)

        item = {"graph": protein, "targets": label}

        if self.transform:
            item = self.transform(item)

        return item
    

@R.register("datasets.GO")
class GO(EC):

    def __init__(self, path, level="mf", percent=30, split="train", transform=None):
        path = os.path.expanduser(path)
        self.path = path
        self.percent = percent
        self.split = split
        self.transform = transform
        npy_dir = os.path.join(path, 'coordinates')
        fasta_file = os.path.join(path, split+'.fasta')

        test_set = set()
        if split == "test":
            with open(os.path.join(path, "nrPDB-GO_2019.06.18_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(data.Protein.residue_symbol2id.get(amino, 0))
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        self.data = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            protein = load_protein(amino_ids.astype(int), pos)
            self.data.append((protein_name, protein))

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}
        with open(os.path.join(path, 'nrPDB-GO_2019.06.18_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1 and level == "mf":
                    level_idx = 1
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 5 and level == "bp":
                    level_idx = 2
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 9 and level == "cc":
                    level_idx = 3
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx > 12:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_go_list = arr[level_idx]
                        protein_go_list = protein_go_list.split(',')
                        for go in protein_go_list:
                            if len(go) > 0:
                                protein_labels.append(go_annotations[go])
                                go_num[go] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(go_annotations)

        self.weights = np.zeros((go_cnt,), dtype=np.float32)
        for go, idx in go_annotations.items():
            self.weights[idx] = len(self.labels)/go_num[go]


class FLIPDataset(data.ProteinDataset):

    def load_csv(self, csv_file, sequence_field="sequence", target_fields=None, verbose=0, **kwargs):
        if target_fields is not None:
            target_fields = set(target_fields)

        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
            fields = next(reader)
            train, valid, test = [], [], []
            _sequences = []
            _targets = defaultdict(list)
            for i, values in enumerate(reader):
                for field, value in zip(fields, values):
                    if field == sequence_field:
                        _sequences.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        _targets[field].append(value)
                    elif field == "set":
                        if value == "train":
                            train.append(i)
                        elif value == "test":
                            test.append(i)
                    elif field == "validation":
                        if value == "True":
                            valid.append(i)

        valid_set = set(valid)
        sequences = [_sequences[i] for i in train if i not in valid_set] \
                + [_sequences[i] for i in valid] \
                + [_sequences[i] for i in test]
        targets = defaultdict(list)
        for key, value in _targets.items():
            targets[key] = [value[i] for i in train if i not in valid_set] \
                        + [value[i] for i in valid] \
                        + [value[i] for i in test]
        self.load_sequence(sequences, targets, verbose=verbose, **kwargs)
        self.num_samples = [len(train) - len(valid), len(valid), len(test)]


@R.register("datasets.AAV")
class AAV(FLIPDataset):

    url = "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/aav/splits.zip"
    md5 = "cabdd41f3386f4949b32ca220db55c58"
    splits = ["train", "valid", "test"]
    target_fields = ["target"]
    region = slice(474, 674)

    def __init__(self, path, split="two_vs_many", keep_mutation_region=False, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        path = os.path.join(path, 'aav')
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        assert split in ['des_mut', 'low_vs_high', 'mut_des', 'one_vs_many', 'sampled', 'seven_vs_many', 'two_vs_many']

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        csv_file = os.path.join(data_path, "splits/%s.csv" % split)

        self.load_csv(csv_file, target_fields=self.target_fields, verbose=verbose, **kwargs)
        if keep_mutation_region:
            for i in range(len(self.data)):
                self.data[i] = self.data[i][self.region]
                self.sequences[i] = self.sequences[i][self.region]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.GB1")
class GB1(FLIPDataset):

    url = "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/gb1/splits.zip"
    md5 = "14216947834e6db551967c2537332a12"
    splits = ["train", "valid", "test"]
    target_fields = ["target"]

    def __init__(self, path, split="two_vs_rest", verbose=1, **kwargs):
        path = os.path.expanduser(path)
        path = os.path.join(path, 'gb1')
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        assert split in ['one_vs_rest', 'two_vs_rest', 'three_vs_rest', 'low_vs_high', 'sampled']

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        csv_file = os.path.join(data_path, "splits/%s.csv" % split)

        self.load_csv(csv_file, target_fields=self.target_fields, verbose=verbose, **kwargs)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.Thermostability")
class Thermostability(FLIPDataset):

    url = "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/meltome/splits.zip"
    md5 = "0f8b1e848568f7566713d53594c0ca90"
    splits = ["train", "valid", "test"]
    target_fields = ["target"]

    def __init__(self, path, split="human_cell", verbose=1, **kwargs):
        path = os.path.expanduser(path)
        path = os.path.join(path, 'thermostability')
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        assert split in ['human', 'human_cell', 'mixed_split']

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        csv_file = os.path.join(data_path, "splits/%s.csv" % split)

        self.load_csv(csv_file, target_fields=self.target_fields, verbose=verbose, **kwargs)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits