import zipfile
import pathlib
import pandas as pds
import numpy as np
import os
import os.path as osp
from rdkit import Chem
import dgl
from rdkit import RDConfig as rdf
from dgl.data.utils import download
from rdkit.Chem import ChemicalFeatures as cf
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from torch.utils.data import DataLoader

BASE_FT = 'BaseFeatures.fdef'
SELF_LOOP = False
MODE = 'dev'
DEV_TGT = 'dev_target.csv'
OUTPUT_DIM = 12
E_FEAT = 'e_feat'
POSITION = 'pos'
N_FEAT = 'n_feat'
DISTANCE = 'distance'
GDB_IDX = 'gdb_idx'
ATOMS = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
ALC_PATH = './Alchemy_data'


class AlchemyBatcher:
    def __init__(self, graph=None, label=None):
        self.graph = graph
        self.label = label

def batcher():
    def batcher_dev(batch):
        graphs, labels = zip(*batch)
        return AlchemyBatcher(graph=dgl.batch(graphs),
                              label=torch.stack(labels, 0))
    return batcher_dev

class AlchemyDataset(Dataset):
    chem_feature_factory = cf.BuildFeatureFactory(osp.join(rdf.RDDataDir, BASE_FT))

    def alchemy_nodes(self, mol):
        atom_feats_dict, is_donor, is_acceptor = defaultdict(list), defaultdict(int), defaultdict(int)
        ddir = rdf.RDDataDir
        mol_featurizer = cf.BuildFeatureFactory(osp.join(ddir, BASE_FT))
        mol_conformers, mol_feats = mol.GetConformers(), mol_featurizer.GetFeaturesForMol(mol)
        geom = mol_conformers[0].GetPositions()
        for i in range(len(mol_feats)):
            if mol_feats[i].GetFamily() == 'Acceptor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_acceptor[u] = 1
            elif mol_feats[i].GetFamily() == 'None':
                continue
            elif mol_feats[i].GetFamily() == 'Donor':
                node_list = mol_feats[i].GetAtomIds()
                for u in node_list:
                    is_donor[u] = 1

        num_atoms = mol.GetNumAtoms()
        for u in range(num_atoms):
            atom = mol.GetAtomWithIdx(u)
            h_u = []
            symbol, atom_type, aromatic = atom.GetSymbol(), atom.GetAtomicNum(), atom.GetIsAromatic()
            atom_feats_dict['node_type'].append(atom_type)
            hybridization, num_h = atom.GetHybridization(), atom.GetTotalNumHs()
            atom_feats_dict[POSITION].append(torch.FloatTensor(geom[u]))
            h_u = h_u + [int(symbol == x) for x in ATOMS ]
            h_u.append(atom_type)
            h_u.append(int(aromatic))
            h_u.append(is_donor[u])
            h_u.append(is_acceptor[u])
            h_u += [
                int(hybridization == x)
                for x in (Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3)]
            h_u.append(num_h)
            atom_feats_dict[N_FEAT].append(torch.FloatTensor(h_u))
        atom_feats_dict['node_type'] = torch.LongTensor(atom_feats_dict['node_type'])
        atom_feats_dict[N_FEAT] = torch.stack(atom_feats_dict[N_FEAT], dim=0)
        atom_feats_dict[POSITION] = torch.stack(atom_feats_dict['pos'], dim=0)
        return atom_feats_dict

    def alchemy_edges(self, mol, self_loop=True):
        bond_feats_dict, mol_conformers = defaultdict(list), mol.GetConformers()
        geom, num_atoms = mol_conformers[0].GetPositions(), mol.GetNumAtoms()
        for u in range(num_atoms):
            for v in range(num_atoms):
                if u == v and not self_loop:
                    continue
                e_uv = mol.GetBondBetweenAtoms(u, v)
                if e_uv is None:
                    bond_type = None
                else:
                    bond_type = e_uv.GetBondType()
                bond_feats_dict[E_FEAT].append([
                    float(bond_type == x)
                    for x in (Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, None)])
                bond_feats_dict[DISTANCE].append(np.linalg.norm(geom[u] - geom[v]))

        bond_feats_dict[E_FEAT] = torch.FloatTensor(bond_feats_dict[E_FEAT])
        bond_feats_dict[DISTANCE] = torch.FloatTensor(bond_feats_dict[DISTANCE]).reshape(-1, 1)
        return bond_feats_dict

    def sdf_to_dgl(self, sdf_file, self_loop=SELF_LOOP):
        mol, g = Chem.MolFromMolBlock(open(str(sdf_file)).read(), removeHs=False), dgl.DGLGraph()
        num_atoms, atom_feats = mol.GetNumAtoms(), self.alchemy_nodes(mol)
        g.add_nodes(num=num_atoms, data=atom_feats)
        if self_loop:
            g.add_edges(
                [i for i in range(num_atoms) for j in range(num_atoms)],
                [j for i in range(num_atoms) for j in range(num_atoms)])
        else:
            g.add_edges(
                [i for i in range(num_atoms) for j in range(num_atoms - 1)], [
                    j for i in range(num_atoms)
                    for j in range(num_atoms) if i != j
                ])

        bond_feats = self.alchemy_edges(mol, self_loop)
        g.edata.update(bond_feats)
        l = torch.FloatTensor(self.target.loc[int(sdf_file.stem)].tolist()) \
            if self.mode == MODE else torch.LongTensor([int(sdf_file.stem)])
        return (g, l)

    def __init__(self, mode = MODE , transform=None):
        self.mode, self.transform, self.file_dir = mode, transform, pathlib.Path(ALC_PATH, mode)
        self._load()

    def normalize(self, mean=None, std=None):
        if std is None:
            std = np.std(np.array([i.np() for i in self.labels]), axis=0)
        if mean is None:
            mean = np.mean(np.array([i.np() for i in self.labels]), axis=0)
        self.mean = mean
        self.std = std

    def _load(self):
        if self.mode == MODE:
            self.target = pds.read_csv(pathlib.Path(self.file_dir, DEV_TGT), index_col=0,
                                       usecols=[GDB_IDX,] + ['property_%d' % x for x in range(OUTPUT_DIM)])
            self.target = self.target[['property_%d' % x for x in range(OUTPUT_DIM)]]

        self.graphs, self.labels = [], []
        for sdf_file in pathlib.Path(self.file_dir, "sdf").glob("**/*.sdf"):
            result = self.sdf_to_dgl(sdf_file)
            if result is None:
                continue
            self.graphs.append(result[0])
            self.labels.append(result[1])
        self.normalize()
        print(len(self.graphs), "loaded!")
