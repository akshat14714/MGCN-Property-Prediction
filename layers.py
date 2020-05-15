import torch as th
import torch.nn as nn
from torch.nn import Softplus as sp
import dgl.function as fn
import numpy as np

DROPOUT = 0.1
THRESHOLD = 14

def gen_uniq_from_nums(a, b):
    val = (a+b)*(a+b+1) / 2 + b
    return val

class AtomEmbedding(nn.Module):
    def __init__(self, dim=128, type_num=100):
        super().__init__()
        self._dim = dim
        self._type_num = type_num
        self.embedding = nn.Embedding(type_num, dim, padding_idx=0)

    def forward(self, g, p_name="node"):
        g.ndata[p_name] = self.embedding(g.ndata["node_type"])
        return g.ndata[p_name]

class EdgeEmbedding(nn.Module):
    def __init__(self, dim=128, edge_num=3000):
        super().__init__()
        self._dim, self._edge_num = dim, edge_num
        self.embedding = nn.Embedding(edge_num, dim, padding_idx=0)

    def generate_edge_type(self, edges):
        atom_type_x, atom_type_y = edges.src["node_type"], edges.dst["node_type"]
        return {
            "type":
            gen_uniq_from_nums(atom_type_x, atom_type_y)
        }

    def forward(self, g, p_name="edge_f"):
        g.apply_edges(self.generate_edge_type)
        return self.embedding(g.edata["type"])

class ShiftSoftplus(sp):
    def __init__(self, beta=1, shift=2, threshold=20):
        super().__init__(beta, threshold)
        self.shift, self.sp = shift, sp(beta, threshold)

    def forward(self, input):
        cc = self.sp(input) - np.log(float(self.shift))
        return cc

class RBFLayer(nn.Module):
    def __init__(self, low=0, high=30, gap=0.1, dim=1):
        super().__init__()
        self._low, self._high = low, high
        self._gap, self._dim = gap, dim
        self._n_centers = int(np.ceil((high - low) / gap))
        centers = np.linspace(low, high, self._n_centers)
        self.centers = th.tensor(centers, dtype=th.float, requires_grad=False)
        self.centers = nn.Parameter(self.centers, requires_grad=False)
        self._fan_out = self._dim * self._n_centers
        self._gap = centers[1] - centers[0]

    def dis2rbf(self, edges):
        rbf = th.exp((-1 / self._gap) * ((edges.data["distance"] - self.centers)**2))
        return {"rbf": rbf}

    def forward(self, g):
        g.apply_edges(self.dis2rbf)
        return g.edata["rbf"]

class VEConv(nn.Module):
    def __init__(self, rbf_dim, dim=64, update_edge=True):
        super().__init__()
        self._rbf_dim, self._dim, self._update_edge = rbf_dim, dim, update_edge
        self.linear_layer1, self.linear_layer2 = nn.Linear(self._rbf_dim, self._dim), nn.Linear(self._dim, self._dim)
        self.linear_layer3, self.dropout_layer = nn.Linear(self._dim, self._dim), nn.Dropout(p=DROPOUT)
        self.activation = nn.sp(beta=0.5, threshold=THRESHOLD)

    def update_rbf(self, edges):
        h = self.linear_layer1(edges.data["rbf"])
        h = self.linear_layer2(self.activation(h))
        return {"h": h}

    def update_edge(self, edges):
        h = self.linear_layer4(self.linear_layer3(edges.data["edge_f"]))
        return {"edge_f": h}

    def forward(self, g):
        g.apply_edges(self.update_rbf)
        if self._update_edge:
            g.apply_edges(self.update_edge)
        g.update_all(message_func=[fn.u_mul_e("new_node", "h", "m_0"), fn.copy_e("edge_f", "m_1")],
                     reduce_func=[fn.sum("m_0", "new_node_0"), fn.sum("m_1", "new_node_1")])
        g.ndata["new_node"] = g.ndata.pop("new_node_0") + g.ndata.pop("new_node_1")
        return g.ndata["new_node"]

class MultiLevelInteraction(nn.Module):
    def __init__(self, rbf_dim, dim):
        super().__init__()

        self._atom_dim, self.activation = dim, nn.sp(beta=0.5, threshold=THRESHOLD)
        self.node_layer1, self.edge_layer1 = nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True)
        self.conv_layer, self.node_layer2 = VEConv(rbf_dim, dim), nn.Linear(dim, dim)
        self.node_layer3, self.dropout_layer = nn.Linear(dim, dim), nn.Dropout(p=DROPOUT)

    def forward(self, g, level=1):
        g.ndata["new_node"] = self.node_layer1(g.ndata["node_%s" %
                                                       (level - 1)])
        node = self.dropout_layer(self.conv_layer(g))
        g.edata["edge_f"] = self.activation(self.edge_layer1(g.edata["edge_f"]))
        new_node = self.node_layer3(self.activation(self.node_layer2(node)))

        g.ndata["node_%s" % (level)] = g.ndata["node_%s" %
                                               (level - 1)] + new_node

        return g.ndata["node_%s" % (level)]
