import dgl
import torch as th
import torch.nn as nn
from layers import AtomEmbedding, RBFLayer, EdgeEmbedding, MultiLevelInteraction

DIM = 128
OUTPUT_DIM = 1
EDGE_DIM = 128
CUTOFF = 5.0
WIDTH = 1
THRESHOLD = 20
NDL = 64

class MGCNModel(nn.Module):
    def __init__(self,
                 dim=DIM,
                 output_dim=OUTPUT_DIM,
                 edge_dim=EDGE_DIM,
                 cutoff=CUTOFF,
                 width=WIDTH,
                 n_conv=3,
                 norm=False,
                 atom_ref=None):
        super().__init__()
        self.name = "MGCN"
        self._dim, self.output_dim = dim, output_dim
        self.edge_dim, self.cutoff, self.atom_ref = edge_dim, cutoff, atom_ref
        self.width = width
        self.n_conv, self.norm = n_conv, norm
        self.activation = nn.Softplus(beta=1, threshold=THRESHOLD)

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1)
        self.embedding_layer, self.edge_embedding_layer = AtomEmbedding(dim), EdgeEmbedding(dim=edge_dim)
        self.rbf_layer = RBFLayer(0, cutoff, width)
        self.conv_layers = nn.ModuleList([
            MultiLevelInteraction(self.rbf_layer._fan_out, dim)
            for i in range(n_conv)
        ])

        self.node_dense_layer1, self.node_dense_layer2 = nn.Linear(dim * (self.n_conv + 1), NDL), nn.Linear(NDL, output_dim)

    def set_mean_std(self, mean, std, device):
        self.mean_per_node, self.std_per_node = th.tensor(mean, device=device), th.tensor(std, device=device)

    def forward(self, g):
        self.embedding_layer(g, "node_0")
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)
        self.edge_embedding_layer(g)
        for idx in range(self.n_conv):
            self.conv_layers[idx](g, idx + 1)
        g.ndata["node"] = th.cat(tuple(g.ndata["node_%d" % (i)]
                                 for i in range(self.n_conv + 1)), 1)
        node = self.node_dense_layer1(g.ndata["node"])
        node = self.activation(node)
        g.ndata["res"] = self.node_dense_layer2(node)

        if self.atom_ref is not None:
            g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]

        if self.norm:
            k1 = g.ndata["res"] * self.std_per_node
            g.ndata["res"] = k1 + self.mean_per_node
        res = dgl.sum_nodes(g, "res")
        return res
