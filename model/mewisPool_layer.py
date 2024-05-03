import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch_geometric.utils import get_laplacian


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, enhance=False):
        super(MLP, self).__init__()

        self.enhance = enhance

        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        if enhance:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        if self.enhance:
            x = self.bn1(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = self.fc2(x)
        if self.enhance:
            x = self.bn2(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)

        x = self.fc3(x)

        return x


class MEWISPool(nn.Module):
    def __init__(self, hidden_dim, device):
        super(MEWISPool, self).__init__()
        self.device = device

        self.gc1 = GINConv(MLP(1, hidden_dim, hidden_dim).to(self.device))
        self.gc2 = GINConv(
            MLP(hidden_dim, hidden_dim, hidden_dim).to(self.device)
        )
        self.gc3 = GINConv(MLP(hidden_dim, hidden_dim, 1).to(self.device))

    def compute_entropy(self, x, L, A, batch):
        # computing local variations; Eq. (5)
        V = (
            x * torch.matmul(L, x)
            - x * torch.matmul(A, x)
            + torch.matmul(A, x * x)
        )
        V = torch.norm(V, dim=1)

        P = torch.cat(
            [torch.softmax(V[batch == i], dim=0) for i in torch.unique(batch)]
        )
        P[P == 0.0] += 1

        H = -P * torch.log(P)

        return H.unsqueeze(-1).to(self.device)

    def loss_fn(self, entropies, probabilities, A, gamma):
        entropy_proba = torch.matmul(entropies.t(), probabilities)[0, 0]
        proba_proba = torch.matmul(
            torch.matmul(probabilities.t(), A), probabilities
        ).sum()
        return gamma - entropy_proba + proba_proba

    def conditional_expectation(
        self, entropies, probabilities, A, threshold, gamma
    ):
        sorted_probabilities = torch.sort(
            probabilities, descending=True, dim=0
        )

        dummy_probabilities = probabilities.detach().clone()
        selected = set()
        rejected = set()

        for i in range(sorted_probabilities.values.size(0)):
            node_index = sorted_probabilities.indices[i].item()
            neighbors = torch.where(A[node_index] == 1)[0]
            if len(neighbors) == 0:
                selected.add(node_index)

            if node_index not in rejected and node_index not in selected:
                s = dummy_probabilities.clone()
                s[node_index] = 1
                s[neighbors] = 0

                loss = self.loss_fn(entropies, s, A, gamma)

                if loss <= threshold:
                    selected.add(node_index)
                    for n in neighbors.tolist():
                        rejected.add(n)

                    dummy_probabilities[node_index] = 1
                    dummy_probabilities[neighbors] = 0

        mewis_indices = sorted(list(selected))
        return mewis_indices

    def graph_reconstruction(self, mewis_indices, x, A):
        x_pooled = x[mewis_indices]

        A2 = torch.matmul(A, A)
        A3 = torch.matmul(A2, A)

        A2 = A2[mewis_indices][:, mewis_indices]
        A3 = A3[mewis_indices][:, mewis_indices]

        identity = torch.eye(len(mewis_indices)).to(self.device)
        one = torch.ones([len(mewis_indices), len(mewis_indices)]).to(
            self.device
        )

        adj_pooled = (one - identity) * torch.clamp(A2 + A3, min=0, max=1)

        return x_pooled, adj_pooled

    def to_edge_index(self, adj_pooled, mewis, batch):
        row1, row2 = torch.where(adj_pooled > 0)
        edge_index_pooled = torch.cat(
            [row1.unsqueeze(0), row2.unsqueeze(0)], dim=0
        )
        batch_pooled = batch[mewis]

        return edge_index_pooled, batch_pooled

    def forward(self, x, edge_index, batch):
        # computing the graph laplacian and adjacency matrix
        batch_nodes = batch.size(0)
        L_indices, L_weights = get_laplacian(edge_index)
        L = torch.sparse_coo_tensor(
            L_indices, L_weights, torch.Size([batch_nodes, batch_nodes])
        ).to(self.device)
        A = (torch.diag(torch.diag(L.to_dense())) - L.to_dense()).to(
            self.device
        )

        entropies = self.compute_entropy(x, L, A, batch)

        # graph convolution and probability scores
        probabilities = self.gc1(entropies, edge_index)
        probabilities = self.gc2(probabilities, edge_index)
        probabilities = self.gc3(probabilities, edge_index)
        probabilities = torch.sigmoid(probabilities).to(self.device)

        # conditional expectation; Algorithm 1
        gamma = entropies.sum().to(self.device)
        loss = self.loss_fn(entropies, probabilities, A, gamma)  # Eq. (9)

        mewis_indices = self.conditional_expectation(
            entropies, probabilities, A, loss, gamma
        )

        # graph reconstruction
        x_pooled, adj_pooled = self.graph_reconstruction(mewis_indices, x, A)
        edge_index_pooled, batch_pooled = self.to_edge_index(
            adj_pooled, mewis_indices, batch
        )

        return x_pooled, edge_index_pooled, batch_pooled, loss
