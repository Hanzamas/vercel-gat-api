"""
GAT Model Architecture for Vercel Deployment
Simplified GAT Model yang sama persis dengan notebook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadGATLayer(nn.Module):
    """
    Enhanced GAT Layer dengan multi-head attention dan layer normalization
    Inspired by HGT but simplified for student-module interaction
    COPIED FROM NOTEBOOK - EXACT MATCH
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int = 2,
                 dropout: float = 0.1, alpha: float = 0.2, concat: bool = True):
        super(MultiHeadGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Multi-head attention parameters
        self.W = nn.Parameter(torch.empty(size=(n_heads, in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * out_features, 1)))

        # Layer normalization (HGT-inspired improvement)
        self.layer_norm = nn.LayerNorm(out_features * n_heads if concat else out_features)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, edge_weights: torch.Tensor = None):
        batch_size, N = h.size(0), h.size(1) if h.dim() == 3 else h.size(0)

        h_heads = []
        attention_weights = []

        for head in range(self.n_heads):
            # Transform features untuk head ini
            h_transformed = torch.mm(h, self.W[head])  # [N, out_features]

            # Compute attention coefficients
            a_input = self._prepare_attentional_mechanism_input(h_transformed)
            e = self.leakyrelu(torch.matmul(a_input, self.a[head]).squeeze(2))

            # Mask dengan adjacency matrix
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

            # Apply edge weights if provided
            if edge_weights is not None:
                attention = attention * edge_weights

            # Softmax
            attention = F.softmax(attention, dim=1)
            attention = self.dropout_layer(attention)

            # Apply attention to features
            h_prime = torch.matmul(attention, h_transformed)  # [N, out_features]

            h_heads.append(h_prime)
            attention_weights.append(attention)

        # Concatenate or average heads
        if self.concat:
            output = torch.cat(h_heads, dim=1)  # [N, n_heads * out_features]
        else:
            output = torch.mean(torch.stack(h_heads), dim=0)  # [N, out_features]

        # Apply layer normalization (HGT-inspired improvement)
        output = self.layer_norm(output)

        return output, attention_weights

    def _prepare_attentional_mechanism_input(self, h):
        N = h.size()[0]  # number of nodes
        h_repeated_in_chunks = h.repeat_interleave(N, dim=0)
        h_repeated_alternating = h.repeat(N, 1)
        all_combinations_matrix = torch.cat([h_repeated_in_chunks, h_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class SimplifiedGATModel(nn.Module):
    """Simplified GAT Model untuk Adaptive Learning focusing on 4 dosen parameters - OPTIMIZED
    COPIED FROM NOTEBOOK - EXACT MATCH
    """
    def __init__(self, n_students: int, n_modules: int,
                 student_features: int = 3, module_features: int = 3,
                 hidden_dim: int = 32, output_dim: int = 16,
                 n_heads: int = 2, dropout: float = 0.3, silent: bool = False):
        super(SimplifiedGATModel, self).__init__()

        self.n_students = n_students
        self.n_modules = n_modules
        self.n_nodes = n_students + n_modules
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if not silent:
            print(f"🔥 OPTIMIZED GAT Model - Dropout: {dropout}")

        # Input embedding for combined features
        max_features = max(student_features, module_features)
        self.input_embedding = nn.Linear(max_features, hidden_dim)

        # Batch normalization (Added)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Simplified GAT layer with optimized dropout
        self.gat_layer = MultiHeadGATLayer(hidden_dim, output_dim, n_heads, dropout*0.5, concat=False)

        # Batch normalization (Added)
        self.bn2 = nn.BatchNorm1d(output_dim)

        # Output predictor untuk student level prediction - optimized
        self.student_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),  # 🔥 Slightly less dropout in predictor
            nn.Linear(hidden_dim // 2, 3)  # Predict level 1, 2, atau 3
        )

    def forward(self, node_features, adj_matrix, edge_weights):
        """Forward pass of SimplifiedGAT - OPTIMIZED"""
        # Input embedding
        h = self.input_embedding(node_features)  # [n_nodes, hidden_dim]

        # Apply Batch normalization
        h = self.bn1(h)

        # GAT layer
        h, attention_weights = self.gat_layer(h, adj_matrix, edge_weights)

        # Apply Batch normalization
        h = self.bn2(h)

        # Extract student embeddings only
        student_embeddings = h[:self.n_students]  # [n_students, output_dim]

        # Generate student level predictions
        student_levels = self.student_predictor(student_embeddings)  # [n_students, 3]

        return {
            'student_levels': student_levels,
            'student_embeddings': student_embeddings,
            'attention_weights': attention_weights,
            'all_embeddings': h
        }