import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pruning_Head(nn.Module):
    def __init__(self, n_heads, d_head):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = n_heads * d_head
        self.act = lambda x: F.relu(x) * torch.sigmoid(x)

        self.q_gate = nn.Linear(self.d_model, self.d_model, bias=False)
        # normal_(self.q_gate.weight, 0, 0.1).clamp_(-1., 1.)
        self.k_gate = nn.Linear(self.d_model, self.d_model, bias=False)
        # normal_(self.k_gate.weight, 0, 0.1).clamp_(-1., 1.)

    def gate_func(self, x):
        x = self.gate_proj(x)
        x = self.act(x)
        return x.view(self.n_heads, 1)

    def forward(self, Q, K, X, Y=None):
        """
        qi: (n_heads, 1, d_head)
        ki: (n_heads, n_img, d_head)
        """

        X1 = self.q_gate(X).view(self.n_heads, 1, self.d_head)
        X1 = self.act(X1)
        Y1 = self.k_gate(Y).view(-1, self.n_heads, self.d_head).transpose(0, 1)
        Y1 = self.act(Y1)

        Q = Q.view(self.n_heads, 1, self.d_head) * X1
        K = K.view(self.n_heads, -1, self.d_head) * Y1
        K = K.permute(0, 2, 1)

        logits = torch.matmul(Q, K).squeeze(1).to(torch.float32) / math.sqrt(self.d_head)  # (n_heads, n_img)
        prob = F.softmax(logits, dim=-1)
        return prob.mean(dim=0)
