import torch.nn

class Atenttion(nn.Module):

    def __init__(self, d_model, n_tokens):
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        Q = self.W_q @ x
        K = self.W_k @ x
        V = self.W_v @ x
        A = Q @ K.transpose()