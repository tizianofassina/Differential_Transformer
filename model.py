import torch
import torch.nn as nn
from normalization import GroupNorm, RMSNorm
from swiglu import SwiGLU


class Attention(nn.Module):
    def __init__(self, h, d, hidden_dim, l_init,  device = 'cpu'):
        super().__init__()
        self.h = h # numbe of heads
        self.hidden_dim = hidden_dim # dimension for attention
        self.d = d

        # Attention weights
        self.q_linear = nn.Linear(hidden_dim, d*h, bias=False).to(device)
        self.k_linear = nn.Linear(hidden_dim, d*h, bias=False).to(device)
        self.v_linear = nn.Linear(hidden_dim, d*h, bias=False).to(device)

        # Linear projection
        self.o_linear = nn.Linear(d*h, hidden_dim, bias=False).to(device)

        self.l_init = l_init
        self.g_norm = GroupNorm(self.d, l_init).to(device)

    def forward(self, x, attention_mask=None):
        q = self.q_linear(x)  # batch, n , d*h
        k = self.k_linear(x)  # batch, n , d*h
        v = self.v_linear(x)  # batch, n , d*h

        v = v.view(x.shape[0], x.shape[1], self.h, self.d)  # batch, n , h, d
        q = q.view(x.shape[0], x.shape[1], self.h, self.d)  # batch, n , h, d
        k = k.view(x.shape[0], x.shape[1], self.h, self.d)  # batch, n , h, d

        q = q.permute(0, 2, 1, 3)  # batch, h, n, d
        k = k.permute(0, 2, 3, 1)  # batch, h, d, n
        v = v.permute(0, 2, 1, 3)  # batch, h, n, d

        q_k = torch.matmul(q, k)/torch.sqrt(torch.tensor(self.d))  # batch, h, n, n

        causal_mask = torch.tril(torch.ones(x.shape[1], x.shape[1], device=x.device)).unsqueeze(0).unsqueeze(1)
        # => [1, 1, n, n], which will broadcast to [B, h, n, n]
        q_k = q_k.masked_fill(causal_mask == 0, float('-inf'))

        if attention_mask is not None:
            # Expand to [B, 1, 1, n]
            # So it can broadcast to [B, h, n, n] along h and the "row" dimension
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # => [B, 1, 1, n]

            # Where attn_mask=0 => set q_k to -inf
            q_k = q_k.masked_fill(attn_mask == 0, float('-inf'))

        soft = torch.softmax(q_k, dim=3)  # batch, h, n, n

        att = torch.matmul(soft, v)

        att = self.g_norm(att)  # batch, h, n, d

        att = att.permute(0, 2, 1, 3)  # batch, n, h, d
        att = att.reshape((x.shape[0], x.shape[1], self.h * self.d))  # batch, n, d*h
        att = self.o_linear(att)  # batch, n, hidden_dim
        return att


class DifferentialAttention(nn.Module):
    def __init__(self, h, d, hidden_dim, l_init, device = 'cpu'):
        super().__init__()
        self.h = h
        self.hidden_dim = hidden_dim
        self.d = d

        # Attention weights
        self.q_linear = nn.Linear(hidden_dim, 2 * d * h, bias=False).to(device)
        self.k_linear = nn.Linear(hidden_dim, 2 * d * h, bias=False).to(device)
        self.v_linear = nn.Linear(hidden_dim, 2 * d * h, bias=False).to(device)

        # Linear projection
        self.o_linear = nn.Linear(2 * d * h, hidden_dim, bias=False).to(device)

        self.l_1 = nn.Parameter(torch.randn(size=(h, self.d)))
        self.l_2 = nn.Parameter(torch.randn(size=(h, self.d)))
        self.l_3 = nn.Parameter(torch.randn(size=(h, self.d)))
        self.l_4 = nn.Parameter(torch.randn(size=(h, self.d)))

        self.l_init = l_init
        self.g_norm = GroupNorm(2 * self.d, l_init).to(device)

    def forward(self, x, attention_mask=None):
        q = self.q_linear(x)  # batch, n , 2*d*h
        k = self.k_linear(x)  # batch, n , 2*d*h
        v = self.v_linear(x)  # batch, n , 2*d*h

        v = v.view(x.shape[0], x.shape[1], self.h, 2 *
                    self.d)  # batch, n , h, 2*d
        q = q.view(x.shape[0], x.shape[1], 2 * self.h,
                    self.d)  # batch, n , 2*h, d
        k = k.view(x.shape[0], x.shape[1],  2 * self.h,
                    self.d )  # batch, n , 2*h, d

        q = q.permute(0, 2, 1, 3)  # batch, 2*h, n, d
        k = k.permute(0, 2, 3, 1)  # batch, 2*h, d, h
        v = v.permute(0, 2, 1, 3)  # batch, h, n, 2*d

        q_k = torch.matmul(q, k)/torch.sqrt(torch.tensor(self.d))  # batch, 2*h, n, n

        causal_mask = torch.tril(torch.ones(x.shape[1], x.shape[1], device=x.device)).unsqueeze(0).unsqueeze(1)
        # => [1, 1, n, n], which will broadcast to [B, h, n, n]
        q_k = q_k.masked_fill(causal_mask == 0, float('-inf'))

        if attention_mask is not None:
            # Expand to [B, 1, 1, n]
            # So it can broadcast to [B, h, n, n] along h and the "row" dimension
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # => [B, 1, 1, n]

            # Where attn_mask=0 => set q_k to -inf
            q_k = q_k.masked_fill(attn_mask == 0, float('-inf'))

        l = (
            torch.exp(torch.sum(self.l_1*self.l_2, dim=1))
            - torch.exp(torch.sum(self.l_2*self.l_3, dim=1))
            + self.l_init  # h, 1
        )
        
        l = l.view(1, self.h, 1, 1)  # 1,h,1,1
        soft_1 = torch.softmax(q_k[:, :self.h, :, :], dim=3)  # batch, h, n, n
        soft_2 = torch.softmax(q_k[:, self.h:, :, :], dim=3)  # batch, h, n, n

        att = (
            torch.matmul(soft_1, v)
            - l * (torch.matmul(soft_2, v))
        ) # batch, h, n, 2*d
        
        att = self.g_norm.forward(att)  # batch, h, n, 2*d

        att = att.permute(0, 2, 1, 3)  # batch, n, h, 2*d
        att = att.reshape((x.shape[0], x.shape[1], 2 * self.d * self.h))  # batch, n, 2*d*h

        att = self.o_linear(att)  # batch, n, hidden_dim
        return att


class DifferentialTransformer(nn.Module):
    def __init__(self, hidden_dim, head_number, head_dim, num_layers, l_init, device = 'cpu'):
        super().__init__()
        self.num_layers = num_layers

        self.diff_attentions = nn.ModuleList([DifferentialAttention(head_number, head_dim, hidden_dim, l_init, device = device) for _ in range(num_layers)])
        self.swiglus = nn.ModuleList([SwiGLU(hidden_dim).to(device) for _ in range(num_layers)])
        self.layer_norms_att = nn.ModuleList([RMSNorm(hidden_dim).to(device) for _ in range(num_layers)])
        self.layer_norms_mlp = nn.ModuleList([RMSNorm(hidden_dim).to(device) for _ in range(num_layers)])

    def forward(self, x, attention_mask=None):

        for indx in range(self.num_layers):

            x_not_norm = x
            x = self.layer_norms_att[indx].forward(x)

            y = self.diff_attentions[indx].forward(x, attention_mask=attention_mask) + x_not_norm
            y_not_norm = y
            y = self.layer_norms_mlp[indx].forward(y)

            x = self.swiglus[indx].forward(y) + y_not_norm

        return x


class Transformer(nn.Module):
    def __init__(self, hidden_dim, head_number, head_dim, num_layers, l_init, device = 'cpu'):
        super().__init__()
        self.num_layers = num_layers

        self.attentions = nn.ModuleList([
            Attention(head_number, head_dim, hidden_dim, l_init, device = device) for _ in range(num_layers)
        ])
        self.swiglus = nn.ModuleList([SwiGLU(hidden_dim).to(device) for _ in range(num_layers)])
        self.layer_norms_att = nn.ModuleList([RMSNorm(hidden_dim).to(device) for _ in range(num_layers)])
        self.layer_norms_mlp = nn.ModuleList([RMSNorm(hidden_dim).to(device) for _ in range(num_layers)])

    def forward(self, x, attention_mask=None):

        for indx in range(self.num_layers):

            x_not_norm = x
            x = self.layer_norms_att[indx].forward(x)

            y = self.attentions[indx].forward(x, attention_mask=attention_mask) + x_not_norm
            y_not_norm = y
            y = self.layer_norms_mlp[indx].forward(y)

            x = self.swiglus[indx].forward(y) + y_not_norm

        return x

# Example usage:
if __name__ == "__main__":
    # Dummy inputs
    batch_size, seq_len, embed_dim, num_heads = 2, 128, 64, 8
    x = torch.randn(batch_size, seq_len, embed_dim, device='cpu')  # flash_attn requires CUDA
    l_init = 0.5
    layer = Transformer(embed_dim, num_heads, embed_dim // num_heads, 10, l_init)

    out = layer(x)
    print(out.shape)  # [2, 128, 64]
