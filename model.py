import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RGB_Module(nn.Module):
    def __init__(self):
        super(RGB_Module, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(SelfAttention, self).__init__()
        self.w = nn.Linear(dim_in, dim_hidden)
        self.linear_q = nn.Linear(dim_hidden, dim_hidden)
        self.linear_k = nn.Linear(dim_hidden, dim_hidden)
        self.linear_v = nn.Linear(dim_hidden, dim_out)

        self._norm_fact = 1 / math.sqrt(dim_hidden)

    def forward(self, x):
        x = self.w(x)
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        att = torch.bmm(dist, v)
        return att


class Time_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_chn = 128
        self.input_stack = nn.Sequential(nn.Conv1d(118, self.in_chn, kernel_size=32, stride=3, padding=0),
                                         nn.BatchNorm1d(self.in_chn),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool1d(4, padding=0),
                                         nn.Dropout(),
                                         nn.Conv1d(self.in_chn, self.in_chn*2, kernel_size=8, stride=1, padding=0),
                                         nn.BatchNorm1d(self.in_chn*2),
                                         nn.ReLU(inplace=True),
                                         nn.Conv1d(self.in_chn*2, self.in_chn*2, kernel_size=4, stride=1, padding=0),
                                         nn.BatchNorm1d(self.in_chn*2),
                                         nn.ReLU(inplace=True),
                                         nn.Conv1d(self.in_chn*2, self.in_chn*2, kernel_size=2, stride=1, padding=0),
                                         nn.BatchNorm1d(self.in_chn*2),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool1d(4, padding=0)
                                         )

        self.attention = SelfAttention(113, 128, 100)
        self.layer_norm = nn.LayerNorm(100, eps=0.00001)
        # Define Transformer encoder layer
        self.embedding = nn.Linear(self.in_chn*2, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Sequential(
            nn.Linear(256 * 100, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.mcdcnn =MultiScaleDilatedConv()

    def forward(self, x):

        x = self.mcdcnn(x)

        # x = self.input_stack(x)

        x = self.attention(x)

        x = x.transpose(1, 2)

        x = self.embedding(x)
        x = self.transformer_encoder(x)

        x = x.view(x.shape[0], -1)
        output = self.cls(x)

        return output


class MultiScaleDilatedConv(nn.Module):
    def __init__(self):
        super(MultiScaleDilatedConv, self).__init__()

        self.chn = 128

        # architecture
        self.dropout = nn.Dropout(p=0.5)
        # Convolutional layers with different dilation rates
        self.path1 = nn.Sequential(nn.Conv1d(118, self.chn, kernel_size=50, stride=5, padding=0, dilation=1),
                                   nn.BatchNorm1d(self.chn),
                                   nn.MaxPool1d(8, padding=0),
                                   nn.Dropout(),
                                   nn.Conv1d(self.chn, self.chn, kernel_size=8, stride=1, padding=0, dilation=1),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(self.chn, self.chn, kernel_size=4, stride=1, padding=0,
                                             dilation=1),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   )
        self.path2 = nn.Sequential(nn.Conv1d(118, self.chn, kernel_size=400, stride=16, padding=0, dilation=1),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool1d(4, padding=0),
                                   nn.Dropout(),
                                   nn.Conv1d(self.chn, self.chn, kernel_size=3, stride=1, padding=0, dilation=1),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(self.chn, self.chn, kernel_size=3, stride=1, padding=0,
                                             dilation=1),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   )
        self.path3 = nn.Sequential(nn.Conv1d(118, self.chn, kernel_size=50, stride=6, padding=0, dilation=3),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool1d(8, padding=0),
                                   nn.Dropout(),
                                   nn.Conv1d(self.chn, self.chn, kernel_size=8, stride=1, padding=0, dilation=3),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(self.chn, self.chn, kernel_size=4, stride=1, padding=0,
                                             dilation=3),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   )

        self.compress = nn.Conv1d(self.chn * 3, 128, kernel_size=1, stride=1, padding=0)
        self.smooth = nn.Conv1d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv_c5 = nn.Conv1d(128, 256, kernel_size=1, stride=1, padding=0)
        # self.compress = nn.Conv1d(self.chn * 4, 128, 1, 1, 0)
        # self.smooth = nn.Conv1d(128, 128, 3, 1, 1)
        # self.conv_c5 = nn.Conv1d(128, 128, 1, 1, 0)
        # self.fc = nn.Linear(128 * 28, 128)

    def sequence_length(self, n_channels=1, height=1, width=3000):  # MUST be changed
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        x2 = F.interpolate(x2, x1.size(2))  # Interpolation, upsampling
        x3 = F.interpolate(x3, x1.size(2))
        out = self.smooth(self.compress(torch.cat([x1, x2, x3], dim=1)))
        out = self.conv_c5(out)

        return out


def normalize_A(A, symmetry=False):
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)   # Sum up the first dimension of A
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_gcn_adj(A, K,device):
    support = []
    for i in range(K):
        if i == 0:
            # support.append(torch.eye(A.shape[1]).cuda())
            temp = torch.eye(A.shape[1])
            temp = temp.to(device)
            support.append(temp)
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support


class GraphConvolution(nn.Module):

    def __init__(self, num_in, num_out, bias=False):

        super(GraphConvolution, self).__init__()

        self.num_in = num_in
        self.num_out = num_out

        self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            # self.bias = nn.Parameter(torch.FloatTensor(num_out).cuda())
            self.bias = nn.Parameter(torch.FloatTensor(num_out))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class GCN(nn.Module):
    def __init__(self, xdim, K, num_out):
        super(GCN, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def forward(self, x, L):
        device = x.device
        adj = generate_gcn_adj(L, self.K, device)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class ChnGNN_Module(nn.Module):
    def __init__(self, xdim, k_adj, num_out):
        super(ChnGNN_Module, self).__init__()
        self.K = k_adj
        self.in_chn = 128
        self.input_stack = nn.Sequential(nn.Conv1d(118, self.in_chn, kernel_size=32, stride=3, padding=0),
                                         nn.BatchNorm1d(self.in_chn),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool1d(4, padding=0),
                                         nn.Dropout(),
                                         nn.Conv1d(self.in_chn, self.in_chn*2, kernel_size=8, stride=1, padding=0),
                                         nn.BatchNorm1d(self.in_chn*2),
                                         nn.ReLU(inplace=True),
                                         nn.Conv1d(self.in_chn*2, self.in_chn*2, kernel_size=4, stride=1, padding=0),
                                         nn.BatchNorm1d(self.in_chn*2),
                                         nn.ReLU(inplace=True),
                                         nn.Conv1d(self.in_chn*2, self.in_chn*2, kernel_size=2, stride=1, padding=0),
                                         nn.BatchNorm1d(self.in_chn*2),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool1d(4, padding=0)
                                         )
        self.layer1 = GCN(xdim, k_adj, num_out)
        self.BN1 = nn.BatchNorm1d(xdim[2])  # Standardize the second dimension
        self.fc1 = nn.Linear(xdim[1] * num_out, num_out)
        self.fc3 = nn.Linear(32, 8)
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]))
        nn.init.xavier_normal_(self.A)

    def forward(self, x):
        x = self.input_stack(x)
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        return result


class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a weight score
        )

    def forward(self, feat1, feat2, feat3):
        # Attention score
        score1 = self.attention_fc(feat1)  # [B, 1]
        score2 = self.attention_fc(feat2)
        score3 = self.attention_fc(feat3)

        # Splicing&Normalization (softmax)
        scores = torch.cat([score1, score2, score3], dim=1)  # [B, 3]
        weights = F.softmax(scores, dim=1)  # [B, 3]

        # Expand the weight dimension for modal multiplication
        w1 = weights[:, 0].unsqueeze(1)  # [B, 1]
        w2 = weights[:, 1].unsqueeze(1)
        w3 = weights[:, 2].unsqueeze(1)

        # Weighted fusion
        fused = w1 * feat1 + w2 * feat2 + w3 * feat3  # [B, 128]
        return fused, weights


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.rgb_module = RGB_Module()
        self.time_module = Time_Module()
        self.chnGNN_module = ChnGNN_Module(xdim=[16, 256, 100], k_adj=40, num_out=128)
        self.time_encoder = nn.Linear(5000, 128)

        self.fusion_module = AttentionFusion(128)

        self.cls = nn.Linear(128, 1)

    def forward(self, time_feat, image_feat):
        time_out = self.time_module(time_feat)
        rgb_out = self.rgb_module(image_feat)

        # time_emb = self.time_encoder(time_feat)
        gnn_out = self.chnGNN_module(time_feat)

        # # Method 1_AttentionFusion
        # fused_feature, attn_weights = self.fusion_module(rgb_out, time_out, gnn_out)

        # # Method 2_multiplication fusion
        fused_feature = torch.mul(time_out, gnn_out)
        fused_feature = torch.mul(fused_feature, rgb_out)

        # # Only Time_Module acc
        # fused_feature = time_out

        # # Only ChnGNN_Module acc
        # fused_feature = gnn_out

        # # Only RGN_Module acc
        # fused_feature = rgb_out

        out = self.cls(fused_feature)

        return out