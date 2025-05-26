import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout=0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgnn1 = HGNN_conv(in_dim, hidden_list[0])

    def forward(self, x, G):
        x_embed = self.hgnn1(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed_1


class VGAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(VGAE, self).__init__()
        self.gc1 = HGNN_conv(in_channels, hidden_channels)
        self.gc_mu = HGNN_conv(hidden_channels, latent_dim)
        self.gc_logvar = HGNN_conv(hidden_channels, latent_dim)

    def encode(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        mu = self.gc_mu(x, adj)
        logvar = self.gc_logvar(x, adj)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        recon_adj = self.decode(z)
        return z, recon_adj, mu, logvar


def vgae_loss_function(recon_adj, adj, mu, logvar):
    BCE = F.binary_cross_entropy(recon_adj.view(-1), adj.view(-1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class GraphGenerator(nn.Module):
    def __init__(self, feature_dim):
        super(GraphGenerator, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, node_features):
        latent = F.relu(self.fc(node_features))
        fake_graph = torch.sigmoid(torch.mm(latent, latent.t()))
        return fake_graph, latent


class GraphDiscriminator(nn.Module):
    def __init__(self, n_nodes):
        super(GraphDiscriminator, self).__init__()
        self.fc1 = nn.Linear(n_nodes, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, graph_matrix):
        x = F.relu(self.fc1(graph_matrix))
        x = torch.sigmoid(self.fc2(x))
        return x


class GraphGAN(nn.Module):
    def __init__(self, n_nodes, feature_dim):
        super(GraphGAN, self).__init__()
        self.generator = GraphGenerator(feature_dim)
        self.discriminator = GraphDiscriminator(n_nodes)

    def forward(self, node_features):
        fake_graph, latent = self.generator(node_features)
        real_graph = torch.sigmoid(torch.mm(node_features, node_features.t()))
        real_disc = self.discriminator(real_graph)
        fake_disc = self.discriminator(fake_graph)
        return fake_graph, real_graph, real_disc, fake_disc, latent


class VAGAR(nn.Module):
    def __init__(self, mi_num, ci_num, hidd_list, hyperpm):
        super(VAGAR, self).__init__()
        in_dim = mi_num + ci_num
        self.HGCN_mi = HGCN(in_dim, hidd_list)
        self.HGCN_ci = HGCN(in_dim, hidd_list)
        latent_dim = hyperpm.n_head * hyperpm.n_hidden * hyperpm.nmodal
        self.VGAE_mi = VGAE(hidd_list[-1], hidd_list[-1] // 2, latent_dim)
        self.VGAE_ci = VGAE(hidd_list[-1], hidd_list[-1] // 2, latent_dim)
        self.linear_x_1 = nn.Linear(latent_dim, 256)
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, 64)
        self.linear_y_1 = nn.Linear(latent_dim, 256)
        self.linear_y_2 = nn.Linear(256, 128)
        self.linear_y_3 = nn.Linear(128, 64)

        self.classifier_head = nn.Linear(128, 1)


        self.graph_gan = GraphGAN(mi_num + ci_num, 64)

    def forward(self, concat_mi_tensor, concat_ci_tensor, G_mi, G_ci):

        mi_feature = self.HGCN_mi(concat_mi_tensor, G_mi)

        ci_feature = self.HGCN_ci(concat_ci_tensor, G_ci)

        z_mi, recon_mi, mu_mi, logvar_mi = self.VGAE_mi(mi_feature, G_mi)
        z_ci, recon_ci, mu_ci, logvar_ci = self.VGAE_ci(ci_feature, G_ci)

        x = torch.relu(self.linear_x_1(z_mi))
        x = torch.relu(self.linear_x_2(x))
        x = torch.relu(self.linear_x_3(x))
        y = torch.relu(self.linear_y_1(z_ci))
        y = torch.relu(self.linear_y_2(y))
        y = torch.relu(self.linear_y_3(y))

        joint_features = torch.cat([x, y], dim=0)
        fake_graph, real_graph, real_disc, fake_disc, latent = self.graph_gan(joint_features)

        return x, y, fake_graph, real_graph, real_disc, fake_disc, recon_mi, mu_mi, logvar_mi, recon_ci, mu_ci, logvar_ci


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[64, 32]):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        return out
