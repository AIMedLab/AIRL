import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class FeatureExtractorCircle(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layer):
        super(FeatureExtractorCircle, self).__init__()
        self.enc = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        for _ in range(num_layer - 2):
            self.enc.append(nn.Linear(hidden_dim, hidden_dim))
            self.enc.append(nn.ReLU())
        self.enc.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feature):
        # feature = [batch * num_feature]
        return self.enc(feature)


class FeatureExtractorImage(nn.Module):
    def __init__(self, num_input_channels, hidden_dim, num_layer):
        super(FeatureExtractorImage, self).__init__()
        self.enc = nn.Sequential(self.conv_block(num_input_channels, hidden_dim))
        for _ in range(num_layer - 1):
            self.enc.append(self.conv_block(hidden_dim, hidden_dim))

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        if len(x.shape) == 5:
            n_batch, n_domain, c, h, w = x.shape
            x = x.reshape(n_batch * n_domain, c, h, w)
            check = True
        else:
            check = False
        x = self.enc(x)
        x = torch.mean(x, dim=(2, 3))
        if check:
            x = x.reshape(n_batch, n_domain, -1)
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Transformer, self).__init__()
        self.trans = nn.TransformerEncoderLayer(d_model=input_dim, nhead=1, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim),
                                    nn.LeakyReLU(0.02))
        self.output_dim = output_dim

    def subsequent_mask(self, size, num_attn, device):
        attn_shape = (size, size)
        mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        mask = torch.from_numpy(mask) == 0
        for i in range(num_attn, size):
            mask[i, :(i-num_attn+1)] = False
        return mask.to(device)

    def forward(self, feature, num_attn, device):
        # feature = [batch * seq_len * num_feature]
        n_batch, n_domain = feature.shape[:2]
        mask = self.subsequent_mask(n_domain, num_attn, device) * 1.0
        feature = self.trans(feature, mask)
        feature = feature.reshape(n_batch * n_domain, -1)
        return self.linear(feature).reshape(n_batch, n_domain, -1)


class TaskClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_label):
        super(TaskClassifier, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_label))

    def forward(self, x):
        # feature = [batch * num_feature]
        return self.classifier(x)


class EvolveClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(EvolveClassifier, self).__init__()
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        self.scale_up = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, output_dim))
        self.scale_down = nn.Sequential(
            nn.Linear(output_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim))

    def nn_construct(self, hidden, hidden_dim, output_dim):
        w1 = hidden[:(hidden_dim * hidden_dim)].view(hidden_dim, hidden_dim)
        b1 = hidden[(hidden_dim * hidden_dim):(hidden_dim * hidden_dim + hidden_dim)].view(hidden_dim)
        w2 = hidden[(hidden_dim * hidden_dim + hidden_dim):(hidden_dim * hidden_dim + hidden_dim + output_dim
                                                            * hidden_dim)].view(output_dim, hidden_dim)
        b2 = hidden[(hidden_dim * hidden_dim + hidden_dim + output_dim * hidden_dim):].view(output_dim)
        return w1, b1, w2, b2

    def forward(self, history, hidden_dim, output_dim):
        # history = [seq_len * output_dim]
        output, _ = self.rnn(self.scale_down(history))
        # output = [seq_len * hidden_dim]
        output = output[-1]
        # output = [hidden_dim]
        output = self.scale_up(output)
        # output = [output_dim]
        return output, self.nn_construct(output, hidden_dim, output_dim)


class AIRL(nn.Module):
    def __init__(self, feature_extractor, transformer, evolve_classifier, n_output, ts_coef, lr, device):
        super(AIRL, self).__init__()
        self.feature_extractor = feature_extractor
        self.evolve_classifier = evolve_classifier
        self.transformer = transformer
        self._init_cls(self.transformer.output_dim, n_output)
        self.n_output = n_output
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.ts_coef = ts_coef
        self.device = device

    def _init_cls(self, hidden_dim, output_dim):
        stdv = 1. / math.sqrt(hidden_dim)
        self.w1 = torch.nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.b1 = torch.nn.Parameter(torch.ones(hidden_dim))
        self.w2 = torch.nn.Parameter(torch.zeros(output_dim, hidden_dim))
        self.b2 = torch.nn.Parameter(torch.ones(output_dim))
        torch.nn.init.uniform_(self.w1, -stdv, stdv)
        torch.nn.init.uniform_(self.b1, -stdv, stdv)
        torch.nn.init.uniform_(self.w2, -stdv, stdv)
        torch.nn.init.uniform_(self.b2, -stdv, stdv)

    def mmd(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)
        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()
        return mean_diff + cova_diff

    def classify(self, input, w1, b1, w2, b2):
        output = F.relu(F.linear(input, w1, b1))
        return F.linear(output, w2, b2)

    def forward(self, features, y_task, n_label, mode=None):
        # features = [batch * seq_len * input_dim]
        # y_task = [batch * seq_len]
        n_batch, n_domain = features.shape[:2]
        history = [torch.cat((self.w1.flatten(), self.b1.flatten(), self.w2.flatten(), self.b2.flatten()), dim=0)]
        z = self.feature_extractor(features)
        # z = [batch * seq_len * feature_dim]
        z_attn = self.transformer(z, n_domain, self.device)
        # z_attn = [batch * seq_len * feature_dim]
        inv_loss = 0
        cls_loss = 0
        ts_loss = 0
        predict = []
        for d in range(n_domain-1):
            zi = z_attn[:, d]
            zj = z[:, (d+1)]
            yi = y_task[:, d]
            yj = y_task[:, (d+1)]
            idx_i_list = [yi == lb for lb in range(n_label)]
            idx_j_list = [yj == lb for lb in range(n_label)]

            for k in range(n_label):
                if idx_i_list[k].sum() > 1 and idx_j_list[k].sum() > 1:
                    inv_loss += self.mmd(zi[idx_i_list[k].squeeze(-1)], zj[idx_j_list[k].squeeze(-1)])
            if d == 0:
                pred = self.classify(torch.cat((zi, zj), dim=0), self.w1, self.b1, self.w2, self.b2)
            else:
                w_vector, (w1, b1, w2, b2) = self.evolve_classifier(torch.stack(history),
                                                                    self.transformer.output_dim, self.n_output)
                ts_loss += F.l1_loss(history[-1], w_vector)
                history.append(w_vector)
                pred = self.classify(torch.cat((zi, zj), dim=0), w1, b1, w2, b2)
            if n_label == 2:
                cls_loss += F.binary_cross_entropy_with_logits(pred, torch.cat((yi, yj), dim=0))
            else:
                cls_loss += nn.CrossEntropyLoss()(pred, torch.cat((yi, yj), dim=0).flatten())
            if d == 0:
                predict.append(pred[:n_batch])
            predict.append(pred[n_batch:])
        inv_loss = inv_loss / (n_domain - 1)
        cls_loss = cls_loss / (n_domain - 1)
        ts_loss = ts_loss / (n_domain - 1)
        loss = inv_loss + cls_loss + self.ts_coef * ts_loss
        if mode == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        predict = [p.unsqueeze(1) for p in predict]
        predict = torch.cat(predict, dim=1).reshape(-1, self.n_output)
        return loss, predict.squeeze(-1)

    def predict(self, features, domain_index):
        history = [torch.cat((self.w1.flatten(), self.b1.flatten(), self.w2.flatten(), self.b2.flatten()), dim=0)]
        for i in range(1, domain_index):
            w_vector, (w1, b1, w2, b2) = self.evolve_classifier(torch.stack(history),
                                                                self.transformer.output_dim, self.n_output)
            history.append(w_vector)
        z = self.feature_extractor(features)
        pred = self.classify(z, w1, b1, w2, b2)
        return pred.squeeze(-1)
