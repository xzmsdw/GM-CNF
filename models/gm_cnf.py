import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.blocks import ConditionEncoder, InvertibleLinear, DualTCN1D

class LinearCouplingLayer(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=256, mask_type='even', use_cond=True):
        super(LinearCouplingLayer, self).__init__()
        self.mask_type = mask_type
        self.use_cond = use_cond
        
        in_dim = (input_dim // 2) + (cond_dim if use_cond else 0)
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

        mask = torch.zeros(input_dim)
        if mask_type == 'even':
            mask[0::2] = 1
        else:
            mask[1::2] = 1
        self.register_buffer('mask', mask)

    def forward(self, x, h_c, reverse=False):
        # x: [B, Dim]
        # h_c: [B, Cond_Dim]
        
        mask = self.mask
        x_masked = x * mask

        x_in = x[:, mask.bool()]

        if self.use_cond:
            mlp_in = torch.cat([x_in, h_c], dim=1)
        else:
            mlp_in = x_in
        
        out = self.net(mlp_in)
        s, t = torch.chunk(out, 2, dim=1) # [B, Dim//2]
        
        s_full = torch.zeros_like(x)
        t_full = torch.zeros_like(x)
        s_full[:, ~mask.bool()] = s
        t_full[:, ~mask.bool()] = t

        s_full = torch.tanh(s_full) * 0.5
        
        if not reverse:
            y = x * torch.exp(s_full * (1 - mask)) + t_full * (1 - mask)
            log_det = torch.sum(s_full * (1 - mask), dim=1)
            return y, log_det
        else:
            y = (x - t_full * (1 - mask)) * torch.exp(-s_full * (1 - mask))
            return y, None

class GaussianMixturePriorVector(nn.Module):
    def __init__(self, num_classes, z_dim, learnable_var=False):
        super(GaussianMixturePriorVector, self).__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.learnable_var = learnable_var
        
        self.means = nn.Parameter(torch.randn(num_classes, z_dim) * 0.5)
        if learnable_var:
            self.log_vars = nn.Parameter(torch.zeros(num_classes, z_dim))
        else:
            self.register_buffer('log_vars', torch.zeros(1))
        self.w_logits = nn.Parameter(torch.zeros(num_classes))

    def get_log_prob_components(self, z):
        z_exp = z.unsqueeze(1)
        means = self.means.unsqueeze(0)
        if self.learnable_var:
            log_vars = self.log_vars.unsqueeze(0)
        else:
            log_vars = self.log_vars
        log_prob_element = -0.5 * (log_vars + (z_exp - means)**2 / torch.exp(log_vars))
        
        log_prob_k = torch.sum(log_prob_element, dim=2)
        
        log_weights = F.log_softmax(self.w_logits, dim=0)
        return log_prob_k + log_weights.unsqueeze(0)
    
    def log_prob(self, z, label=None):
        log_comps = self.get_log_prob_components(z)
        if label is not None:
            return log_comps.gather(1, label.unsqueeze(1)).squeeze(1)
        else:
            return torch.logsumexp(log_comps, dim=1)

class GM_CNF(nn.Module):
    def __init__(self, args):
        super(GM_CNF, self).__init__()
        
        self.feature_dim = 128
        self.args = args
        use_sn = not getattr(args, 'ablation_no_sn', False)
        self.backbone = DualTCN1D(
            in_channels=args.c_in_x, 
            output_dim=self.feature_dim,
            patch_size=args.patch_size,
            use_sn=use_sn
        )

        self.cond_encoder = nn.Sequential(
            nn.Linear(args.c_in_c, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.n_blocks = 4
        self.flows = nn.ModuleList()
        use_cond = not getattr(args, 'ablation_no_cond', False)
        for i in range(self.n_blocks):
            mask_type = 'even' if i % 2 == 0 else 'odd'

            self.flows.append(InvertibleLinear(self.feature_dim))

            self.flows.append(
                LinearCouplingLayer(self.feature_dim, 64, hidden_dim=256, mask_type=mask_type, use_cond=use_cond)
            )
            self.flows.append(nn.BatchNorm1d(self.feature_dim))

        learn_var = getattr(args, 'ablation_learnable_var', False)
        self.prior = GaussianMixturePriorVector(args.num_classes, self.feature_dim, learnable_var=learn_var)

    def forward_features(self, x, c):
        h = self.backbone(x)

        if c.dim() == 3: 
            c = c.mean(dim=2)
        h_c = self.cond_encoder(c)
        
        return h, h_c

    def forward_flow(self, h, h_c):
        z = h
        log_det_sum = 0
        for flow in self.flows:
            if isinstance(flow, nn.BatchNorm1d):
                z = flow(z)
            else:
                z, log_det = flow(z, h_c, reverse=False)
                log_det_sum = log_det_sum + log_det
                
        return z, log_det_sum

    def compute_loss(self, x, c, label=None):
        h, h_c = self.forward_features(x, c)
        z, log_det = self.forward_flow(h, h_c)
        
        log_p_z = self.prior.log_prob(z, label)
        
        log_likelihood = log_p_z + log_det

        loss = -torch.mean(log_likelihood)

        loss = loss / self.feature_dim

        return loss
    def predict_dist(self, x, c):
        h, h_c = self.forward_features(x, c)
        z, log_det = self.forward_flow(h, h_c)

        if getattr(self.args, 'ablation_learnable_var', False):
            log_probs = self.prior.get_log_prob_components(z) # [B, K]
            preds = torch.argmax(log_probs, dim=1)
            min_dist = -torch.max(log_probs, dim=1)[0]
        else:
            means = self.prior.means
            dists = torch.sum((z.unsqueeze(1) - means.unsqueeze(0)) ** 2, dim=2)
            preds = torch.argmin(dists, dim=1)
            min_dist, _ = torch.min(dists, dim=1)
        
        return preds, min_dist, z