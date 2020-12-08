import torch
import torch.nn as nn
import torch.nn.functional as NF
class NeRF(nn.Module):
    def __init__(self, hparams):
        # D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.input_ch = hparams.nerf_point_encode * 2 * 3
        self.input_ch_views = hparams.nerf_dir_encode * 2 * 3
        self.output_ch = 4

        self.hparams = hparams
        self.D = hparams.nerf_network_depth
        self.C = hparams.nerf_channels
        self.skips = hparams.nerf_skips
        self.use_viewdirs = hparams.nerf_view_dirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.C)] + 
            [
                nn.Linear(self.C, self.C) 
                if i not in self.skips 
                else nn.Linear(self.C + self.input_ch, self.C) 
                for i in range(self.D - 1) 
            ]
        )
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + self.C, self.C//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if self.use_viewdirs:
            self.feature_linear = nn.Linear(self.C, self.C)
            self.alpha_linear = nn.Linear(self.C, 1)
            self.rgb_linear = nn.Linear(self.C//2, 3)
        else:
            self.output_linear = nn.Linear(self.C, output_ch)

    def forward(self, points, views):
        p = points
        for i, l in enumerate(self.pts_linears):
            p = self.pts_linears[i](p)
            p = NF.relu(p)
            if i in self.skips:
                p = torch.cat([points, p], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(p)
            feature = self.feature_linear(p)
            p = torch.cat([feature, views], -1)
        
            for i, l in enumerate(self.views_linears):
                p = self.views_linears[i](p)
                p = NF.relu(p)

            rgb = self.rgb_linear(p)
            return rgb, alpha
        else:
            return self.output_linear(h)
