import torch
import torch.nn as nn
import torch.nn.functional as NF
class NeRF(nn.Module):
    def __init__(self, hparams):
        # D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.input_ch = hparams.nerf_point_encode * 2 * 3 + 3
        self.input_ch_views = hparams.nerf_dir_encode * 2 * 3 + 3
        self.output_ch = 4

        self.hparams = hparams
        self.D = hparams.nerf_network_depth
        self.C = hparams.nerf_channels
        self.skips = hparams.nerf_skips
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.C)] + 
            [
                nn.Linear(self.C, self.C) 
                if i not in self.skips 
                else nn.Linear(self.C + self.input_ch, self.C) 
                for i in range(1, self.D) 
            ]
        )
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linear = nn.Sequential(
            # nn.Linear(self.input_ch_views + self.C, self.C//2),
            nn.Linear(self.input_ch_views + self.C, self.C//2),
            nn.ReLU()
        )

        self.feature_linear = nn.Linear(self.C, self.C)
        self.alpha_linear = nn.Linear(self.C, 1)
        self.rgb_linear = nn.Sequential(
            nn.Linear(self.C//2, 3),
            nn.Sigmoid()
        )

    def forward_points(self, points):
        p = points
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                p = torch.cat([points, p], -1)
            p = l(p)
            p = NF.relu(p)

        return self.alpha_linear(p)

    def forward(self, points, views):
        """
        points: BxPC
        views: BxDC

        """
        p = points
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                p = torch.cat([points, p], -1)
            p = l(p)
            p = NF.relu(p)

        alpha = self.alpha_linear(p)
        feature = self.feature_linear(p)
        p = torch.cat([feature, views], -1)
        p = self.views_linear(p)
        rgb = self.rgb_linear(p)
        return rgb, alpha
