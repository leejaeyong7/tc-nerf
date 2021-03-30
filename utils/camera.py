import torch
import torch.nn.functional as NF
from .transforms import *
class Camera:
    def __init__(self, K, E, shape):
        '''
        Creates commonly used variables from intrinsics, extrinsics

        Args:
            K: Bx3x3 intrinsic matrix
            E: Bx4x4 extrinsic matrix
            shape: (H, W) image size
        '''
        self.H, self.W = shape
        # convert them to BxHxWx3x3 format, since we want to batch the
        # ray based ops
        self.K = K.view(1, 1, 1, 3, 3)
        self.E = E.view(1, 1, 1, 4, 4)
        self.R = self.E[..., :3, :3]
        self.t = self.E[..., :3, 3:]
        self.Ki = self.inverse_intrinsic(self.K)
        self.Rt = self.R.transpose(-1, -2)
        self.C = -self.Rt @ self.t
        self.grid_K = self.generate_grid_intrinsics()

    def inverse_intrinsic(self, K):
        Ki = K.clone()
        fx = K[..., 0:1, 0:1]
        fy = K[..., 1:2, 1:2]
        Ki[..., 0:1, :] /= fx
        Ki[..., 1:2, :] /= fy
        Ki[..., 0:1, 0:1] /= fx
        Ki[..., 1:2, 1:2] /= fy
        Ki[..., :2, -1] *= -1
        return Ki


    def generate_grid_intrinsics(self):
        dev = self.K.device
        grid_K = torch.eye(2, device=dev)
        grid_K[0, 0] = 2 / float(self.W)
        grid_K[1, 1] = 2 / float(self.H)
        return grid_K.view(1, 1, 1, 2, 2)

    def resize(self, new_shape):
        NH, NW = new_shape
        self.K = self.K.clone()
        self.K[:, :, :, 0] *= NW / float(self.W)
        self.K[:, :, :, 1] *= NH / float(self.H)
        self.Ki = self.inverse_intrinsic(self.K)
        self.H, self.W = new_shape
        self.grid_K = self.generate_grid_intrinsics()

    def resize_by_scale(self, scale: float):
        # clone buffer to ensuer inplace operation
        self.K = self.K.clone()
        self.K[:, :, :, 0] *= scale
        self.K[:, :, :, 1] *= scale
        self.Ki = self.inverse_intrinsic(self.K)
        self.H = int(scale * self.H)
        self.W = int(scale * self.W)
        self.grid_K = self.generate_grid_intrinsics()

    ###############################
    # inverse projection related
    # all projection related takes
    # NxHxWx... format shapes and outputs NxHxWx... format shapes
    def pixel_points(self, offset=0.5):
        '''
        Given width and height, creates a mesh grid, and returns homogeneous 
        coordinates
        of image in a 3 x W*H Tensor

        Arguments:
            width {Number} -- Number representing width of pixel grid image
            height {Number} -- Number representing height of pixel grid image

        Returns:
            torch.Tensor -- 1x2xHxW, oriented in x, y order
        '''
        dev = self.K.device
        W = self.W
        H = self.H
        O = offset
        x_coords = torch.linspace(O, W - 1 + O, W, device=dev)
        y_coords = torch.linspace(O, H - 1 + O, H, device=dev)

        # HxW grids
        y_grid_coords, x_grid_coords = torch.meshgrid([y_coords, x_coords])

        # HxWx2 grids => 1xHxWx2 grids
        return torch.stack([ x_grid_coords, y_grid_coords ], 2).unsqueeze(0)

    def camera_rays(self):
        x = from_vector(to_homogeneous(self.pixel_points()))
        Ki = self.Ki
        return to_vector(Ki @ x)

    def back_project(self, depth_maps):
        '''
        Given depth map, back project its depth to obtain world coordinates

        Args:
            depth_map: NxHxWx1 depths
        Returns:
            NxHxWx3 points in world coordinates
        '''
        Rt = self.Rt
        t = self.t

        # 1xHxWx3 * NxHxWx1
        r = self.camera_rays()
        p = from_vector(r * depth_maps)

        return to_vector(Rt @ (p - t))

    def to_world_normals(self, normal_maps):
        '''
        Given depth map, back project its depth to obtain world coordinates

        Args:
            depth_map: NxHxWx1 depths
        Returns:
            NxHxWx3 points in world coordinates
        '''
        Rt = self.Rt
        t = self.t
        return to_vector(Rt @ from_vector(normal_maps))

    def back_project_patches(self, patches):
        '''
        Given depth map, back project its depth to obtain world coordinates

        Args:
            patches: NxHxWx3xA patches
        Returns:
            NxHxWx3xA points in world coordinates
        '''
        Rt = self.Rt
        t = self.t
        Ki = self.Ki
        # NxHxWx3xA => 3xA => 3xA => 3xA
        return Rt @ ((Ki @ patches) - t)

    def project(self, world_p):
        p = from_vector(world_p)
        K = self.K
        R = self.R
        t = self.t

        return to_vector(from_homogeneous(K @ (R @ p + t)))


    def get_homographies(self, plane_maps):
        '''
        Given per-pixel planes in reference camera, obtain homographies for each pixel
        '''
        K = self.K
        Ki = self.Ki
        R = self.R
        Rt = self.Rt
        t = self.t

        normal = from_vector(plane_maps[:, :, :, :3])
        dists = from_vector(plane_maps[:, :, :, 3:])

        rel_R = R @ Rt[:1]
        rel_t = (-rel_R @ t[:1]) + t

        # obtain distance from plane
        pnf_planes = -(normal / -dists).transpose(3, 4)
        cam_h = rel_R - (rel_t @ pnf_planes)

        # NxHxWx3x3
        return K @ cam_h @ Ki[:1]

    def normalize(self, points):
        projected = from_homogeneous(points)
        return self.grid_K @ projected - 1

    def init(self, N, ranges):
        dev = self.K.device
        min_d, max_d = ranges
        H, W = self.H, self.W
        Rt = self.Rt

        depths = torch.linspace(1 / max_d, 1 / min_d, N + 1, device=dev).view(1, 1, 1, -1)
        min_ds = depths[..., :-1]
        max_ds = depths[..., 1:]
        sampled_depths = 1/ (torch.rand((1, H, W, N), dtype=torch.float, device=dev) * (max_ds - min_ds) + min_ds)
        nhwd = sampled_depths.permute(3, 1, 2, 0)
        rs = self.camera_rays()

        # NxHxWx3 world points, 1xHxWx3 directions
        dirs = to_vector(Rt @ from_vector(self.camera_rays()))
        return nhwd, dirs 




    def sample(self, N, ranges):
        '''
        Given min/max ranges of depths, return sampled depths based on uniform distrb.
        '''
        dev = self.K.device
        min_d, max_d = ranges
        H, W = self.H, self.W
        Rt = self.Rt
        depths = torch.linspace(1 / max_d, 1 / min_d, N + 1, device=dev).view(1, 1, 1, -1)
        min_ds = depths[..., :-1]
        max_ds = depths[..., 1:]
        # 1xHxWxN
        sampled_depths = 1/ (torch.rand((1, H, W, N), dtype=torch.float, device=dev) * (max_ds - min_ds) + min_ds)
        nhwd = sampled_depths.permute(3, 1, 2, 0)
        rs = self.camera_rays()

        # NxHxWx3 world points, 1xHxWx3 directions
        pts = self.back_project(nhwd)
        dirs = to_vector(Rt @ from_vector(self.camera_rays()))

        # dirs = rs
        # pts = rs * nhwd

        return pts, nhwd, dirs 

    def render(self, rgb, sigma, depths, add_noise=False):
        # sigma = NxHxWx1
        # depths = NxHxWx1
        # rgb = NxHxWx3
        noise_std = 1
        if(add_noise):
            noise = torch.randn_like(sigma) * noise_std
        else:
            noise = 0.0
        sig = NF.relu(sigma + noise) + 1e-10
        dev = self.K.device
        one_e_10 = torch.ones_like(depths[:1]) * 1e10
        dists = torch.cat((depths[1:] - depths[:-1], one_e_10), 0)
        # alpha = 1 - (-sig * dists)
        # Ti = alpha.clone().cumsum(0).roll(1, 0).exp()
        # Ti[0] = 1
        # w  = Ti * (1 - alpha).exp()

        alpha = 1 - (-sig * dists).exp()
        Ti = (1 - alpha).cumprod(0).roll(1, 0)
        Ti[0] = 1
        w = alpha * Ti

        rgb_map = (w * rgb).sum(0)
        depth_map = (w * depths).sum(0)
        return rgb_map, depth_map, w
