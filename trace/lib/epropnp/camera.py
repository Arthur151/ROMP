"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import torch

from .common import yaw_to_rot_mat, quaternion_to_rot_mat, skew


def project_a(x3d, pose, cam_mats, z_min: float):
    if pose.size(-1) == 4:
        x3d_rot = x3d @ (yaw_to_rot_mat(pose[..., -1])).transpose(-1, -2)
    else:
        x3d_rot = x3d @ quaternion_to_rot_mat(pose[..., 3:]).transpose(-1, -2)
    x2dh_proj = (x3d_rot + pose[..., None, :3]) @ cam_mats.transpose(-1, -2)
    z = x2dh_proj[..., 2:3].clamp(min=z_min)
    x2d_proj = x2dh_proj[..., :2] / z  # (*, n, 2)
    return x2d_proj, x3d_rot, z


def project_b(x3d, pose, cam_mats, z_min: float):
    if pose.size(-1) == 4:
        x2dh_proj = x3d @ (cam_mats @ yaw_to_rot_mat(pose[..., -1])).transpose(-1, -2) \
                    + (cam_mats @ pose[..., :3, None]).squeeze(-1).unsqueeze(-2)
    else:
        x2dh_proj = x3d @ (cam_mats @ quaternion_to_rot_mat(pose[..., 3:])).transpose(-1, -2) \
                    + (cam_mats @ pose[..., :3, None]).squeeze(-1).unsqueeze(-2)
    z = x2dh_proj[..., 2:3].clamp(min=z_min)
    x2d_proj = x2dh_proj[..., :2] / z
    return x2d_proj, z


class PerspectiveCamera(object):

    def __init__(
            self,
            cam_mats=None,
            z_min=0.1,
            img_shape=None,
            allowed_border=200,
            lb=None,
            ub=None):
        """
        Args:
            cam_mats (Tensor): Shape (*, 3, 3)
            img_shape (Tensor | None): Shape (*, 2) in [h, w]
            lb (Tensor | None): Shape (*, 2), lower bound in [x, y]
            ub (Tensor | None): Shape (*, 2), upper bound in [x, y]
        """
        super(PerspectiveCamera, self).__init__()
        self.z_min = z_min
        self.allowed_border = allowed_border
        self.set_param(cam_mats, img_shape, lb, ub)

    def set_param(self, cam_mats, img_shape=None, lb=None, ub=None):
        self.cam_mats = cam_mats
        if img_shape is not None:
            self.lb = -0.5 - self.allowed_border
            self.ub = img_shape[..., [1, 0]] + (-0.5 + self.allowed_border)
        else:
            self.lb = lb
            self.ub = ub

    def project(self, x3d, pose, out_jac=False, clip_jac=True):
        """
        Args:
            x3d (Tensor): Shape (*, n, 3)
            pose (Tensor): Shape (*, 4 or 7)
            out_jac (bool | Tensor): Shape (*, n, 2, 4 or 6)

        Returns:
            Tuple[Tensor]:
                x2d_proj: Shape (*, n, 2)
                jac: Shape (*, n, 2, 4 or 6), Jacobian w.r.t. the local pose in tangent space
        """
        if out_jac is not False:
            x2d_proj, x3d_rot, zcam = project_a(x3d, pose, self.cam_mats, self.z_min)
        else:
            x2d_proj, zcam = project_b(x3d, pose, self.cam_mats, self.z_min)

        lb, ub = self.lb, self.ub
        if lb is not None and ub is not None:
            requires_grad = x2d_proj.requires_grad
            if isinstance(lb, torch.Tensor):
                lb = lb.unsqueeze(-2)
                x2d_proj = torch.max(lb, x2d_proj, out=x2d_proj if not requires_grad else None)
            else:
                x2d_proj.clamp_(min=lb)
            if isinstance(ub, torch.Tensor):
                ub = ub.unsqueeze(-2)
                x2d_proj = torch.min(x2d_proj, ub, out=x2d_proj if not requires_grad else None)
            else:
                x2d_proj.clamp_(max=ub)

        if out_jac is not False:
            if not isinstance(out_jac, torch.Tensor):
                out_jac = None
            jac = self.project_jacobian(
                x3d_rot, zcam, x2d_proj, out_jac=out_jac, dof=4 if pose.size(-1) == 4 else 6)
            if clip_jac:
                if lb is not None and ub is not None:
                    clip_mask = (zcam == self.z_min) | ((x2d_proj == lb) | (x2d_proj == ub))
                else:
                    clip_mask = zcam == self.z_min
                jac.masked_fill_(clip_mask[..., None], 0)
        else:
            jac = None

        return x2d_proj, jac

    def project_jacobian(self, x3d_rot, zcam, x2d_proj, out_jac, dof):
        if dof == 4:
            d_xzcam_d_yaw = torch.stack(
                (x3d_rot[..., 2], -x3d_rot[..., 0]), dim=-1).unsqueeze(-1)
        elif dof == 6:
            d_x3dcam_d_rot = skew(x3d_rot * 2)
        else:
            raise ValueError('dof must be 4 or 6')
        if zcam.requires_grad or x2d_proj.requires_grad:
            assert out_jac is None, 'out_jac is not supported for backward'
            d_x2d_d_x3dcam = torch.cat(
                (self.cam_mats[..., None, :2, :2] / zcam.unsqueeze(-1),
                 (self.cam_mats[..., None, :2, 2:3] - x2d_proj.unsqueeze(-1)) / zcam.unsqueeze(-1)),
                dim=-1)
            # (b, n, 2, 4 or 6)
            jac = torch.cat(
                (d_x2d_d_x3dcam,
                 d_x2d_d_x3dcam[..., ::2] @ d_xzcam_d_yaw if dof == 4
                 else d_x2d_d_x3dcam @ d_x3dcam_d_rot), dim=-1)
        else:
            if out_jac is None:
                jac = torch.empty(x3d_rot.shape[:-1] + (2, dof),
                                  device=x3d_rot.device, dtype=x3d_rot.dtype)
            else:
                jac = out_jac
            # d_x2d_d_xycam (b, n, 2, 2) = (b, 1, 2, 2) / (b, n, 1, 1)
            jac[..., :2] = self.cam_mats[..., None, :2, :2] / zcam.unsqueeze(-1)
            # d_x2d_d_zcam (b, n, 2, 1) = ((b, 1, 2, 1) - (b, n, 2, 1)) / (b, n, 1, 1)
            jac[..., 2:3] = (self.cam_mats[..., None, :2, 2:3] - x2d_proj.unsqueeze(-1)
                             ) / zcam.unsqueeze(-1)
            jac[..., 3:] = jac[..., ::2] @ d_xzcam_d_yaw if dof == 4 \
                else jac[..., :3] @ d_x3dcam_d_rot
        return jac

    @staticmethod
    def get_quaternion_transfrom_mat(quaternions):
        """
        Get the transformation matrix that maps the local rotation delta in 3D tangent
        space to the 4D space where the quaternion is embedded.

        Args:
            quaternions (torch.Tensor): (*, 4), the quaternion that determines the source
                tangent space

        Returns:
            torch.Tensor: (*, 4, 3)
        """
        w, i, j, k = torch.unbind(quaternions, -1)
        transfrom_mat = torch.stack(
            ( i,  j,  k,
             -w, -k,  j,
              k, -w, -i,
             -j,  i, -w),
            dim=-1)
        return transfrom_mat.reshape(quaternions.shape[:-1] + (4, 3))

    def reshape_(self, *batch_shape):
        self.cam_mats = self.cam_mats.reshape(*batch_shape, 3, 3)
        if isinstance(self.lb, torch.Tensor):
            self.lb = self.lb.reshape(*batch_shape, 2)
        if isinstance(self.ub, torch.Tensor):
            self.ub = self.ub.reshape(*batch_shape, 2)
        return self

    def expand_(self, *batch_shape):
        self.cam_mats = self.cam_mats.expand(*batch_shape, -1, -1)
        if isinstance(self.lb, torch.Tensor):
            self.lb = self.lb.expand(*batch_shape, -1)
        if isinstance(self.ub, torch.Tensor):
            self.ub = self.ub.expand(*batch_shape, -1)
        return self

    def repeat_(self, *batch_repeat):
        self.cam_mats = self.cam_mats.repeat(*batch_repeat, 1, 1)
        if isinstance(self.lb, torch.Tensor):
            self.lb = self.lb.repeat(*batch_repeat, 1)
        if isinstance(self.ub, torch.Tensor):
            self.ub = self.ub.repeat(*batch_repeat, 1)
        return self

    def shallow_copy(self):
        return PerspectiveCamera(
            cam_mats=self.cam_mats,
            z_min=self.z_min,
            allowed_border=self.allowed_border,
            lb=self.lb,
            ub=self.ub)
