"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import torch


def huber_kernel(s_sqrt, delta):
    half_rho = torch.where(s_sqrt <= delta,
                           0.5 * torch.square(s_sqrt),
                           delta * s_sqrt - 0.5 * torch.square(delta))
    return half_rho


def huber_d_kernel(s_sqrt, delta, eps: float = 1e-10):
    if s_sqrt.requires_grad or delta.requires_grad:
        rho_d_sqrt = (delta.clamp(min=eps).sqrt() * s_sqrt.clamp(min=eps).rsqrt()).clamp(max=1.0)
    else:
        rho_d_sqrt = (delta / s_sqrt.clamp_(min=eps)).clamp_(max=1.0).sqrt_()
    return rho_d_sqrt


class HuberPnPCost(object):

    def __init__(self, delta=1.0, eps=1e-10):
        super(HuberPnPCost, self).__init__()
        self.eps = eps
        self.delta = delta

    def set_param(self, *args, **kwargs):
        pass

    def compute(self, x2d_proj, x2d, w2d, jac_cam=None,
                out_residual=False, out_cost=False, out_jacobian=False):
        """
        Args:
            x2d_proj: Shape (*, n, 2)
            x2d: Shape (*, n, 2)
            w2d: Shape (*, n, 2)
            jac_cam: Shape (*, n, 2, 4 or 6), Jacobian of x2d_proj w.r.t. pose
            out_residual (Tensor | bool): Shape (*, n*2) or equivalent shape
            out_cost (Tensor | bool): Shape (*, )
            out_jacobian (Tensor | bool): Shape (*, n*2, 4 or 6) or equivalent shape
        """
        bs = x2d_proj.shape[:-2]
        pn = x2d_proj.size(-2)
        delta = self.delta
        if not isinstance(delta, torch.Tensor):
            delta = x2d.new_tensor(delta)
        delta = delta[..., None]

        residual = (x2d_proj - x2d) * w2d
        s_sqrt = residual.norm(dim=-1)

        if out_cost is not False:
            half_rho = huber_kernel(s_sqrt, delta)
            if not isinstance(out_cost, torch.Tensor):
                out_cost = None
            cost = torch.sum(half_rho, dim=-1, out=out_cost)
        else:
            cost = None

        # robust rescaling
        if out_residual is not False or out_jacobian is not False:
            rho_d_sqrt = huber_d_kernel(s_sqrt, delta, eps=self.eps)
            if out_residual is not False:
                if isinstance(out_residual, torch.Tensor):
                    out_residual = out_residual.view(*bs, pn, 2)
                else:
                    out_residual = None
                residual = torch.mul(
                    residual, rho_d_sqrt[..., None],
                    out=out_residual).view(*bs, pn * 2)
            if out_jacobian is not False:
                assert jac_cam is not None
                dof = jac_cam.size(-1)
                if isinstance(out_jacobian, torch.Tensor):
                    out_jacobian = out_jacobian.view(*bs, pn, 2, dof)
                else:
                    out_jacobian = None
                # rescaled jacobian
                jacobian = torch.mul(
                    jac_cam, (w2d * rho_d_sqrt[..., None])[..., None],
                    out=out_jacobian).view(*bs, pn * 2, dof)
        if out_residual is False:
            residual = None
        if out_jacobian is False:
            jacobian = None
        return residual, cost, jacobian

    def reshape_(self, *batch_shape):
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.reshape(*batch_shape)
        return self

    def expand_(self, *batch_shape):
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.expand(*batch_shape)
        return self

    def repeat_(self, *batch_repeat):
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.repeat(*batch_repeat)
        return self

    def shallow_copy(self):
        return HuberPnPCost(
            delta=self.delta,
            eps=self.eps)


class AdaptiveHuberPnPCost(HuberPnPCost):

    def __init__(self,
                 delta=None,
                 relative_delta=0.5,
                 eps=1e-10):
        super(HuberPnPCost, self).__init__()
        self.delta = delta
        self.relative_delta = relative_delta
        self.eps = eps

    def set_param(self, x2d, w2d):
        # compute dynamic delta
        x2d_std = torch.var(x2d, dim=-2).sum(dim=-1).sqrt()  # (num_obj, )
        self.delta = w2d.mean(dim=(-2, -1)) * x2d_std * self.relative_delta  # (num_obj, )

    def shallow_copy(self):
        return AdaptiveHuberPnPCost(
            delta=self.delta,
            relative_delta=self.relative_delta,
            eps=self.eps)
