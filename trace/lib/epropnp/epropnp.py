"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import math
import torch

from abc import ABCMeta, abstractmethod
from functools import partial
from pyro.distributions import MultivariateStudentT

from .common import evaluate_pnp, pnp_normalize, pnp_denormalize
from .distributions import VonMisesUniformMix, AngularCentralGaussian


def cholesky_wrapper(mat, default_diag=None, force_cpu=True):
    device = mat.device
    if force_cpu:
        mat = mat.cpu()
    try:
        tril = torch.cholesky(mat, upper=False)
    except RuntimeError:
        n_dims = mat.size(-1)
        tril = []
        default_tril_single = torch.diag(mat.new_tensor(default_diag)) if default_diag is not None \
            else torch.eye(n_dims, dtype=mat.dtype, device=mat.device)
        for cov in mat.reshape(-1, n_dims, n_dims):
            try:
                tril.append(torch.cholesky(cov, upper=False))
            except RuntimeError:
                tril.append(default_tril_single)
        tril = torch.stack(tril, dim=0).reshape(mat.shape)
    return tril.to(device)


class EProPnPBase(torch.nn.Module, metaclass=ABCMeta):
    """
    End-to-End Probabilistic Perspective-n-Points.

    Args:
        mc_samples (int): Number of total Monte Carlo samples
        num_iter (int): Number of AMIS iterations
        normalize (bool)
        eps (float)
        solver (dict): PnP solver
    """
    def __init__(
            self,
            mc_samples=512,
            num_iter=4,
            normalize=False,
            eps=1e-5,
            solver=None):
        super(EProPnPBase, self).__init__()
        assert num_iter > 0
        assert mc_samples % num_iter == 0
        self.mc_samples = mc_samples
        self.num_iter = num_iter
        self.iter_samples = self.mc_samples // self.num_iter
        self.eps = eps
        self.normalize = normalize
        self.solver = solver

    @abstractmethod
    def allocate_buffer(self, *args, **kwargs):
        pass

    @abstractmethod
    def initial_fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def gen_new_distr(self, *args, **kwargs):
        pass

    @abstractmethod
    def gen_old_distr(self, *args, **kwargs):
        pass

    @abstractmethod
    def estimate_params(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        return self.solver(*args, **kwargs)

    def monte_carlo_forward(self, x3d, x2d, w2d, camera, cost_fun,
                            pose_init=None, force_init_solve=True, **kwargs):
        """
        Monte Carlo PnP forward. Returns weighted pose samples drawn from the probability
        distribution of pose defined by the correspondences {x_{3D}, x_{2D}, w_{2D}}.

        Args:
            x3d (Tensor): Shape (num_obj, num_points, 3)
            x2d (Tensor): Shape (num_obj, num_points, 2)
            w2d (Tensor): Shape (num_obj, num_points, 2)
            camera: Camera object of batch size (num_obj, )
            cost_fun: PnPCost object of batch size (num_obj, )
            pose_init (Tensor | None): Shape (num_obj, 4 or 7), optional. The target pose
                (y_{gt}) can be passed for training with Monte Carlo pose loss
            force_init_solve (bool): Whether to force using the initialization solver when
                pose_init is not None

        Returns:
            Tuple:
                pose_opt (Tensor): Shape (num_obj, 4 or 7), PnP solution y*
                cost (Tensor | None): Shape (num_obj, ), is not None when with_cost=True
                pose_opt_plus (Tensor | None): Shape (num_obj, 4 or 7), y* + Δy, used in derivative
                    regularization loss, is not None when with_pose_opt_plus=True, can be backpropagated
                pose_samples (Tensor): Shape (mc_samples, num_obj, 4 or 7)
                pose_sample_logweights (Tensor): Shape (mc_samples, num_obj), can be backpropagated
                cost_init (Tensor | None): Shape (num_obj, ), is None when pose_init is None, can be
                    backpropagated
        """
        if self.normalize:
            transform, x3d, pose_init = pnp_normalize(x3d, pose_init, detach_transformation=True)

        assert x3d.dim() == x2d.dim() == w2d.dim() == 3
        num_obj = x3d.size(0)

        evaluate_fun = partial(
            evaluate_pnp,
            x3d=x3d, x2d=x2d, w2d=w2d, camera=camera, cost_fun=cost_fun, out_cost=True)
        cost_init = evaluate_fun(pose=pose_init)[1] if pose_init is not None else None

        pose_opt, pose_cov, cost, pose_opt_plus = self.solver(
            x3d, x2d, w2d, camera, cost_fun,
            pose_init=pose_init, cost_init=cost_init,
            with_pose_cov=True, force_init_solve=force_init_solve,
            normalize_override=False, **kwargs)

        if num_obj > 0:
            pose_samples = x3d.new_empty((self.num_iter, self.iter_samples) + pose_opt.size())
            logprobs = x3d.new_empty((self.num_iter, self.num_iter, self.iter_samples, num_obj))
            cost_pred = x3d.new_empty((self.num_iter, self.iter_samples, num_obj))

            distr_params = self.allocate_buffer(num_obj, dtype=x3d.dtype, device=x3d.device)

            with torch.no_grad():
                self.initial_fit(pose_opt, pose_cov, camera, *distr_params)

            for i in range(self.num_iter):
                # ===== step 1: generate samples =====
                new_trans_distr, new_rot_distr = self.gen_new_distr(i, *distr_params)
                # (iter_sample, num_obj, 3)
                pose_samples[i, :, :, :3] = new_trans_distr.sample((self.iter_samples, ))
                # (iter_sample, num_obj, 1 or 4)
                pose_samples[i, :, :, 3:] = new_rot_distr.sample((self.iter_samples, ))

                # ===== step 2: evaluate integrand =====
                cost_pred[i] = evaluate_fun(pose=pose_samples[i])[1]

                # ===== step 3: evaluate proposal mixture logprobs =====
                # (i + 1, iter_sample, num_obj)
                # all samples (i + 1, iter_sample, num_obj) on new distr (num_obj, )
                logprobs[i, :i + 1] = new_trans_distr.log_prob(pose_samples[:i + 1, :, :, :3]) \
                                      + new_rot_distr.log_prob(pose_samples[:i + 1, :, :, 3:]).flatten(2)
                if i > 0:
                    old_trans_distr, old_rot_distr = self.gen_old_distr(i, *distr_params)
                    # (i, iter_sample, num_obj)
                    # new samples (iter_sample, num_obj) on old distr (i, 1, num_obj)
                    logprobs[:i, i] = old_trans_distr.log_prob(pose_samples[i, :, :, :3]) \
                                      + old_rot_distr.log_prob(pose_samples[i, :, :, 3:]).flatten(2)
                # (i + 1, i + 1, iter_sample, num_obj) -> (i + 1, iter_sample, num_obj)
                mix_logprobs = torch.logsumexp(logprobs[:i + 1, :i + 1], dim=0) - math.log(i + 1)

                # ===== step 4: get sample weights =====
                # (i + 1, iter_sample, num_obj)
                pose_sample_logweights = -cost_pred[:i + 1] - mix_logprobs

                # ===== step 5: estimate new proposal =====
                if i == self.num_iter - 1:
                    break  # break at last iter
                with torch.no_grad():
                    self.estimate_params(
                        i,
                        pose_samples[:i + 1].reshape(((i + 1) * self.iter_samples, ) + pose_opt.size()),
                        pose_sample_logweights.reshape((i + 1) * self.iter_samples, num_obj),
                        *distr_params)

            pose_samples = pose_samples.reshape((self.mc_samples, ) + pose_opt.size())
            pose_sample_logweights = pose_sample_logweights.reshape(self.mc_samples, num_obj)

        else:
            pose_samples = x2d.new_zeros((self.mc_samples, ) + pose_opt.size())
            pose_sample_logweights = x3d.reshape(self.mc_samples, 0) \
                                     + x2d.reshape(self.mc_samples, 0) \
                                     + w2d.reshape(self.mc_samples, 0)

        if self.normalize:
            pose_opt = pnp_denormalize(transform, pose_opt)
            pose_samples = pnp_denormalize(transform, pose_samples)
            if pose_opt_plus is not None:
                pose_opt_plus = pnp_denormalize(transform, pose_opt_plus)

        return pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_init


class EProPnP4DoF(EProPnPBase):
    """
    End-to-End Probabilistic Perspective-n-Points for 4DoF pose estimation.
    The pose is parameterized as [x, y, z, yaw], where yaw is the
    rotation around the Y-axis in radians.
    Adopted proposal distributions:
        position: multivariate t-distribution, degrees of freedom = 3
        orientation: 0.75 von Mises distribution + 0.25 uniform distribution
    """

    def allocate_buffer(self, num_obj, dtype=torch.float32, device=None):
        trans_mode = torch.empty((self.num_iter, num_obj, 3), dtype=dtype, device=device)
        trans_cov_tril = torch.empty((self.num_iter, num_obj, 3, 3), dtype=dtype, device=device)
        rot_mode = torch.empty((self.num_iter, num_obj, 1), dtype=dtype, device=device)
        rot_kappa = torch.empty((self.num_iter, num_obj, 1), dtype=dtype, device=device)
        return trans_mode, trans_cov_tril, rot_mode, rot_kappa

    def initial_fit(self, pose_opt, pose_cov, camera, trans_mode, trans_cov_tril,
                    rot_mode, rot_kappa):
        trans_mode[0], rot_mode[0] = pose_opt.split([3, 1], dim=-1)
        trans_cov_tril[0] = cholesky_wrapper(pose_cov[:, :3, :3], [1.0, 1.0, 4.0])
        rot_kappa[0] = 0.33 / pose_cov[:, 3, 3, None].clamp(min=self.eps)

    @staticmethod
    def gen_new_distr(iter_id, trans_mode, trans_cov_tril, rot_mode, rot_kappa):
        new_trans_distr = MultivariateStudentT(3, trans_mode[iter_id], trans_cov_tril[iter_id])
        new_rot_distr = VonMisesUniformMix(rot_mode[iter_id], rot_kappa[iter_id])
        return new_trans_distr, new_rot_distr

    @staticmethod
    def gen_old_distr(iter_id,
                      trans_mode, trans_cov_tril,
                      rot_mode, rot_kappa):
        mix_trans_distr = MultivariateStudentT(
            3, trans_mode[:iter_id, None], trans_cov_tril[:iter_id, None])
        mix_rot_distr = VonMisesUniformMix(
            rot_mode[:iter_id, None], rot_kappa[:iter_id, None])
        return mix_trans_distr, mix_rot_distr

    def estimate_params(self, iter_id, pose_samples, pose_sample_logweights,
                        trans_mode, trans_cov_tril, rot_mode, rot_kappa):
        sample_weights_norm = torch.softmax(pose_sample_logweights, dim=0)
        # translation var mean
        # (cum_sample, num_obj, 3) -> (num_obj, 3)
        trans_mode[iter_id + 1] = (sample_weights_norm[..., None] * pose_samples[..., :3]).sum(dim=0)
        trans_dev = pose_samples[..., :3] - trans_mode[iter_id + 1]  # (cum_sample, num_obj, 3)
        # (cum_sample, num_obj, 1, 1) * (cum_sample, num_obj, 3, 1)
        # * (cum_sample, num_obj, 1, 3) -> (num_obj, 3, 3)
        trans_cov = (sample_weights_norm[..., None, None] * trans_dev.unsqueeze(-1)
                     * trans_dev.unsqueeze(-2)).sum(dim=0)
        trans_cov_tril[iter_id + 1] = cholesky_wrapper(trans_cov, [1.0, 1.0, 4.0])
        # rotation estimation
        mean_vector = pose_samples.new_empty((pose_samples.size(1), 2))  # [sin, cos]
        # (cum_sample, num_obj, 1) -> (num_obj, 1)
        torch.sum(sample_weights_norm[..., None] * pose_samples[..., 3:].sin(), dim=0,
                  out=mean_vector[:, :1])
        torch.sum(sample_weights_norm[..., None] * pose_samples[..., 3:].cos(), dim=0,
                  out=mean_vector[:, 1:])
        rot_mode[iter_id + 1] = torch.atan2(mean_vector[:, :1], mean_vector[:, 1:])
        r_sq = torch.square(mean_vector).sum(dim=-1, keepdim=True)  # (num_obj, 1)
        rot_kappa[iter_id + 1] = 0.33 * r_sq.sqrt().clamp(min=self.eps) \
                                 * (2 - r_sq) / (1 - r_sq).clamp(min=self.eps)


class EProPnP6DoF(EProPnPBase):
    """
    End-to-End Probabilistic Perspective-n-Points for 6DoF pose estimation.
    The pose is parameterized as [x, y, z, w, i, j, k], where [w, i, j, k]
    is the unit quaternion.
    Adopted proposal distributions:
        position: multivariate t-distribution, degrees of freedom = 3
        orientation: angular central Gaussian distribution
    """

    def __init__(self,
                 *args,
                 acg_mle_iter=3,
                 acg_dispersion=0.001,
                 **kwargs):
        super(EProPnP6DoF, self).__init__(*args, **kwargs)
        self.acg_mle_iter = acg_mle_iter
        self.acg_dispersion = acg_dispersion

    def allocate_buffer(self, num_obj, dtype=torch.float32, device=None):
        trans_mode = torch.empty((self.num_iter, num_obj, 3), dtype=dtype, device=device)
        trans_cov_tril = torch.empty((self.num_iter, num_obj, 3, 3), dtype=dtype, device=device)
        rot_cov_tril = torch.empty((self.num_iter, num_obj, 4, 4), dtype=dtype, device=device)
        return trans_mode, trans_cov_tril, rot_cov_tril

    def initial_fit(self,
                    pose_opt, pose_cov, camera,
                    trans_mode, trans_cov_tril,
                    rot_cov_tril):
        trans_mode[0], rot_mode = pose_opt.split([3, 4], dim=-1)
        trans_cov_tril[0] = cholesky_wrapper(pose_cov[:, :3, :3])

        eye_4 = torch.eye(4, dtype=pose_opt.dtype, device=pose_opt.device)
        transform_mat = camera.get_quaternion_transfrom_mat(rot_mode)
        rot_cov = (transform_mat @ pose_cov[:, 3:, 3:].inverse() @ transform_mat.transpose(-1, -2)
                   + eye_4).inverse()
        rot_cov.div_(rot_cov.diagonal(
            offset=0, dim1=-1, dim2=-2).sum(-1)[..., None, None])
        rot_cov_tril[0] = cholesky_wrapper(
            rot_cov + rot_cov.det()[:, None, None] ** 0.25 * (self.acg_dispersion * eye_4))

    @staticmethod
    def gen_new_distr(iter_id, trans_mode, trans_cov_tril, rot_cov_tril):
        new_trans_distr = MultivariateStudentT(3, trans_mode[iter_id], trans_cov_tril[iter_id])
        new_rot_distr = AngularCentralGaussian(rot_cov_tril[iter_id])
        return new_trans_distr, new_rot_distr

    @staticmethod
    def gen_old_distr(iter_id, trans_mode, trans_cov_tril, rot_cov_tril):
        mix_trans_distr = MultivariateStudentT(
            3, trans_mode[:iter_id, None], trans_cov_tril[:iter_id, None])
        mix_rot_distr = AngularCentralGaussian(rot_cov_tril[:iter_id, None])
        return mix_trans_distr, mix_rot_distr

    def estimate_params(self, iter_id, pose_samples, pose_sample_logweights,
                        trans_mode, trans_cov_tril, rot_cov_tril):
        sample_weights_norm = torch.softmax(pose_sample_logweights, dim=0)
        # translation var mean
        # (cum_sample, num_obj, 3) -> (num_obj, 3)
        trans_mode[iter_id + 1] = (sample_weights_norm[..., None] * pose_samples[..., :3]).sum(dim=0)
        trans_dev = pose_samples[..., :3] - trans_mode[iter_id + 1]  # (cum_sample, num_obj, 3)
        # (cum_sample, num_obj, 1, 1) * (cum_sample, num_obj, 3, 1)
        # * (cum_sample, num_obj, 1, 3) -> (num_obj, 3, 3)
        trans_cov = (sample_weights_norm[..., None, None] * trans_dev.unsqueeze(-1)
                     * trans_dev.unsqueeze(-2)).sum(dim=0)
        trans_cov_tril[iter_id + 1] = cholesky_wrapper(trans_cov)
        # rotation estimation
        eye_4 = torch.eye(4, dtype=pose_samples.dtype, device=pose_samples.device)
        rot = pose_samples[..., 3:]  # (cum_sample, num_obj, 4)
        r_r_t = rot[:, :, :, None] * rot[:, :, None, :]  # (cum_sample, num_obj, 4, 4)
        rot_cov = eye_4.expand(pose_samples.size(1), 4, 4).clone()
        for _ in range(self.acg_mle_iter):
            # (cum_sample, num_obj, 1, 1)
            M = rot[:, :, None, :] @ rot_cov.inverse() @ rot[:, :, :, None]
            invM_weighted = sample_weights_norm[..., None, None] / M.clamp(min=self.eps)
            invM_weighted_norm = invM_weighted / invM_weighted.sum(dim=0)
            # (num_obj, 4, 4) trace equals 1
            rot_cov = (invM_weighted_norm * r_r_t).sum(dim=0) + eye_4 * self.eps
        rot_cov_tril[iter_id + 1] = cholesky_wrapper(
            rot_cov + rot_cov.det()[:, None, None] ** 0.25 * (self.acg_dispersion * eye_4))
