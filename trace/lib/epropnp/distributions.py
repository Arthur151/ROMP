"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import math
import numpy as np
import torch

from torch.distributions import VonMises
from torch.distributions.multivariate_normal import _batch_mahalanobis, _standard_normal, _batch_mv
from pyro.distributions import TorchDistribution, constraints
from pyro.distributions.util import broadcast_shape


class AngularCentralGaussian(TorchDistribution):
    arg_constraints = {'scale_tril': constraints.lower_cholesky}
    has_rsample = True

    def __init__(self, scale_tril, validate_args=None, eps=1e-6):
        q = scale_tril.size(-1)
        assert q > 1
        assert scale_tril.shape[-2:] == (q, q)
        batch_shape = scale_tril.shape[:-2]
        event_shape = (q,)
        self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        self._unbroadcasted_scale_tril = scale_tril
        self.q = q
        self.area = 2 * math.pi ** (0.5 * q) / math.gamma(0.5 * q)
        self.eps = eps
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.expand(
            broadcast_shape(value.shape[:-1], self._unbroadcasted_scale_tril.shape[:-2])
            + self.event_shape)
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, value)
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return M.log() * (-self.q / 2) - half_log_det - math.log(self.area)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        normal = _standard_normal(shape,
                                  dtype=self._unbroadcasted_scale_tril.dtype,
                                  device=self._unbroadcasted_scale_tril.device)
        gaussian_samples = _batch_mv(self._unbroadcasted_scale_tril, normal)
        gaussian_samples_norm = gaussian_samples.norm(dim=-1)
        samples = gaussian_samples / gaussian_samples_norm.unsqueeze(-1)
        samples[gaussian_samples_norm < self.eps] = samples.new_tensor(
            [1.] + [0. for _ in range(self.q - 1)])
        return samples


class VonMisesUniformMix(VonMises):

    def __init__(self, loc, concentration, uniform_mix=0.25, **kwargs):
        super(VonMisesUniformMix, self).__init__(loc, concentration, **kwargs)
        self.uniform_mix = uniform_mix

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        assert len(sample_shape) == 1
        x = np.empty(tuple(self._extended_shape(sample_shape)), dtype=np.float32)
        uniform_samples = round(sample_shape[0] * self.uniform_mix)
        von_mises_samples = sample_shape[0] - uniform_samples
        x[:uniform_samples] = np.random.uniform(
            -math.pi, math.pi, size=tuple(self._extended_shape((uniform_samples,))))
        x[uniform_samples:] = np.random.vonmises(
            self.loc.cpu().numpy(), self.concentration.cpu().numpy(),
            size=tuple(self._extended_shape((von_mises_samples,))))
        return torch.from_numpy(x).to(self.loc.device)

    def log_prob(self, value):
        von_mises_log_prob = super(VonMisesUniformMix, self).log_prob(value) + np.log(1 - self.uniform_mix)
        log_prob = torch.logaddexp(
            von_mises_log_prob,
            torch.full_like(von_mises_log_prob, math.log(self.uniform_mix / (2 * math.pi))))
        return log_prob
