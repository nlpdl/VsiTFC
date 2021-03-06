from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models.new_model_dy.logging import WARN


def EM_routing_by_agreement(actn_in, votes_in, votes_in_mask, beta_u, beta_a, lbd, iterations):
    """ EM routing-by-agreement
    Args:
        actn_in (torch.Tensor): Activations of inputs (like bottom capsules).
            [batch, num_in_caps]
        votes_in (torch.Tensor): Voter sequence of transformed inputs.
            [batch, length, num_in_caps, num_out_caps, dim_out_caps]
        votes_in_mask (torch.Tensor): mask for input
            [batch, length, num_in_caps, num_out_caps]
        beta_u (torch.Tensor): [num_out_caps]
        beta_a (torch.Tensor): [num_out_caps]
        lbd (float): lambda
        iterations (int): Routing iteration

    Returns:
        (torch.Tensor): Upper capsules.
            [batch, length, num_out_caps, dim_out_caps]
        (torch.Tensor): Last routing weights.
            [batch, length, num_in_caps, num_out_caps]
    """
    ln_2pi = math.log(2 * math.pi)
    eps = 1e-8

    def _e_step(_mu, _sigma_sq, _actn_out):
        """
        Args:
            _mu (torch.Tensor): Mean.
                [batch, length, num_out_caps, dim_out_caps]
            _sigma_sq (torch.Tensor): Squared sigma.
                [batch, length, num_out_caps, dim_out_caps]
            _actn_out:  Activation of output capsule.
                [batch, length, num_out_caps]

        Returns:
            (torch.Tensor): Routing weight.
                [batch, length, num_in_caps, num_out_caps]
        """
        _mu = _mu.unsqueeze(2)
        _sigma_sq = _sigma_sq.unsqueeze(2) + eps

        # [batch, length, num_in_caps, num_out_caps, dim_out_caps]
        _log_p_j = -1. * (votes_in - _mu) ** 2 / (2 * _sigma_sq) \
                   - torch.log(_sigma_sq.sqrt()) \
                   - 0.5 * ln_2pi

        # [batch, length, num_in_caps, num_out_caps]
        _log_ap = _log_p_j.sum(-1) + _actn_out[:, :, None, :].log()

        if votes_in_mask is not None:
            _log_ap = _log_ap.masked_fill(votes_in_mask, -1e18)

        _r = F.softmax(_log_ap, dim=3)

        return _r

    def _m_step(_r):
        """
        Args:
            _r (torch.Tensor): Routing weight. [batch, length, num_in_caps, num_out_caps]

        Returns:
            (torch.Tensor): Mean. [batch, length, num_out_caps, dim_out_caps]
            (torch.Tensor): Squared sigma. [batch, length, num_out_caps, dim_out_caps]
            (torch.Tensor): Activation of output capsule.
                [batch, length, num_out_caps, dim_out_caps]
        """
        # [batch, length, num_in_caps, num_out_caps]
        _actn_r = actn_in[:, None, :, None] * _r

        _actn_r = _actn_r / (_actn_r.sum(3, keepdim=True) + eps)

        # [batch, length, num_out_caps]
        _actn_r_sum = _actn_r.sum(2)

        # [batch, length, num_in_caps, num_out_caps]
        _r1 = _actn_r / (_actn_r_sum.unsqueeze(2) + eps)

        # [batch, length, num_in_caps, num_out_caps, 1]
        _r1 = _r1.unsqueeze(-1)

        # [batch, length, num_out_caps, dim_out_caps]
        _mu = (_r1 * votes_in).sum(2)
        _sigma_sq = (_r1 * ((votes_in - _mu.unsqueeze(2))**2)).sum(2)

        # [batch, length, num_out_caps, dim_out_caps]
        _cost = beta_u[None, None, :, None] + \
            _sigma_sq.add(eps).sqrt().log() * _actn_r_sum.unsqueeze(-1)

        # [batch, length, num_out_caps]
        _actn_out = F.sigmoid(lbd * (beta_a[None, None, :] - _cost.sum(-1)))

        return _mu, _sigma_sq, _actn_out

    # Initialize routing weights as zeros
    batch, length, num_in_caps, num_out_caps, dim_out_caps = votes_in.size()

    r = actn_in.new_full(
        [batch, length, num_in_caps, num_out_caps],
        1 / num_out_caps
    )

    # routing-by-agreement
    for i in range(iterations):
        mu, sigma_sq, actn_out = _m_step(r)
        if i < iterations - 1:
            r = _e_step(mu, sigma_sq, actn_out)

    return mu, r


class CapsuleLayer(nn.Module):
    def __init__(self, num_out_caps, num_in_caps, dim_in_caps, dim_out_caps, num_iterations=2,
                 share_route_weights_for_in_caps=False):
        super(CapsuleLayer, self).__init__()

        self.num_out_caps = num_out_caps

        self.num_iterations = num_iterations
        assert num_iterations > 1, "num_iterations must at least be 1."

        self.share_route_weights_for_in_caps = share_route_weights_for_in_caps
        
        if share_route_weights_for_in_caps:
            WARN("{}: Argument 'num_in_caps' will be ignored.".format(self.__class__))
            print('kljkljklj',num_out_caps,dim_in_caps,dim_out_caps)
            self.route_weights = nn.Parameter(0.01 * torch.randn(num_out_caps, dim_in_caps, dim_out_caps))
        else:
            self.route_weights = nn.Parameter(0.01 * torch.randn(num_in_caps, num_out_caps, dim_in_caps, dim_out_caps))

    def __repr__(self):
        return super().__repr__() + "\n(routing_weights): {}".format(self.route_weights.size())

    @staticmethod
    def squash(tensor, dim=-1, eps=1e-8):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        norm = torch.sqrt(squared_norm) + eps
        return scale * tensor / norm

    def forward(self, inputs_u, inputs_mask):
        """
        Args:
            inputs_u: Tensor. [batch_size, num_in_caps, dim_in_caps]
            inputs_mask: Tensor. [batch_size, num_in_caps]

        Returns: Tensor. [batch_size, num_out_caps, dim_out_caps]

        """
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        # Compute u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        if self.share_route_weights_for_in_caps:
            # priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, None, :, :, :]).squeeze(-2)
            inputs_u_r = inputs_u.view(batch_size * num_in_caps, dim_in_caps)
            route_weight_r = self.route_weights.transpose(0, 1).reshape(dim_in_caps, -1)
            priors_u_hat = inputs_u_r @ route_weight_r
            priors_u_hat = priors_u_hat.view(batch_size, num_in_caps, self.num_out_caps, -1)
        else:
            priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, :, :, :, :]).squeeze(-2)

        # Initialize logits
        # logits_b: [batch_size, num_in_caps, num_out_caps]
        logits_b = inputs_u.new_zeros(batch_size, num_in_caps, self.num_out_caps)

        # Routing
        for i in range(self.num_iterations):
            # probs: [batch_size, num_in_caps, num_out_caps]
            if inputs_mask is not None:
                logits_b = logits_b + inputs_mask.unsqueeze(-1) * -1e18
            probs_c = F.softmax(logits_b, dim=-1)
            # outputs_v: [batch_size, num_out_caps, dim_out_caps]
            outputs_v = self.squash((probs_c.unsqueeze(-1) * priors_u_hat).sum(dim=1))

            if i != self.num_iterations - 1:
                # delta_logits: [batch_size, num_in_caps, num_out_caps]
                delta_logits = (priors_u_hat * outputs_v.unsqueeze(1)).sum(dim=-1)
                logits_b = logits_b + delta_logits

        # outputs_v: [batch_size, num_out_caps, dim_out_caps]
        return outputs_v


class ContextualCapsuleLayer(CapsuleLayer):
    def __init__(self, num_out_caps, num_in_caps, dim_in_caps, dim_out_caps,
                 dim_context=None,
                 num_iterations=2,
                 share_route_weights_for_in_caps=False):
        super().__init__(num_out_caps, num_in_caps, dim_in_caps, dim_out_caps, num_iterations,
                         share_route_weights_for_in_caps)
        self.linear_u_hat = nn.Linear(dim_out_caps, dim_out_caps)
        self.linear_v = nn.Linear(dim_out_caps, dim_out_caps)
        self.linear_delta = nn.Linear(dim_out_caps, 1, False)

        self.dim_out_caps = dim_out_caps

        self.contextual = dim_context is not None
        if self.contextual:
            self.linear_c = nn.Linear(dim_context, dim_out_caps, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_u_hat.weight, 0, 0.001)
        nn.init.normal_(self.linear_v.weight, 0, 0.001)
        nn.init.normal_(self.linear_delta.weight, 0, 0.001)
        if self.contextual:
            nn.init.normal_(self.linear_c.weight, 0, 0.001)

    def compute_delta(self, priors_u_hat, outputs_v, contexts=None):
        """
        Args:
            priors_u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
            outputs_v: [batch_size, num_out_caps, dim_out_caps]
            contexts: [batch_size, dim_context]

        Returns: Tensor. [batch_size, num_in_caps, num_out_caps]

        """
        # [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        u = priors_u_hat
        v = outputs_v[:, None, :, :]
        # [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        delta = self.linear_u_hat(u) + self.linear_v(v)
        if self.contextual:
            c = contexts[:, None, None, :]
            delta = delta + self.linear_c(c)

        delta = F.tanh(delta)

        # [batch_size, num_in_caps, num_out_caps]
        delta = self.linear_delta(delta).squeeze(-1)  # [batch_size, num_in_caps, num_out_caps]
        delta = F.tanh(delta)
        return delta * (self.dim_out_caps ** -0.5)

    def compute_delta_sequence(self, priors_u_hat, outputs_v, contexts=None):
        """
        Args:
            priors_u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
            outputs_v: [batch_size, length, num_out_caps, dim_out_caps]
            contexts: [batch_size, length, dim_context]

        Returns: Tensor. [batch_size, length, num_in_caps, num_out_caps]

        """
        # [batch_size, length, num_in_caps, num_out_caps, dim_out_caps]
        u = priors_u_hat[:, None, :, :, :]
        v = outputs_v[:, :, None, :, :]
        # [batch_size, length, num_in_caps, num_out_caps, dim_out_caps]
        delta = self.linear_u_hat(u) + self.linear_v(v)

        # [batch, length, 1, 1, dim_context]
        c = contexts[:, :, None, None, :]
        # [batch, length, num_in_caps, num_out_caps, dim_out_caps]
        delta = delta + self.linear_c(c)

        delta = F.tanh(delta)

        # [batch_size, length, num_in_caps, num_out_caps]
        delta = self.linear_delta(delta).squeeze(-1)
        delta = F.tanh(delta)
        return delta * (self.dim_out_caps ** -0.5)

    def forward(self, inputs_u, inputs_mask, context=None):
        """
        Args:
            inputs_u: Tensor. [batch_size, num_in_caps, dim_in_caps]
            inputs_mask: Tensor. [batch_size, num_in_caps]
            context: [batch_size, dim_context]

        Returns: Tensor. [batch_size, num_out_caps, dim_out_caps]

        """
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        # Compute u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        if self.share_route_weights_for_in_caps:
            # priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, None, :, :, :]).squeeze(-2)
            inputs_u_r = inputs_u.view(batch_size * num_in_caps, dim_in_caps)
            route_weight_r = self.route_weights.transpose(0, 1).reshape(dim_in_caps, -1)
            priors_u_hat = inputs_u_r @ route_weight_r
            priors_u_hat = priors_u_hat.view(batch_size, num_in_caps, self.num_out_caps, -1)
        else:
            priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, :, :, :, :]).squeeze(-2)

        # Initialize logits
        # logits_b: [batch_size, num_in_caps, num_out_caps]
        logits_b = inputs_u.new_zeros(batch_size, num_in_caps, self.num_out_caps)

        # Routing
        for i in range(self.num_iterations):
            # probs: [batch_size, num_in_caps, num_out_caps]
            if inputs_mask is not None:
                logits_b = logits_b + inputs_mask.unsqueeze(-1) * -1e18
            probs_c = F.softmax(logits_b, dim=-1)
            # outputs_v: [batch_size, num_out_caps, dim_out_caps]
            outputs_v = self.squash((probs_c.unsqueeze(-1) * priors_u_hat).sum(dim=1))

            if i != self.num_iterations - 1:
                # delta_logits: [batch_size, num_in_caps, num_out_caps]
                delta_logits = self.compute_delta(priors_u_hat, outputs_v, context)
                logits_b = logits_b + delta_logits

        # outputs_v: [batch_size, num_out_caps, dim_out_caps]
        return outputs_v, probs_c

    def forward_sequence(self, inputs_u, inputs_mask, context_sequence=None, cache=None):
        """
        Args:
            inputs_u (torch.Tensor). [batch_size, num_in_caps, dim_in_caps]
            inputs_mask (torch.Tensor). [batch_size, num_in_caps]
            context_sequence (torch.Tensor) : [batch_size, length, dim_context]

        Returns: Tensor. [batch_size, length, num_out_caps, dim_out_caps]

        """
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        length = context_sequence.size(1)
        # Compute u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        if cache is not None:
            priors_u_hat = cache
        else:
            priors_u_hat = self.compute_caches(inputs_u)

        # Initialize logits
        # logits_b: [batch_size, length, num_in_caps, num_out_caps]
        logits_b = inputs_u.new_zeros(batch_size, length, num_in_caps, self.num_out_caps)
        # [batch, 1, num_in_caps, 1]
        if inputs_mask is not None:
            routing_mask = inputs_mask[:, None, :, None].expand_as(logits_b)

        # Routing
        for i in range(self.num_iterations):
            # probs: [batch_size, length, num_in_caps, num_out_caps]
            if inputs_mask is not None:
                logits_b = logits_b.masked_fill(routing_mask, -1e18)
            probs_c = F.softmax(logits_b, dim=-1)

            # # [batch, num_out_caps, length,
            # _interm = probs_c.permute([0, 3, 1, 2]) @ prior_u_hat.transpose(1, 2))
            # outputs_v: [batch_size, length, num_out_caps, dim_out_caps]
            outputs_v = self.squash((probs_c.unsqueeze(-1) * priors_u_hat.unsqueeze(1)).sum(2))

            if i != self.num_iterations - 1:
                # delta_logits: [batch_size, length, num_in_caps, num_out_caps]
                delta_logits = self.compute_delta_sequence(
                    priors_u_hat, outputs_v, context_sequence
                )
                logits_b = logits_b + delta_logits

        # outputs_v: [batch_size, length, num_out_caps, dim_out_caps]
        return outputs_v, probs_c

    def compute_caches(self, inputs_u):
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        if self.share_route_weights_for_in_caps:
            inputs_u_r = inputs_u.contiguous().view(batch_size * num_in_caps, dim_in_caps)
            route_weight_r = self.route_weights.transpose(0, 1).reshape(dim_in_caps, -1)
            priors_u_hat = inputs_u_r @ route_weight_r
            priors_u_hat = priors_u_hat.contiguous().view(batch_size, num_in_caps, self.num_out_caps, -1)
        else:
            priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, :, :, :,
                                                            :]).squeeze(-2)
        return priors_u_hat


class EMContextualCapsuleLayer(nn.Module):
    def __init__(self, num_out_caps, num_in_caps,
                 dim_in_caps, dim_out_caps, dim_context,
                 lbd=1e-03, num_iterations=2,
                 share_route_weights_for_in_caps=False):
        super(EMContextualCapsuleLayer, self).__init__()

        self.num_out_caps = num_out_caps

        self.num_iterations = num_iterations
        assert num_iterations > 1, "num_iterations must at least be 1."

        self.share_route_weights_for_in_caps = share_route_weights_for_in_caps
        if share_route_weights_for_in_caps:
            WARN("{}: Argument 'num_in_caps' will be ignored.".format(self.__class__))
            self.route_weights = nn.Parameter(
                0.01 * torch.randn(num_out_caps, dim_in_caps, dim_out_caps)
            )
        else:
            self.route_weights = nn.Parameter(
                0.01 * torch.randn(num_in_caps, num_out_caps, dim_in_caps, dim_out_caps)
            )

        self.linear_u_hat = nn.Linear(dim_out_caps, dim_out_caps)
        self.linear_c = nn.Linear(dim_context, dim_out_caps)
        self.linear_vote = nn.Linear(dim_out_caps, dim_out_caps)
        self.linear_actn = nn.Linear(dim_in_caps, 1)

        self.lbd = lbd
        self.beta_u = nn.Parameter(torch.zeros(num_out_caps))
        self.beta_a = nn.Parameter(torch.zeros(num_out_caps))

        self.reset_parameters()

    def __repr__(self):
        return super().__repr__() + "\n(routing_weights): {}".format(self.route_weights.size())

    def reset_parameters(self):
        nn.init.normal_(self.linear_u_hat.weight, 0, 0.001)
        nn.init.normal_(self.linear_vote.weight, 0, 0.001)
        nn.init.normal_(self.linear_c.weight, 0, 0.001)

    def compute_caches(self, inputs_u):
        """
        Args:
            inputs_ud (torch.Tensor): [batch, num_in_caps, dim_in_caps]

        Returns:

        """
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        if self.share_route_weights_for_in_caps:
            inputs_u_r = inputs_u.view(batch_size * num_in_caps, dim_in_caps)
            route_weight_r = self.route_weights.transpose(0, 1).reshape(dim_in_caps, -1)
            priors_u_hat = inputs_u_r @ route_weight_r
            priors_u_hat = priors_u_hat.view(batch_size, num_in_caps, self.num_out_caps, -1)
        else:
            priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, :, :, :,
                                                            :]).squeeze(-2)

        actn = F.sigmoid(self.linear_actn(inputs_u)).squeeze(-1)
        # actn = inputs_u.new_ones([batch_size, num_in_caps])
        return priors_u_hat, actn

    def forward_sequence(self, inputs_u, inputs_mask, context_sequence=None, cache=None):
        """
        Args:
            inputs_u (torch.Tensor). [batch_size, num_in_caps, dim_in_caps]
            inputs_mask (torch.Tensor). [batch_size, num_in_caps]
            context_sequence (torch.Tensor) : [batch_size, length, dim_context]

        Returns: Tensor. [batch_size, length, num_out_caps, dim_out_caps]

        """
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        length = context_sequence.size(1)
        # Compute u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        if cache is not None:
            (priors_u_hat, actn_in) = cache
        else:
            priors_u_hat, actn_in = self.compute_caches(inputs_u)

        # Routing
        # [batch_size, length, num_in_caps, num_out_caps, dim_out_caps]
        u = self.linear_u_hat(priors_u_hat[:, None, :, :, :])
        c = self.linear_c(context_sequence[:, :, None, None, :])

        # [batch_size, length, num_in_caps, num_out_caps, dim_out_caps]
        vote_in = self.linear_vote(F.tanh(u + c))
        # vote_in = F.tanh(u + c)

        # [batch, 1, num_in_caps, 1]
        routing_mask = inputs_mask[:, None, :, None].expand(
            batch_size, length, num_in_caps, self.num_out_caps
        )

        mu, routing_weights = EM_routing_by_agreement(
            actn_in, vote_in, routing_mask,
            self.beta_u, self.beta_a,
            self.lbd, self.num_iterations
        )

        # outputs_v: [batch_size, length, num_out_caps, dim_out_caps]
        return mu, routing_weights
