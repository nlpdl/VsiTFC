# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from collections import OrderedDict
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.nn as nn


@dataclass
class NewDyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def bottle(v):
        return v.contiguous().view(-1, v.size(2))

@register_criterion(
    "new_dy_criterion", dataclass=NewDyCriterionConfig
)
class NewDyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.criterion = MultiCriterion(
            weights=dict(wploss_future=1.),
            wploss_future=MultiTargetNMTCriterion(label_smoothing=label_smoothing)
        )
        # self.l1loss = nn.SmoothL1Loss()
        # self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        ## ???loss
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        ## l1??????
        # embeding = net_output[1]['embeding']
        # future_caps_emb = net_output[1]['future_caps']#??????h
        # target = sample["target"]
        # target_emb = embeding(target)#?????????????????????
        # future_caps_emb = bottle(future_caps_emb)
        # target_emb = bottle(target_emb)
        # # l1_loss = self.l1loss(future_caps_emb,target_emb)
        # l1_loss = self.cos_sim(future_caps_emb,target_emb).sum()

        # l1?????????
        target = model.get_targets(sample, net_output)
        l1_lprobs = model.get_normalized_probs(tuple([net_output[1]['future_caps']]), log_probs=True)
        l1_loss, l1_nll_loss = label_smoothed_nll_loss(
            l1_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        l3_loss = net_output[1]['sim2']

        # future_caps_lprobs = model.get_normalized_probs(tuple([net_output[1]['future_caps']]), log_probs=True)
        # target_label, target_scores= convert_to_future_labels(target)
        # params_wploss_future = dict(
        #         inputs=future_caps_lprobs,
        #         labels=target_label,
        #         target_scores=target_scores,
        #         update=True)  
        # params_dict = {}
        # params_dict["wploss_future"] = params_wploss_future
        # l1_loss = self.criterion(
        #     reduce=reduce,
        #     normalization=1.0,
        #     # params for each criterion
        #     **params_dict
        # )
        # l1_loss = l1_loss['loss']


        # alpha = 0.5 ??????????????????????????????
        beta = 0.6
        alpha = 0.8


        l2_loss = net_output[1]['sim']
        # l = l1_loss + l2_loss## min(l1 - l2)
        # l = (1 - alpha) * l1_loss + alpha * l2_loss## min(l1 - l2)
        l = (1 - alpha) * l1_loss + alpha * (l2_loss + l3_loss)## min(l1 - l2)
        # l = (1 - alpha) * l1_loss + alpha * l2_loss## min(l1 - l2)


        # final_loss = (1 - alpha) * loss + alpha * (l2_loss + l3_loss) + l1_loss
        # final_loss = loss + alpha * l2_loss + l1_loss + beta * l3_loss 
        final_loss = loss + l



        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": final_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return final_loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

class Criterion(nn.Module):
    def _compute_loss(self, inputs, labels, **kwargs):
        raise NotImplementedError

    def forward(self, inputs, labels, normalization=1.0, reduce=True, **kwargs):
        loss = self._compute_loss(inputs, labels, **kwargs).div(normalization)  # (batch, )

        if reduce:
            loss = loss.sum()

        return loss

class NMTCriterion(Criterion):
    def __init__(self, padding_idx=1, label_smoothing=0.0):

        super().__init__()

        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing

        if label_smoothing > 0:

            self.criterion = nn.KLDivLoss(size_average=False, reduce=False)

        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=padding_idx, reduce=False)

        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 2))
        one_hot[0][self.padding_idx] = 0
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _compute_loss(self, inputs, labels, **kwargs):
        batch_size = labels.size(0)

        scores = self._bottle(inputs)  # [batch_size * seq_len, d_words]

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            tdata = gtruth.detach()

            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()  # mask of PAD

            one_hot = self._smooth_label(num_tokens)  # Do label smoothing
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if mask.numel() > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_.detach()

        loss = self.criterion(scores, gtruth).view((batch_size, -1)).sum(-1)
        return loss



class MultiTargetNMTCriterion(NMTCriterion):
    def _construct_target(self, targets, target_scores, num_tokens):
        """
        Args:
            targets: A Tensor with shape [batch*length, max_target] represents the indices of
                targets in the vocabulary.
            target_scores: A Tensor with shape [batch*length, max_target] represents the
                probabilities of targets in the vocabulary.
            num_tokens: An Integer represents the total number of words.

        Returns:
            A Tensor with shape [batch*length, num_tokens].
        """
        # Initialize a temporary tensor.
        if self.confidence < 1:
            tmp = self._smooth_label(num_tokens)  # Do label smoothing
            target_scores = target_scores * self.confidence
        else:
            tmp = torch.zeros(1, num_tokens)
        if targets.is_cuda:
            tmp = tmp.cuda()

        pad_positions = torch.nonzero(target_scores.sum(-1).eq(0)).squeeze()

        # [batch*length, num_tokens]
        tmp = tmp.repeat(targets.size(0), 1)

        if torch.numel(pad_positions) > 0:
            tmp.index_fill_(0, pad_positions, 0.)
        tmp.scatter_(1, targets, 0.)
        tmp.scatter_add_(1, targets, target_scores)

        return tmp

    def _compute_loss(self, inputs, labels, **kwargs):
        batch_size = labels.size(0)
        scores = self._bottle(inputs)  # [batch_size * seq_len, num_tokens]
        num_tokens = scores.size(-1)

        targets = self._bottle(labels)  # [batch_size * seq_len, max_target]
        target_scores = self._bottle(kwargs['target_scores'])
        
        gtruth = self._construct_target(targets, target_scores, num_tokens)
        loss = self.criterion(scores, gtruth)
        loss = loss.sum(-1).view(batch_size, -1)  # [batch, seq_len]
        length_norm = kwargs['target_scores'].sum(-1).ne(0).float().sum(-1)  # [batch, ]
        loss = loss.sum(-1).div(length_norm)

        return loss

def convert_to_future_labels(labels, padding_idx=1):
    """
    Args:
        padding_idx:
        labels: [batch, seq_len]

    Returns:
        future labels .
            [batch, seq_len, seq_len]
    """
    batch_size, seq_len = labels.size()
    seq_mask = labels.ne(padding_idx)

    # use upper triangle to masking in descending manner
    # [batch, seq_len, seq_len]
    step_mask = torch.triu(labels.new_ones(seq_len, seq_len), 1).byte()
    mask = step_mask.unsqueeze(0) * seq_mask.unsqueeze(1)
    scores = get_average_score(mask)

    # tile through timesteps
    # [batch, seq_len, seq_len]
    future_labels = labels.unsqueeze(1).repeat(1, seq_len, 1)
    # masking padded position by padding_idx
    future_labels.masked_fill_(1 - mask, padding_idx)

    return future_labels, scores

def get_average_score(mask):
    mask = mask.float()
    scores = mask / mask.sum(-1, keepdim=True)
    scores = torch.where(torch.isnan(scores),
                         torch.zeros_like(scores),
                         scores)
    return scores

class MultiCriterion(nn.Module):
    """
        Class for easily managing multiple criterions, which receives multiple instances of `
    Criterion`, computing loss respectively and summing them.
    """

    def __init__(self, weights=None, **named_criterions):
        """
        Args:
            weights (dict: weights for each criterions
            **named_criterions: {criterion_name: criterion}
        Notes:
            Each criterion must implement `_compute_loss(**kwargs)`
        """
        super(MultiCriterion, self).__init__()

        # remove None items.
        for kk in list(named_criterions.keys()):
            if named_criterions[kk] is None:
                named_criterions.pop(kk)

        for name, criterion in named_criterions.items():
            assert hasattr(criterion, "_compute_loss"), \
                "{} ({}) must have method \"_compute_loss\"".format(criterion, name)

        self.criterions = nn.Sequential(OrderedDict(named_criterions))
        self.num_criterions = len(self.criterions)
        self.names = list(named_criterions.keys())

        self.weights = weights if weights is not None \
            else {name: 1. for name in self.names}

    def add(self, name, criterion, weight=None):
        self._assert(name, criterion)

        self.names.append(name)
        self.criterions.add_module(name, criterion)
        if weight is None:
            weight = 1.
        self.weights[name] = weight

    @staticmethod
    def _assert(name, criterion):
        assert hasattr(criterion, "_compute_loss"), \
            "{} ({}) must have method \"_compute_loss\"".format(criterion, name)

    def compute_loss(self, **named_states):
        """
        Compute each loss respectively, and summing them.

        Args:
            named_states (dict): key-value params corresponds to specific criterion (name)
        Returns:
            losses (dict): dictionary of each named losses and a final loss summing of them.
        """
        losses = dict()
        list_losses = []
        for name, criterion in self.criterions._modules.items():
            # scalar
            if named_states[name] is None:
                loss = torch.tensor(0.0)
                loss = loss.cuda()
            else:
                loss = criterion._compute_loss(**named_states[name])  # [batch,]
            losses[name] = loss
            if named_states[name] is not None and named_states[name]['update'] is True:
                list_losses.append(self.weights[name] * loss)

        # losses["loss"] = torch.cat(list_losses, dim=-1).sum(-1)  # scalar
        losses["loss"] = sum(list_losses)  # [batch, ]

        return losses

    def forward(self, normalization=1.0, reduce=True, **named_states):
        """
        Args:
            **named_states: inputs for criterions. Must match corresponding names.

        Returns:

        """
        losses = self.compute_loss(**named_states)  # dict of [batch, ]
        for kk in losses:
            losses[kk] = losses[kk].div(normalization)
            if reduce:
                losses[kk] = losses[kk].sum()
        return losses