# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch.nn as nn


@dataclass
class NewUpTirlCriterionConfig(FairseqDataclass):
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
    "new_up_tirl_criterion", dataclass=NewUpTirlCriterionConfig
)
class NewUpTirlCriterion(FairseqCriterion):
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
        self.l1loss = nn.SmoothL1Loss()
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        ## 主loss
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        ## l1范数
        # embeding = net_output[1]['embeding']
        # future_caps_emb = net_output[1]['future_caps']#输出h
        # target = sample["target"]
        # target_emb = embeding(target)#目标语言词向量
        # future_caps_emb = bottle(future_caps_emb)
        # target_emb = bottle(target_emb)
        # # l1_loss = self.l1loss(future_caps_emb,target_emb)
        # l1_loss = self.cos_sim(future_caps_emb,target_emb).sum()

        ## l1交叉熵
        target = model.get_targets(sample, net_output)
        # target_label = convert_to_future_labels(target, padding_idx=1)
        target_label = convert_to_past_labels(target, padding_idx=1)
        target_label = torch.flip(target_label,dims = [1])


        l1_loss_list = []
        future_caps = net_output[1]['future_caps']


        for index in range(len(future_caps)):
            l1_lprobs = model.get_normalized_probs(tuple([future_caps[index]]), log_probs=True)
            l1_loss, l1_nll_loss = label_smoothed_nll_loss(
                l1_lprobs,
                target_label[:,index],
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            l1_loss_list.append(l1_loss)
        # print(len(l1_loss_list))
        # print(sum(l1_loss_list))
        # assert 1 == 0

        alpha = 0.6



        l2_loss = net_output[1]['sim']
        l3_loss = net_output[1]['sim2']
        l = (1 - alpha) * sum(l1_loss_list) + alpha * (l2_loss + l3_loss)## min(l1 - l2)
        # l = (1 - alpha) * l1_loss + alpha * l2_loss## min(l1 - l2)



        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data+l.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss + l, sample_size, logging_output

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

    # tile through timesteps
    # [batch, seq_len, seq_len]
    future_labels = labels.unsqueeze(1).repeat(1, seq_len, 1)
    # masking padded position by padding_idx
    future_labels.masked_fill_(1 - mask, padding_idx)

    return future_labels

def convert_to_past_labels(labels, padding_idx=1):
    """
    Args:
        padding_idx:
        labels: [batch, seq_len]

    Returns:
        descending labels .
            [batch, seq_len, seq_len]
    """
    batch_size, seq_len = labels.size()
    seq_mask = labels.ne(padding_idx)

    # use upper triangle to masking in descending manner
    # [batch, seq_len, seq_len]
    step_mask = torch.tril(labels.new_ones(seq_len, seq_len), 0).byte()
    mask = step_mask.unsqueeze(0) * seq_mask.unsqueeze(1)

    # tile through timesteps
    # [batch, seq_len, seq_len]
    past_labels = labels.unsqueeze(1).repeat(1, seq_len, 1)
    # masking padded position by padding_idx
    past_labels.masked_fill_(1 - mask, padding_idx)

    return past_labels