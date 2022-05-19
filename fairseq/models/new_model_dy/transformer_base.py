# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple
import copy
import random
import torch
import torch.nn as nn
from torch import Tensor
from fairseq.models.new_model_dy.utils import *
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.new_model_dy import (
    TransformerConfig,
    TransformerDecoderBase,
    TransformerEncoderBase,
)


class TransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True
        # self.future_linear = nn.Linear(256,6000)
        # self.past_linear = nn.Linear(256,6000)
        # self.img_linear = nn.Linear(2048,256)
        # self.L2_linear = nn.Linear(cfg.decoder.embed_dim,cfg.decoder.embed_dim)
        # self.relu = nn.ReLU(inplace=True)
        # self.output_layer = nn.Linear(
        #         cfg.decoder.output_dim, len(self.decoder.dictionary)
        #     )
        # nn.init.normal_(
        #         self.output_layer.weight, mean=0, std=cfg.decoder.output_dim**-0.5
        #     )
        # self.relu = nn.Sigmoid()
    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        img_global,
        en_de,
        matrix,
        fr_o,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """        
        # future_info = get_not_translated_for_all_mbert_one_hot_step(self.decoder.dictionary,prev_output_tokens,en_de)
        # future_info = self.decoder.embed_tokens(future_info)
        # print('fu',future_info.shape)
        # assert 1 == 0
        # future_info = tgt_mask(self.decoder.dictionary,prev_output_tokens,en_de)
        # future_info = self.decoder.embed_tokens(future_info)

        # print(graph)
        # assert 1 == 0

        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,img_global = img_global,matrix = matrix
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            img_global = img_global,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        # img_global_emb = encoder_out['img_global']#转换一下形状
        # word_emb = decoder_out[1]['future_caps']
        # word_emb_h = decoder_out[1]["no_gate_x"]
        # word_emb = self.relu(self.L2_linear(word_emb))

        # sim = calculate_cosine_distance(word_emb,img_global_emb,fr_o)
        # if sim.data < 0:
        #     sim.data = torch.tensor(0).cuda()
        
        # sim_h = calculate_cosine_distance_plus(word_emb_h,img_global_emb,fr_o)
        # if sim2.data < 0:
        #     sim2.data = torch.tensor(0).cuda()

        
        # future_info_emb = self.decoder.embed_tokens(future_info)
        # future_info_emb = self.decoder.output_layer(future_info_emb)
        # decoder_out[1]['future_caps'] = self.output_layer(decoder_out[1]['future_caps'])

        # decoder_out[1]['dictionary'] = self.decoder.dictionary
        # decoder_out[1]['embeding'] = self.decoder.embed_tokens
        # decoder_out[1]['en_de'] = en_de
        # decoder_out[1]['prev_output_tokens'] = prev_output_tokens
        # decoder_out[1]['future_caps'] = self.future_linear(decoder_out[1]['future_caps'])
        # decoder_out[1]['past_caps'] = self.past_linear(decoder_out[1]['past_caps'])
        # decoder_out[1]['sim_h'] = sim_h
        # decoder_out[1]['sim2'] = sim2
        
        return decoder_out


    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


