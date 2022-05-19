# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from asyncio.log import logger
from fairseq.modules.quant_noise import quant_noise
import math
from re import M
from tkinter import N, X
from typing import Any, Dict, List, Optional
# from legacy.fairseq.fairseq.models.transformer.transformer_decoder import attention
import torch.nn.functional as F
from torch.autograd import Variable
from fairseq.models.new_model_up_tirl.utils import *
import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.new_model_up_tirl import TransformerConfig
from .capsule import ContextualCapsuleLayer
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer, 
    MultiheadAttention,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBase":
        return "TransformerDecoder"
    else:
        return module_name


class TransformerDecoderBase(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.img_linear = nn.Linear(2048,embed_dim)
        self.pre_capsule_layer_norm = nn.LayerNorm(embed_dim)
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        # self.out_and_cap_ffn = MultiInputPositionwiseFeedForward(
        #             size=self.embed_dim, hidden_size=2048, dropout=cfg.dropout,
        #             inp_sizes=[self.embed_dim // 2]
        #         )
        

        # self.self_attn_layer_norm = LayerNorm(embed_dim, export=cfg.export)
        # self.final_layer_norm = LayerNorm(embed_dim, export=cfg.export)
        # self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        # self.dropout_module = FairseqDropout(
        #     cfg.dropout, module_name=self.__class__.__name__
        # )
        # self.activation_dropout_module = FairseqDropout(
        #     float(cfg.activation_dropout), module_name=self.__class__.__name__
        # )

        # self.fc1 = self.build_fc1(
        #     embed_dim,
        #     cfg.encoder.ffn_embed_dim,
        #     cfg.quant_noise.pq,
        #     cfg.quant_noise.pq_block_size,
        # )
        # self.fc2 = self.build_fc2(
        #     cfg.encoder.ffn_embed_dim,
        #     embed_dim,
        #     cfg.quant_noise.pq,
        #     cfg.quant_noise.pq_block_size,
        # )


        # self.out_and_cap_ffn = MultiInputPositionwiseFeedForward(
        #             size=self.embed_dim, hidden_size=2048, dropout=cfg.dropout,
        #             inp_sizes=[self.embed_dim // 2 ,self.embed_dim // 2]
        #         )
        self.gate = GateModel(embed_dim)
        capsule_num = 4
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)
        self.capsule_layer = ContextualCapsuleLayer(
                    num_out_caps=int(capsule_num*1.5), num_in_caps=None,
                    dim_in_caps=embed_dim,
                    dim_out_caps=embed_dim // capsule_num,
                    dim_context=embed_dim,
                    num_iterations=3,
                    share_route_weights_for_in_caps=True)
        self.linear_bca_past = nn.Linear(int(embed_dim // 2), embed_dim)
        self.linear_bca_future = nn.Linear(int(embed_dim // 2), embed_dim)
        # self.attention = RNNAttention(embed_dim)
        self.attention = LuongAttn('general', embed_dim)
        self.concat = nn.Linear(embed_dim * 2, embed_dim)

        self.img_attn = self.build_self_attention(256, cfg)


        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

    def residual_connection(self, x, residual):
        return residual + x

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    
    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=cfg.quant_noise.pq,
            qn_block_size=cfg.quant_noise.pq_block_size,
        )
    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        img_global = None,
        used_x = None,#上一时刻x
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            img_global = img_global,
            used_x =used_x,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        img_global = None,
        used_x = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            img_global = img_global,
            used_x = used_x,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        img_global = None,
        used_x = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        enc_mask = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        oov_enc = encoder_out["oov_enc"][0].transpose(0,1)
        final_x = encoder_out["final_x"][0]
        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)
        img_global = self.img_linear(img_global)
        emb = x

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        img_global = img_global.transpose(0, 1)

        

        



        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        # query_x_img = torch.cat([x,img_global])
        # residual = x
        # oov, oov_attn = self.img_attn(
        #         query=x,
        #         key=img_global,
        #         value=img_global,
        #     )
        # oov, oov_attn = self.img_attn(
        #         query=query_x_img,
        #         key=x,
        #         value=x,
        #     )
        
        # oov = oov.transpose(0,2)
        # oov = nn.Linear(oov.size(-1),x.size(0)).cuda()(oov)
        # oov = oov.transpose(0,2)
        # T x B x C -> B x T x C
        # oov = self.residual_connection(oov, residual)
        # oov = self.self_attn_layer_norm(oov)

        # residual = oov
        # oov = self.activation_fn(self.fc1(oov))
        # oov = self.activation_dropout_module(oov)
        # oov = self.fc2(oov)
        # oov = self.dropout_module(oov)
        # oov = self.residual_connection(oov, residual)
        # oov = self.final_layer_norm(oov)



        x = x.transpose(0, 1)
        # oov = oov.transpose(0,1)
        img_global = img_global.transpose(0,1)
        now_x = x#记录当前x
        # start_x = x
        if x.size(1) != 1:#如果不是解码
            start = torch.zeros(x.size(0),1)
            start_t = start.fill_(2).long().cuda()
            start_t = self.embed_tokens(start_t)
            start_x = slict_x(x, start_t)

        if x.size(1) == 1 and used_x != None:#如果是解码，而且不是第一个
            start_x = used_x#就等于上一时刻
        elif x.size(1) == 1 and used_x == None:#解码，但是是第一个
            start_de_x = torch.zeros(x.size(0),1)
            start_de_x_t = start_de_x.fill_(2).long().cuda()
            start_de_x_t = self.embed_tokens(start_de_x_t)
            start_x = start_de_x_t# 构造开始符
        
        up_tirl_x = up_mask_h(start_x,self.embed_tokens)
        up_tirl_x_list = []
        

        for i in up_tirl_x:
            capsule_query = self.pre_capsule_layer_norm(i)
            capsules, _ = self.capsule_layer.forward_sequence(
                    final_x,#encoder中的边融合
                    None,
                    capsule_query,
                )
            
            capsules = capsules.view(emb.size(0), emb.size(1), -1)
            
            (past_caps, future_caps,_) = torch.chunk(capsules, 3, -1)
            past_caps = self.linear_bca_past(past_caps)
            future_caps = self.linear_bca_future(future_caps)
            up_tirl_x_list.append(future_caps)

        # capsule_query = self.pre_capsule_layer_norm(start_x)
        # capsules, _ = self.capsule_layer.forward_sequence(
        #         final_x,#encoder中的边融合
        #         None,
        #         capsule_query,
        #     )
        
        # capsules = capsules.view(emb.size(0), emb.size(1), -1)
        
        # (past_caps, future_caps,_) = torch.chunk(capsules, 3, -1)
        # past_caps = self.linear_bca_past(past_caps)
        # future_caps = self.linear_bca_future(future_caps)


        # out = []
        # t_x = x.transpose(0,1)
        # t_future_caps = future_caps.transpose(0,1)
        # print(t_future_caps.size(0))
        # assert 1 == 0
        # for i in range(t_future_caps.size(0)):
        #     attn_weights = self.attention(t_x[i].unsqueeze(0), t_future_caps)
        #     context = attn_weights.bmm(t_future_caps.transpose(0,1))

        #     # Get attentional hidden state h˜t = tanh(Wc[ct;ht])
        #     temmp_x = t_x[i].squeeze(0)
        #     context = context.squeeze(1)

        #     contextual_input = torch.cat((temmp_x, context), 1)
        #     attentional_hidden = torch.tanh(self.concat(contextual_input))
        #     out.append(attentional_hidden.unsqueeze(1))
        # out = torch.cat(out,dim = 1)

        out = []
        t_x = x.transpose(0,1)
        # t_future_caps = future_caps.transpose(0,1)
        for i in range(t_x.size(0)):
            attn_weights = self.attention(t_x[i].unsqueeze(0), up_tirl_x_list[i].transpose(0,1))
            context = attn_weights.bmm(up_tirl_x_list[i])

            # Get attentional hidden state h˜t = tanh(Wc[ct;ht])
            temmp_x = t_x[i].squeeze(0)
            context = context.squeeze(1)

            contextual_input = torch.cat((temmp_x, context), 1)
            attentional_hidden = torch.tanh(self.concat(contextual_input))
            out.append(attentional_hidden.unsqueeze(1))
        out = torch.cat(out,dim = 1)

        no_gate_x = x
        
        # x = self.gate(x,future_caps)




        # output = self.attention(x, future_caps)
        # output = output.bmm(future_caps)
        # output = torch.cat((x, output), -1)
        # attentional_hidden = torch.tanh(self.concat(contextual_input))
        # print('out',output.shape)
        # assert 1 == 0
        # x = self.out_and_cap_ffn(x, future_caps)

        if self.layer_norm is not None:
            out = self.layer_norm(out)
        if self.project_out_dim is not None:
            out = self.project_out_dim(out)

        return out, {"attn": [attn], "inner_states": inner_states,"past_caps":past_caps,"future_caps":up_tirl_x_list,"x":x,"now_x":now_x,"no_gate_x":no_gate_x}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            TransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )


class GateModel(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gate_linear = nn.Linear(2 * embed_dim,1)
        self.gate_layer_norm = torch.nn.LayerNorm(1, 1e-5, True)
        self.et_layer_norm = torch.nn.LayerNorm(embed_dim, 1e-5, True)
    
    def forward(self,x,futrue):
        futrue = self.et_layer_norm(futrue)
        mix = torch.cat([x,futrue],dim = 2)
        gate = self.relu(self.gate_layer_norm(self.gate_linear(mix)))#64*t*1
        x = x +  gate * futrue
        return x


class RNNAttention(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.fc_softmax = nn.Softmax(dim=-1)
    
    def forward(self,s,futrue):
        
        futrue = self.fc(futrue)
        fc = s.transpose(1,2) @ futrue
        fc = self.fc_softmax(fc)
        print('fc',fc.shape)

        c = []
        for index in range(s.size(1)):
            # print(a[:,:index].sum(1).unsqueeze(-2).transpose(2,1).bmm(h[:,index:index+1]).shape)
            print('1',fc[:,index].shape)
            print('2',s.shape)
            print('3',fc[:,index].transpose(1,2).bmm(s).shape)
            print('4',s[:,index:index+1].shape)
            t = fc[:,index].sum(1).unsqueeze(-2).bmm(s)
            print('t',t.shape)
            assert 1 == 0
            # print(a[index].sum(1).unsqueeze(-2).transpose(2,1).shape)
            # t = a[index].sum(-1).unsqueeze(-2).bmm(h[:,index:index+1])
            
            # t = a[:,:index].sum(1).unsqueeze(-2).transpose(2,1) @ h[:,index:index+1]
            # t = t.transpose(2,1)
            # t = nn.Linear(t.size(-1),1).cuda()(t)
            # t = t.transpose(2,1)
            c.append(t)
        assert 1 == 0

class MultiInputPositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, size, hidden_size, dropout=0.1, inp_sizes=[]):
        super().__init__()
        self.total_input_size = size + sum(inp_sizes)
        self.w_1 = nn.Linear(self.total_input_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = nn.LayerNorm(self.total_input_size)

        # Save a little memory, by doing inplace.
        self.dropout_1 = nn.Dropout(dropout, inplace=False)
        self.relu = nn.ReLU(inplace=False)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, *x):
        inps = torch.cat(x, -1)
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(inps))))
        output = self.dropout_2(self.w_2(inter))
        return output + x[0]

class LuongAttn(torch.nn.Module):
  def __init__(self, method, hidden_size):
    super(LuongAttn, self).__init__()
    self.method = method
    if self.method not in ['dot', 'general', 'concat']:
      raise ValueError(self.method, "is not attention method")
    self.hidden_size = hidden_size
    if method == 'general':
      self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
    elif method == 'concat':
      self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
      self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

  def dot_score(self, hidden, encoder_output):
    return torch.sum(hidden * encoder_output, dim=2)

  def general_score(self, hidden, encoder_output):
    energy = self.attn(encoder_output)
    return torch.sum(hidden * energy, dim=2)

  def concat_score(self, hidden, encoder_output):
    energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
    return torch.sum(self.v * energy, dim=2)

  def forward(self, hidden, encoder_outputs):
    if self.method == 'general':
      attn_energies = self.general_score(hidden, encoder_outputs)
    if self.method == 'dot':
      attn_energies = self.dot_score(hidden, encoder_outputs)
    if self.method == 'concat':
      attn_energies = self.concat_score(hidden, encoder_outputs)

    return F.softmax(attn_energies.t(), dim=1).unsqueeze(1)