# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from asyncio.log import logger
import math
from typing import Dict, List, Optional
import pickle
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    MultiheadAttention,
)
from fairseq.modules.quant_noise import quant_noise
from fairseq.models.generator_model import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor, dropout
from fairseq.models.transformer import (
    TransformerConfig,
)
from fairseq.models.generator_model.utils import SublayerConnectionv2,PositionwiseFeedForward, deal_graph


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name


class TransformerEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = cfg.encoder.layerdrop
        self.return_fc = return_fc

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = cfg.max_source_positions

        self.embed_tokens = embed_tokens
        # self.img_linear = nn.Linear(2048,embed_dim)
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)
        # self.boxprobs = pickle.load(open('/home/sxy/Projects/CP/work/bpe-data/boxporbs.pkl', 'rb'))
        self.trans_obj = nn.Sequential(nn.Linear(2048, embed_dim), nn.ReLU(), nn.Dropout(cfg.dropout),
                                    nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(cfg.dropout))
        # self.norm = nn.LayerNorm(embed_dim)
        # self.trans_obj = nn.Sequential(nn.Linear(2048, embed_dim), nn.ReLU(), nn.Dropout(cfg.dropout))
        #10 * 2048 -》10 * 256 
        # self.boxfeats = pickle.load(open('/home/sxy/Projects/CP/work/bpe-data/{}.res.pkl'.format('train'), 'rb'))
        # self.valid_boxfeats = pickle.load(open('/home/sxy/Projects/CP/work/bpe-data/{}.res.pkl'.format('valid'), 'rb'))
        # self.test_boxfeats = pickle.load(open('/home/sxy/Projects/CP/work/bpe-data/{}.res.pkl'.format('test'), 'rb'))
        # self.boxfeats.update(self.valid_boxfeats)
        # self.boxfeats.update(self.test_boxfeats)
        # self.trans_obj = nn.Linear(2048,embed_dim)
        # self.img_attn = self.build_self_attention(embed_dim, cfg)
        # self.x_attn = self.build_self_attention(embed_dim,cfg)
        # self.ffn_x = PositionwiseFeedForward(embed_dim,2048,cfg.dropout)
        # self.ffn_img = PositionwiseFeedForward(embed_dim,2048,cfg.dropout)
        # self.res_ffn_x = SublayerConnectionv2(embed_dim,cfg.dropout)
        # self.res_ffn_img = SublayerConnectionv2(embed_dim,cfg.dropout)
        # self.res_x = SublayerConnectionv2(embed_dim,cfg.dropout)
        # self.res_img = SublayerConnectionv2(embed_dim,cfg.dropout)
        # self.x_gate = SublayerConnectionv2(embed_dim,cfg.dropout)
        # self.img_gate = SublayerConnectionv2(embed_dim,cfg.dropout)
        # self.x2o_linear = nn.Linear(2*embed_dim,embed_dim)
        # self.o2x_linear = nn.Linear(2*embed_dim,embed_dim)
        # self.final_layer_norm = LayerNorm(embed_dim, export=cfg.export)
        # self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)

        
        # self.boxprobs = pickle.load(open('/home/sxy/Projects/CP/work/bpe-data/boxporbs.pkl', 'rb'))
        # self.allboxfeats = pickle.load(open('/home/sxy/Projects/CP/work/bpe-data/train.res.pkl', 'rb'))
        # self.G = snap.TFIn("/home/sxy/Projects/CP/work/bpe-data/train.bpe.graph")


        # self.dropout_module = FairseqDropout(
        #     cfg.dropout, module_name=self.__class__.__name__
        # )
        # self.activation_dropout_module = FairseqDropout(
        #     float(cfg.activation_dropout), module_name=self.__class__.__name__
        # )

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
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
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None
    
    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=cfg.quant_noise.pq,
            qn_block_size=cfg.quant_noise.pq_block_size,
        )

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TranformerEncoderLayerFusion(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        img_global = None,
        graph = None,
        matrix = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings,img_global,graph = graph,matrix = matrix
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        img_global = None,
        graph = None,
        matrix = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        # topk = 5
        # regions_num = 10
        # thre = 0.0

        # obj_feat = src_tokens.new_zeros(src_tokens.size(0), regions_num, topk, 2048).float()
        # print('obj',obj_feat.shape)
        # matrix = src_tokens.new_zeros(src_tokens.size(0), src_tokens.size(1), regions_num*topk).float()
        # print('matrix',matrix.shape)
        # print(self.G)
        # assert 1 == 0
        # for ib, img in enumerate(img_global):
        #     # phrase_num, 5, 2048 (numpy)
        #     boxfeat = torch.tensor(self.allboxfeats[img]).reshape(-1, 5, 2048)
            
        #     # phrase_num * 5
        #     img_boxprobs = torch.tensor(self.boxprobs[img])
            
        #     ge_thre = (img_boxprobs >= thre).byte()
        #     # keep top 1
        #     ge_thre[list(range(0, ge_thre.size(0), 5))] = 1
        #     obj_feat[ib, :boxfeat.size(0)] = boxfeat[:, :topk]

        #     for item in aligns[ib]:
        #         ## item: text_word_id, object_id
        #         objixs = src_tokens.new_tensor([n+item[1] * topk for n in range(topk)])
        #         matrix[ib, item[0], objixs] = ge_thre[objixs].float().cuda()
        # assert 1 == 0
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # topk = 5
        # objdim = 2048
        # thre = 0.0
        # # img_global = self.img_linear(img_global)#变换图形形状
        # imgs, aligns, regions_num = deal_graph(graph)
        #         # B Tobj
        # obj_feat = src_tokens.new_zeros(src_tokens.size(0), max(regions_num), topk, objdim).float()
        # # B Tx Tobj*topk
        # matrix = src_tokens.new_zeros(src_tokens.size(0), src_tokens.size(1), max(regions_num)*topk).float()
        # for ib, img in enumerate(imgs):
        #     # phrase_num, 5, 2048 (numpy)
        #     boxfeat = torch.tensor(self.boxfeats[img]).reshape(-1, 5, objdim)
        #     # phrase_num * 5
        #     img_boxprobs = torch.tensor(self.boxprobs[img])
        #     ge_thre = (img_boxprobs >= thre).byte()
        #     # keep top 1
        #     ge_thre[list(range(0, ge_thre.size(0), 5))] = 1
        #     obj_feat[ib, :boxfeat.size(0)] = boxfeat[:, :topk]
        #     for item in aligns[ib]:
        #         ## item: text_word_id, object_id
        #         objixs = src_tokens.new_tensor([n+item[1] * topk for n in range(topk)])
        #         matrix[ib, item[0], objixs] = ge_thre[objixs].float().cuda()

        # # batch_size, objectnum, objdim
        # obj_feat = obj_feat.view(src_tokens.size(0), -1, objdim)

        img_global = self.trans_obj(img_global)#2048 -> 256
        matrix_oj = matrix.transpose(1,2)
        d,e = matrix_oj.max(dim = -1)
        img_global = img_global * d.unsqueeze(-1)


        matrix = matrix[:,:x.size(-2)]
        matrix = matrix.unsqueeze(-1)#b * t * o
        matrix4oobj = matrix.transpose(1,2)# b * o * t
        batch, objn, xn = matrix4oobj.size(0), matrix4oobj.size(1), matrix4oobj.size(2)
        # B x T x C -> T x B x C
        # print('matrix',matrix.shape)
        # 
        # print('maxt',matrix.shape)
        # print('un',matrix.unsqueeze(-1).shape)
        # assert 1 == 0
        img_global = img_global.transpose(0, 1)
        x = x.transpose(0, 1)

        # img_global = self.norm(img_global)

        # out = torch.cat([x,img_global],dim = 0)

        # out = out.transpose(0,2)

        # out = nn.Linear(out.size(-1),x.size(0)).cuda()(out)
        # out = out.transpose(0,2)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)
        o = img_global
        # encoder layers
        for id ,layer in enumerate(self.layers):
            lr,o = layer(
            x,
            o,
            matrix,
            matrix4oobj,
            batch,
            objn,
            xn,
            encoder_padding_mask=encoder_padding_mask if has_pads else None
            )  

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        


        # query_x_img = torch.cat([x,img_global])

        
        
        # residual = x
        # oov_enc, oov_attn = self.img_attn(
        #         query=img_global,
        #         key=img_global,
        #         value=img_global,
        #     )
        # oov_enc = self.res_img(img_global,oov_enc)

        # x_enc,x_attn = self.img_attn(
        #         query=x,
        #         key=x,
        #         value=x,
        #     )
        # x_enc = self.res_x(x,x_enc)

        # # oov_enc, oov_attn = self.img_attn(
        # #         query=query_x_img,
        # #         key=x,
        # #         value=x,
        # #     )
        # # oov_enc = oov_enc.transpose(0,2)
        # # oov_enc = nn.Linear(oov_enc.size(-1),x.size(0)).cuda()(oov_enc)
        # # oov_enc = oov_enc.transpose(0,2)

        # # residual = oov_enc
        # # oov_enc = self.activation_fn(self.fc1(oov_enc))
        # # oov_enc = self.activation_dropout_module(oov_enc)
        # # oov_enc = self.fc2(oov_enc)
        # # oov_enc = self.dropout_module(oov_enc)
        # # oov_enc = self.residual_connection(oov_enc, residual)
        # # oov_enc = self.final_layer_norm(oov_enc)

        # x_enc = x_enc.transpose(0,1)
        # oov_enc = oov_enc.transpose(0,1)

        # # text to image gate
        # new_x_ep = x_enc.unsqueeze(2).expand(batch, xn, objn, x_enc.size(-1))
        # o_ep = oov_enc.unsqueeze(1).expand(batch, xn, objn, img_global.size(-1))
        # # b * x * o * t
        # x_o_gate = torch.sigmoid(self.x2o_linear(torch.cat([new_x_ep,o_ep],-1)))
        # # b * x * d
        # x_o = (x_o_gate * matrix * o_ep).sum(2)


        # # image to text gate
        # x_ep = x_enc.unsqueeze(1).expand(batch, objn, xn, x_enc.size(-1))
        # new_o_ep = oov_enc.unsqueeze(2).expand(batch, objn, xn, img_global.size(-1))
        # #b * o * x * t
        # o_x_gate = torch.sigmoid(self.o2x_linear(torch.cat([x_ep,new_o_ep],-1)))
        # #b * o * d
        # o_x = (o_x_gate * matrix4oobj * x_ep).sum(2)
        
        # x_enc = self.x_gate(x_enc,x_o)
        # oov_enc = self.img_gate(oov_enc,o_x)
        

        # final_x = self.res_ffn_x(x_enc,self.ffn_x(x_enc))
        # final_o = self.res_ffn_img(oov_enc,self.ffn_img(oov_enc))




        # print(oov_enc.shape)
        # assert 1 == 0


        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
            "img_global": img_global.transpose(0,1),
        }
    
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

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        # if len(encoder_out["final_x"]) == 0:
        #     new_final_x = []
        # else:
        #     new_final_x = [
        #         encoder_out["final_x"][0].index_select(0, new_order)
        #     ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]




        # if len(encoder_out["oov_enc"]) == 0:
        #     new_oov_enc = []
        # else:
        #     new_oov_enc = [encoder_out["oov_enc"][0].index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)
        # logger.info('------------------------------')
        # logger.info(new_oov_enc[0].shape)
        # logger.info(new_encoder_embedding[0].shape)
        # logger.info(new_encoder_out[0].shape)
        # logger.info(new_final_x[0].shape)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1

        }

            # "oov_enc": new_oov_enc,
            # "final_x": new_final_x,
            
    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerEncoder(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )
