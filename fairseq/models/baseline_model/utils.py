from calendar import c
from collections import Counter
from re import S
from traceback import format_tb
from typing import List
import numpy as np
import random
import copy
mask = '<unk>'
mask_probability = 0.15



def up_mask_h(emb,embed_tokens):
    size = emb.size(1)
    up_mask_h_list = []
    for index in range(emb.size(1)):
        pad = torch.zeros(emb.size(0),size - (index + 1))
        print(emb.size(0),size,size - index + 1,index)
        pad_t = pad.fill_(1).long().cuda()
        pad_t = embed_tokens(pad_t)
        temp_emb = emb[:,:index+1]
        up_mask_h_list.append(torch.cat([temp_emb,pad_t],dim = 1))
    return up_mask_h_list
        




def mask_decoder(prev_output_tokens):
        temp = copy.deepcopy(prev_output_tokens)
        for i in range(len(temp)):
            tow_dig = random.sample(set(i for i in range(temp.size(1))), 2)
            temp[i][tow_dig[0]] = 1
            temp[i][tow_dig[1]] = 1
        return temp

def tgt_mask(tgt_dict,translated,en_de):
    for i in range(len(en_de)):     
        word = tgt_dict.string(translated[i]).split()
        en_de_word = list(en_de[i].values())
        for j in range(len(word)):
            if word[j] not in en_de_word:
                translated[i][j] = 1
    return translated




def get_not_translated(multi_bert_tokenizer,tgt_dict,src_dict,translated,en_de):
    all_list = []#记录该batch的信息
    for t,ed in zip(translated,en_de):
        line_list = ''#记录这一行的信息
        str_t = src_dict.string(t)#将未翻译的token转为string
        for str_t_word in str_t.split(' '):#遍历每一个单词
            temp = multi_bert_tokenizer.tokenize(str_t_word)#这里进行分词，会将一个词转化为一个列表，里面记录所有分词信息
            for t in temp:
                if t in list(ed.keys()):#如果该分词在词表中出现了，那么就把词表中对应的词加入未翻译词表
                    line_list = line_list + ' ' + ed[t]
        all_list.append(tgt_dict.encode_line(line_list))#转化为token矩阵
    return all_list






def get_not_translated_mask(multi_bert_tokenizer,tgt_dict,src_dict,translated,en_de):
    all_list = []#记录该batch的信息
    for t,ed in zip(translated,en_de):
        line_list = ''#记录这一行的信息
        str_t = src_dict.string(t)#将未翻译的token转为string
        for str_t_word in str_t.split(' '):#遍历每一个单词
            temp = multi_bert_tokenizer.tokenize(str_t_word)#这里进行分词，会将一个词转化为一个列表，里面记录所有分词信息
            for t in temp:
                if t in list(ed.keys()):#如果该分词在词表中出现了，那么就把词表中对应的词加入未翻译词表
                    if random.random() > mask_probability:
                        line_list = line_list +' '+ ed[t]
                    else:
                        line_list = line_list +' '+ mask
        all_list.append(tgt_dict.encode_line(line_list))#转化为token矩阵
    return all_list

def get_not_translated_for_all(multi_bert_tokenizer,tgt_dict,src_dict,translated,all_obj):
    all_list = []#记录该batch的信息
    for t in translated:
        line_list = ''#记录这一行的信息
        str_t = src_dict.string(t)#将未翻译的token转为string
        for str_t_word in str_t.split(' '):#遍历每一个单词
            temp = multi_bert_tokenizer.tokenize(str_t_word)#这里进行分词，会将一个词转化为一个列表，里面记录所有分词信息
            for t in temp:
                if t in all_obj:#如果该分词在词表中出现了，那么就把词表中对应的词加入未翻译词表
                    line_list = line_list + ' ' + all_obj[t]
        all_list.append(tgt_dict.encode_line(line_list))#转化为token矩阵
    return all_list

import copy


def get_not_translated_for_all_mbert_one_hot_step(tgt_dict,translated,en_de):
    all_list = []#记录该batch的信息
    # size = max(len(v) for v in en_de)#取得最长的一段话作为pad长度   
    size = 8#转化为10个词的长度
    translated_size = translated.size(-1)
    # print(translated)
    for t,ed in zip(translated,en_de):
        line_list = []#记录这一行的信息 t * size
        t_word = tgt_dict.string(t).split()#将token转化为文字列表
        strat_mbert_token_list = list(set(ed.values()))#初始是一整个列表
        # if len(strat_mbert_token_list) > size-1:#如果超出范围
        #     strat_mbert_token_list = strat_mbert_token_list[:size-1]#截断，-1是因为在token的时候会多一个终止符

        mbert_token = ' '.join(strat_mbert_token_list)#最初始是终止符，所以要将所有信息加在最开始,111去重
        mbert_token = tgt_dict.encode_line(mbert_token)
        if len(mbert_token) > translated_size:
                mbert_token = mbert_token[:translated_size]
        line_list.append(mbert_token)
        max = 0
        for i in range(len(t_word)):#遍历句子
            mbert_token = ''#每次循环需要重置一次
            temp_t = t_word[i+1:]#获取未翻译句子
            for t_t in temp_t:#遍历未翻译单词
                # print(ed.values())
                if t_t in list(ed.values()):#如果未翻译在目标词中,视为未翻译
                    mbert_token = mbert_token + ' ' + t_t
                    # if t_t in mbert_token.split():#有重复的就不计入
                    #     continue
                    # else:
                    #
            # if len(mbert_token.split()) > size-1:#限制大小
            #     mbert_token = ' '.join(mbert_token.split()[:size-1])
            # print(mbert_token)
            mbert_token = tgt_dict.encode_line(mbert_token)#将所有未翻译信息转化为token
            # print(mbert_token)
            # print(int(mbert_token.max()))
            # if int(mbert_token.max()) >= 6000:
            #     assert 1 == 0
            if len(mbert_token) > translated_size:
                mbert_token = mbert_token[:translated_size]
            line_list.append(mbert_token)#一行的信息
        # for i in line_list:
            # print('df',len(i))
        line_list = our_collate_tokens(line_list,1,translated_size = translated_size)#补齐
        all_list.append(line_list)
    return torch.stack(all_list).cuda().long()



def get_not_translated_for_all_mbert_one_hot(tgt_dict,translated,en_de,npzfile,pad):
    all_list = []#记录该batch的信息
    # size = max(len(v) for v in en_de)#取得最长的一段话作为pad长度   
    size = 10#转化为10个词的长度
    for t,ed in zip(translated,en_de):
        line_list = []#记录这一行的信息 t * size
        mbert_token = ' '.join(list(set(ed.values())))#将所有未翻译信息转化为token
        mbert_token = tgt_dict.encode_line(mbert_token)
        # mbert_token = pad_token(mbert_token,size)#padding
        # for _ in range(1):
        for _ in range(len(t)):
            line_list.append(mbert_token)
        line_list = our_collate_tokens(line_list,1)
        all_list.append(line_list)
    return torch.stack(all_list).cuda().long()
        
        

def pad_token(token_list,size):
    for _ in range(size - len(token_list)):
        token_list.append(1)
    return token_list


def get_not_translated_for_all_mbert_matrix(tgt_dict,translated,en_de,npzfile,pad,size):
    all_list = []#记录该batch的信息
    # size = max(len(v) for v in en_de)#取得最长的一段话作为pad长度   
    for t,ed in zip(translated,en_de):
        line_list = []#记录这一行的信息 t*size*768
        str_t = tgt_dict.string(t)#将未翻译的token转为string

        line_word_list = str_t.split(' ')#每一行的每一个单词
        line_word_list.insert(0,'')
        for word_index in range(len(line_word_list)):#遍历目标语言的每一个单词
            word_step_list = []#记录每一个时间步之后出现的未翻译向量,最终进入上一层的大小应该是size*768
            has_cul = []#已经出现过的单词
            for after_step in line_word_list[word_index:]:#遍历该时间步之后的词
                if after_step in has_cul:#如果当前时间步计算未来信息时，单词已经记录，就不再重新计算
                    continue
                if after_step in list(ed.values()):#当前时间步之后出现了实体词
                    word_step_list.append(torch.from_numpy(npzfile[after_step]).cuda())#将当前时间步之后的实体添加至未翻译向量
                    has_cul.append(after_step)#记录，该单词已经出现过
            if len(word_step_list) == 0:#如果当前时间步后面没有实体词，就加一个空向量
                word_step_list.append(torch.zeros([768]).cuda())
            line_list.append(torch.stack(word_step_list))
            line_list = padding(size,line_list,pad)#按照最长的统一格式size*768
        all_list.append(torch.stack(line_list))#t*size*768
    return torch.stack(all_list)  

# def padding(size,value,pad):
#     new_value = []
#     for i in value:
#         if i.size(0) == size:
#             new_value.append(i)
#         else:
#             for j in range(size-i.size(0)):
#                 print(j)
#                 i = torch.cat([i,pad])
#             new_value.append(i)
#     return new_value


def get_all_future_info(en_de,npzfile,pad,size):#获取所有的未来信息
    all_list = []#记录该batch的信息
    for ed in en_de:
        line_list = []#记录这一行的信息
        info = list(ed.values())#取得所有未来信息
        line_list.extend([torch.from_numpy(npzfile[str_t_word]) for str_t_word in info])#将所有的未来信息加入
        if len(line_list) == 0:
            all_list.append(pad)#加入一个空向量
        else:
            all_list.append(torch.stack(line_list).cuda())#转化为词向量
    # size = max(v.size(0) for v in all_list)
    size = size
    all_list = padding(size,all_list,pad)#按照最长的统一格式
    return torch.stack(all_list)  

def get_not_translated_for_all_mbert(tgt_dict,translated,en_de,npzfile,pad):
    all_list = []#记录该batch的信息
    for t,ed in zip(translated,en_de):
        line_list = []#记录这一行的信息
        str_t = tgt_dict.string(t)#将未翻译的token转为string
        for str_t_word in str_t.split(' '):#遍历未翻译目标语言的每一个单词
            if str_t_word in list(ed.values()):#如果未翻译单词在实体中出现了
                line_list.append(torch.from_numpy(npzfile[str_t_word]))#加入信息
        if len(line_list) == 0:#如果未来没有信息
            all_list.append(pad)#加入一个空向量
        else:#如果有,就将这几个向量叠加
            all_list.append(torch.stack(line_list).cuda())#转化为词向量\
    size = max(v.size(0) for v in all_list)
    all_list = padding(size,all_list,pad)#按照最长的统一格式
    return torch.stack(all_list)  

def padding(size,value,pad):
    new_value = []
    for i in value:
        if i.size(0) == size:
            new_value.append(i)
        else:
            for j in range(size-i.size(0)):
                i = torch.cat([i,pad])
            new_value.append(i)
    return new_value


def get_alignment(de_raw_sentence: List[str], embedding_type,tokenizer,embedder,multi_bert_tokenizer):
    de_alignment = []
    id = 0
    de_raw_tokens = de_raw_sentence.split()
    for word in de_raw_tokens:
        de_alignment.extend([id for _ in range(len(get_tokenized_tokens(word, embedding_type,tokenizer,embedder,multi_bert_tokenizer)))])
        id += 1
#     print(de_alignment, len(de_alignment))
    assert id == len(de_raw_tokens), (id, len(de_raw_tokens))
    de_alignment_dict = Counter(de_alignment)
    return de_alignment, de_alignment_dict


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        # a_norm = torch.norm(a)
        # b_norm = torch.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        # a_norm = torch.norm(a, p=1)
        # b_norm = torch.norm(b, p=1)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    # similiarity = a.mul(b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist



def get_aligned_embeddings(de_alignment, de_alignment_dict, de_sentence_embed):
    id, de_emb_average, alignment_de_embeds = 0, 0., []
    
    for i, de_emb in enumerate(de_sentence_embed):
        ### apply alignment
        if de_alignment[i] == id:
            de_emb_average += de_emb
        else:
            de_emb_average /= de_alignment_dict[id]
            alignment_de_embeds.append(de_emb_average)
            id += 1
            de_emb_average = de_emb
    de_emb_average  /= de_alignment_dict[id]
    alignment_de_embeds.append(de_emb_average)
    assert len(alignment_de_embeds) == max(de_alignment)+1, (len(alignment_de_embeds), max(de_alignment)+1)
    return alignment_de_embeds
                        


# def compute_word2words_similarity(query,de_sentence_embed, sim_type=1,
#                                   norm_similarity=False, k=2, sparse_top_k=False):
#     similarity = []
#     for i, key in enumerate(de_sentence_embed):
# #         print(key)
#         if sim_type == 1: similarity.append(1-cosine_distance(query.cpu(), key.cpu()))
#         elif sim_type == 2: similarity.append(np.exp(-np.linalg.norm(query.cpu()-key.cpu(), ord=2, axis=-1)))
#         else: e
    
#     if sparse_top_k:
#         top_k_similarity = sorted(similarity, reverse=True)[k-1]
#         similarity = [i if i >= top_k_similarity else 0 for i in similarity]
#     if norm_similarity:
#         similarity = np.array(similarity) / np.array(similarity).sum()
#         similarity = similarity.tolist()
    
#     return similarity
def compute_word2words_similarity(keys, query, sim_type=1,
                                  norm_similarity=False, k=2, sparse_top_k=False):
    similarity = []
    for i, key in enumerate(keys):
        if sim_type == 1: similarity.append(1-cosine_distance(query.cpu(), key.cpu()))
        elif sim_type == 2: similarity.append(np.exp(-np.linalg.norm(query-key, ord=2, axis=-1)))
        else: e
    
    if sparse_top_k:
        top_k_similarity = sorted(similarity, reverse=True)[k-1]
        similarity = [i if i >= top_k_similarity else 0 for i in similarity]
    if norm_similarity:
        similarity = np.array(similarity) / np.array(similarity).sum()
        similarity = similarity.tolist()
    
    return similarity

def get_tokenized_tokens(string, embedding_type,tokenizer,embedder,multi_bert_tokenizer):
    if embedding_type == 1: en_temp = [string]
    elif embedding_type == 2: en_temp = tokenizer.tokenize(string)
    else: en_temp = multi_bert_tokenizer.tokenize(string)
    return en_temp



def get_embedding(string : str,tokenizer, embedder,embedding_type=1):
    if embedding_type == 1:   # seq_len * embed_dim
        return [model[w] for w in tokenizer.tokenize(string)]
    elif embedding_type == 2: # seq_len * embed_dim
        return [(embedder.encode(w, output_value = 'token_embeddings')[0]).mean(axis=0) for w in string.split()][:len(string.split())]
        return [(embedder.encode(w, output_value = 'token_embeddings')[0][1:-1]).mean(axis=0) for w in string.split()]
    elif embedding_type == 3: # seq_len * embed_dim
        res = embedder.encode(string, output_value = 'token_embeddings')[1:-1]
        return res
        
    elif embedding_type == 4: # seq_len * embed_dim
        return embedder.encode(string, output_value = 'token_embeddings').mean(axis=0)
    elif embedding_type == 5: # seq_len * embed_dim
        embs = []
        for w in string.split():
            emb, sub_ws = 0., multi_bert_tokenizer.tokenize(w)
            # print(sub_ws)
#             emb = (embedder.encode(sub_ws[0], output_value = 'token_embeddings')).mean(axis=0)
            for sub_w in sub_ws: emb += (embedder.encode(sub_w, output_value = 'token_embeddings')).mean(axis=0)
            embs.append(emb/float(len(sub_ws)))
#             embs.append(emb)
        return embs
    elif embedding_type == 6:
        embs = []
        for w in string.split():
            emb = (embedder.encode(w, output_value = 'token_embeddings')).mean(axis=0)
            embs.append(emb)
        return embs
    elif embedding_type == 7:
        embs = []
        for w in string.split():
            emb, sub_ws = 0., multi_bert_tokenizer.tokenize(w)
            # print(sub_ws)
            for sub_w in sub_ws:
                try:
                    emb += (model[sub_w])
                except:
                    emb += np.full(768,0)
                    continue
            embs.append(emb/float(len(sub_ws)))
        return embs


from typing import Optional, Tuple
import numpy as np
import torch
def our_collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
    translated_size = 10
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = translated_size
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        # print(dst.numel(),src.numel())
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res
