# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
import json

from tqdm import tqdm

from torch.utils.data import Dataset
from transformers import BertTokenizer


def build_tokenizer(fnames, max_seq_len, dat_fname):
    # print(fnames)
    # print(max_seq_len)
    # print(dat_fname)
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 7):

                # text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                text_left, _, text_right = [s.lower().strip() for s in lines[i+2].partition("$T$")]
                # aspect = lines[i+1].lower().strip()
                aspect = lines[i+3].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer,tokenizer1):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        # fin = open(fname+'.graph', 'rb')
        # idx2graph = pickle.load(fin)
        # fin.close()

        all_data = []
        for i in range(0, len(lines), 7):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 4].strip()
            select_len = int(lines[i + 6].strip())
            text = text_left + " " + aspect + " " + text_right
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            # context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            # left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            # right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_len = np.sum(left_indices != 0)
            right_len = np.sum(right_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            text_len = np.sum(text_indices != 0)
            if tokenizer1 is not None:
                lstm_text_left, _, lstm_text_right = [s.lower().strip() for s in lines[i+2].partition("$T$")]
                lstm_aspect = lines[i + 3].lower().strip()
                # lstm_text_left, _, lstm_text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                # lstm_aspect = lines[i + 1].lower().strip()
                left_with_aspect_indices = tokenizer1.text_to_sequence(lstm_text_left + " " + lstm_aspect)
                right_with_aspect_indices = tokenizer1.text_to_sequence( lstm_aspect + " " + lstm_text_right, reverse=True)
                aspect_indices = tokenizer1.text_to_sequence(lstm_aspect)
                text_indices = tokenizer1.text_to_sequence(lstm_text_left + " " + lstm_aspect + " " + lstm_text_right)
            else:
                text_indices = []
                text_indices = pad_and_truncate(text_indices, tokenizer.max_seq_len)
                left_with_aspect_indices = []
                left_with_aspect_indices = pad_and_truncate(left_with_aspect_indices, tokenizer.max_seq_len)
                right_with_aspect_indices = []
                right_with_aspect_indices = pad_and_truncate(right_with_aspect_indices, tokenizer.max_seq_len)
                aspect_indices = []
                aspect_indices = pad_and_truncate(aspect_indices, tokenizer.max_seq_len)
            # aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
            polarity = int(polarity) + 1

            concat_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            context_asp_attention_mask = [0] + [1] * (text_len)
            special_mask = [1] + [0]*left_len + [1]*aspect_len + [0]*right_len+ [1]*(aspect_len+2)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)
            context_asp_attention_mask = pad_and_truncate(context_asp_attention_mask, tokenizer.max_seq_len)
            special_mask = pad_and_truncate(special_mask, tokenizer.max_seq_len)

            text_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            # dependency_graph = np.pad(idx2graph[i], \
                # ((0,tokenizer.max_seq_len-idx2graph[i].shape[0]),(0,tokenizer.max_seq_len-idx2graph[i].shape[0])), 'constant')

            data = {
                'concat_bert_indices': concat_bert_indices, #
                'concat_segments_indices': concat_segments_indices, #
                'attention_mask': context_asp_attention_mask, #
                'text_bert_indices': text_bert_indices, #
                'aspect_bert_indices': aspect_bert_indices, #
                'text_indices': text_indices, #
                'special_mask':special_mask,
                # 'context_indices': context_indices,
                # 'left_indices': left_indices,
                'left_with_aspect_indices': left_with_aspect_indices,#
                # 'right_indices': right_indices,
                'right_with_aspect_indices': right_with_aspect_indices, #
                'aspect_indices': aspect_indices, #
                # 'aspect_boundary': aspect_boundary,
                # 'dependency_graph': dependency_graph,
                'text': text,
                'polarity': polarity,
                'select_len': select_len,
                'term': aspect,
            }
            all_data.append(data)
        #     print('aaaaaaaaaaa')
        #     print(text)
        #     print(concat_bert_indices)
        #     print(concat_segments_indices)
        #     print(context_asp_attention_mask)
        #     print(text_bert_indices)
        #     print(aspect_bert_indices)
        #     # print(lstm_text_left)
        #     # print(lstm_aspect)
        #     # print(lstm_text_right)
        #     print(text_indices)
        #     print(left_with_aspect_indices)
        #     print(right_with_aspect_indices)
        #     print(aspect_indices)
        #     if i>8:
        #         break
        # print(all_data)
        # a = hh
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                select_len = d['select_len']
                text_list = list(d['token'])
                tok = list(d['token'])       # word token
                length = len(tok)            # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])   # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']   # label
                pos = list(d['pos'])         # pos_tag 
                head = list(d['head'])       # head
                deprel = list(d['deprel'])   # deprel
                short = list(d['short'])
                # position
                aspect_post = [aspect['from'], aspect['to']] 
                post = [i-aspect['from'] for i in range(aspect['from'])] \
                       +[0 for _ in range(aspect['from'], aspect['to'])] \
                       +[i-aspect['to']+1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]   
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]
                
                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,\
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list,'short':short, 'select_len':select_len}
                all_data.append(sample)

    return all_data


class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, tokenizer, opt, vocab_help):

        parse = ParseData
        post_vocab, pos_vocab, dep_vocab, pol_vocab = vocab_help
        data = list()
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            text = tokenizer.text_to_sequence(obj['text'])
            aspect = tokenizer.text_to_sequence(obj['aspect'])  # max_length=10
            post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]
            post = tokenizer.pad_sequence(post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]
            pos = tokenizer.pad_sequence(pos, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]
            deprel = tokenizer.pad_sequence(deprel, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            mask = tokenizer.pad_sequence(obj['mask'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            
            
            # left_len = np.sum(left_indices != 0)
            # aspect_len = np.sum(aspect_indices != 0)
            # aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
            

            adj = np.ones(opt.max_length) * opt.pad_id
            if opt.parseadj:
                from absa_parser import headparser
                # * adj
                headp, syntree = headparser.parse_heads(obj['text'])
                adj = softmax(headp[0])
                adj = np.delete(adj, 0, axis=0)
                adj = np.delete(adj, 0, axis=1)
                adj -= np.diag(np.diag(adj))
                if not opt.direct:
                    adj = adj + adj.T
                adj = adj + np.eye(adj.shape[0])
                adj = np.pad(adj, (0, opt.max_length - adj.shape[0]), 'constant')
            
            if opt.parsehead:
                from absa_parser import headparser
                headp, syntree = headparser.parse_heads(obj['text'])
                syntree2head = [[leaf.father for leaf in tree.leaves()] for tree in syntree]
                head = tokenizer.pad_sequence(syntree2head[0], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            else:
                head = tokenizer.pad_sequence(obj['head'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post', truncating='post')
            length = obj['length']
            polarity = polarity_dict[obj['label']]
            # short 根据 obj['short']
            mask_0 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            mask_1 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            mask_2 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            mask_3 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            mask_4 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            mask_5 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            short_length = len(obj['short'])
            assert len(obj['short']) == len(obj['short'][0])
            for i in range(short_length):
                for j in range(short_length):
                    mask_0[i][j] = 0
                    if obj['short'][i][j] == 1:
                        mask_1[i][j] = 0 
                        mask_2[i][j] = 0
                        mask_3[i][j] = 0
                        mask_4[i][j] = 0
                        mask_5[i][j] = 0
                    elif obj['short'][i][j] == 2:
                        mask_2[i][j] = 0
                        mask_3[i][j] = 0
                        mask_4[i][j] = 0
                        mask_5[i][j] = 0
                    elif obj['short'][i][j] == 3:
                        mask_3[i][j] = 0
                        mask_4[i][j] = 0
                        mask_5[i][j] = 0
                    elif obj['short'][i][j] == 4:
                        mask_4[i][j] = 0
                        mask_5[i][j] = 0
                    elif obj['short'][i][j] == 5:
                        mask_5[i][j] = 0    

            for i in range(short_length):
                mask_1[i][i] = 0 
                mask_2[i][i] = 0
                mask_3[i][i] = 0
                mask_4[i][i] = 0
                mask_5[i][i] = 0
            

            short_mask = np.asarray([mask_0, mask_1, mask_2, mask_3, mask_4], dtype='float32')

            data.append({
                'text': text, 
                'aspect': aspect, 
                'post': post,
                'pos': pos,
                'deprel': deprel,
                'head': head,
                'adj': adj,
                'mask': mask,
                'length': length,
                'polarity': polarity,
                'short_mask': short_mask,
            })

        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)


class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            select_len = obj['select_len']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end: ]

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)                   # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)          # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)  
            offset = len(left) 
            for ori_i, w in enumerate(term):        
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term) 
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i+offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len-2*len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()
                    
            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens)
            tok_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')


            context_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(
                bert_tokens)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(term_tokens)+[tokenizer.sep_token_id]
            context_asp_len = len(context_asp_ids)
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)
            context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            # context_asp_attention_mask = [1] * context_asp_len + paddings
            special_mask = [1]+ [0]*len(left_tokens) + [1]*len(term_tokens) + [0]*len(right_tokens)+ [1]*(asp_end - asp_start+2)
            special_mask = special_mask + (opt.max_length - len(special_mask)) * [0]
            context_asp_ids += paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            # context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            special_mask = np.asarray(special_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')

            row_short = obj['short']
            
            for i in range(context_len-1):
                if tok2ori_map[i+1] == tok2ori_map[i]:
                    a = row_short[i]
                    row_short = np.insert(row_short, i, values=a, axis=0)


            column_short = row_short
            for j in range(context_len-1):
                if tok2ori_map[j+1] == tok2ori_map[j]:
                    a = column_short[:,j]
                    column_short = np.insert(column_short, j, values=a, axis=1)      


            mask_0 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            mask_1 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            mask_2 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            mask_3 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            mask_4 = [[-99999] * opt.max_length for _ in range(opt.max_length)]
            short_length = len(obj['short'])
            assert len(obj['short']) == len(obj['short'][0])
            for i in range(1):
                for j in range(context_len):
                    mask_0[i][j] = 0
                    mask_1[i][j] = 0
                    mask_2[i][j] = 0
                    mask_3[i][j] = 0
                    mask_4[i][j] = 0
            for i in range(context_len):
                for j in range(context_len):
                    mask_0[i+1][j+1] = 0
                    if column_short[i][j] == 1:
                        mask_1[i+1][j+1] = 0
                        mask_2[i+1][j+1] = 0
                        mask_3[i+1][j+1] = 0
                        mask_4[i+1][j+1] = 0
                    elif column_short[i][j] == 2:
                        mask_2[i+1][j+1] = 0
                        mask_3[i+1][j+1] = 0
                        mask_4[i+1][j+1] = 0
                    elif column_short[i][j] == 3:
                        mask_3[i+1][j+1] = 0
                        mask_4[i+1][j+1] = 0
                    elif column_short[i][j] == 4:
                        mask_4[i+1][j+1] = 0
            short_mask = np.asarray([mask_0, mask_1, mask_2, mask_3, mask_4], dtype='float32')


            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': src_mask,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
                'short_mask': short_mask,
                'special_mask':special_mask,
                'text': text,
                'term': ' '.join(term),
                'select_len': select_len
            }
            # if 'touchscreen functions'in text:
            #     print(data)
            #     a = hh
            self.data.append(data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# The tech guy then said the $T$ does not do 1-to-1 exchange and I have to direct my concern to the `` sales '' team , which is the retail shop which I bought my netbook from .
# service center
# -1
# 3
# 13

# the tech guy then said the service center does not do 1-to-1 exchange and i have to direct my concern to the `` sales '' team , which is the retail shop which i bought my netbook from .

# 101 1996 6627 3124 2059 2056 1996 2326 2415 2515 2025 2079 1015 1011
#  2000 1011 1015 3863 1998 1045 2031 2000 3622 2026 5142 2000 1996 1036
#  1036 4341 1005 1005 2136 1010 2029 2003 1996 7027 4497 2029 1045 4149
#  2026 5658 8654 2013 1012  102 2326 2415  102 

#  9 19 20 21 22  9 23 24

# 9   19   20   21   22    9   23   24   25   26   27 3500   31    6
#     1   32   30   33   34   35   30    9 3500   37 3500   39   40   41
#    42    9   43   44   41    1   45   34 3500   48   18 

# 23 24