import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math, copy
import numpy as np
from basemodel_1D import TemporalConvNet

from config import ACTIVATION_FUNCTIONS


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_mask_bidirectional(size, atten_len_a, atten_len_b):
    attn_shape = (1, size, size)
    past_all_mask = np.triu(np.ones(attn_shape), k=atten_len_b).astype('uint8')
    past_all_mask = torch.from_numpy(past_all_mask)
    past_all_mask = past_all_mask == 0
    past_all_mask = past_all_mask.byte()
    no_need_mask = np.triu(np.ones(attn_shape), k=-atten_len_a + 1).astype('uint8')
    no_need_mask = torch.from_numpy(no_need_mask)
    gene_mask = no_need_mask * past_all_mask

    return gene_mask.to(device)



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)).to(device)
        self.b_2 = nn.Parameter(torch.zeros(features)).to(device)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = layer
        self.norm = LayerNorm(layer[0].size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class MultiModalEncoder(nn.Module):

    def __init__(self, layer, N, modal_num):
        super(MultiModalEncoder, self).__init__()
        self.modal_num = modal_num
        self.layers = layer
        self.norm = nn.ModuleList()
        for i in range(self.modal_num):
            self.norm.append(LayerNorm(layer[0].size))


    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        _x = torch.chunk(x, self.modal_num, dim=-1)
        _x_list = []
        for i in range(self.modal_num):
            _x_list.append(self.norm[i](_x[i]))

        x = torch.cat(_x_list, dim=-1)

        return x


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))


class MultiModalSublayerConnection(nn.Module):

    def __init__(self, size, modal_num, dropout):
        super(MultiModalSublayerConnection, self).__init__()
        self.modal_num = modal_num

        self.norm = nn.ModuleList()
        for i in range(self.modal_num):
            self.norm.append(LayerNorm(size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        residual = x

        _x_list = []
        _x = torch.chunk(x, self.modal_num, -1)
        for i in range(self.modal_num):
            _x_list.append(self.norm[i](_x[i]))
        x = torch.cat(_x_list, dim=-1)

        return self.dropout(sublayer(x)) + residual


class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList()
        self.sublayer.append(SublayerConnection(size, dropout))
        self.sublayer.append(SublayerConnection(size, dropout))

        self.size = size

    def forward(self, x, mask):

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiModalEncoderLayer(nn.Module):

    def __init__(self, size, modal_num, mm_atten, mt_atten, feed_forward, dropout):
        super(MultiModalEncoderLayer, self).__init__()
        self.modal_num = modal_num

        self.mm_atten = mm_atten
        self.mt_atten = mt_atten
        self.feed_forward = feed_forward

        mm_sublayer = MultiModalSublayerConnection(size, modal_num, dropout)
        mt_sublayer = nn.ModuleList()
        for i in range(modal_num):
            mt_sublayer.append(SublayerConnection(size, dropout))
        ff_sublayer = nn.ModuleList()
        for i in range(modal_num):
            ff_sublayer.append(SublayerConnection(size, dropout))

        self.sublayer = nn.ModuleList()
        self.sublayer.append(mm_sublayer)
        self.sublayer.append(mt_sublayer)
        self.sublayer.append(ff_sublayer)

        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.mm_atten(x, x, x))

        _x = torch.chunk(x, self.modal_num, dim=-1)
        _x_list = []
        for i in range(self.modal_num):
            # feature = self.sublayer[1][i](_x[i], lambda x: self.mt_atten[i](x, x, x, mask[i]))
            feature = self.sublayer[1][i](_x[i], lambda x: self.mt_atten[i](x, x, x, mask=None))
            feature = self.sublayer[2][i](feature, self.feed_forward[i])
            _x_list.append(feature)
        x = torch.cat(_x_list, dim=-1)

        return x


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):

        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiModalAttention(nn.Module):
    def __init__(self, h, d_model, modal_num, dropout=0.1):

        super(MultiModalAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.modal_num = modal_num
        self.mm_linears = nn.ModuleList()
        for i in range(self.modal_num):
            linears = clones(nn.Linear(d_model, d_model), 4)
            self.mm_linears.append(linears)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        query = torch.chunk(query, self.modal_num, dim=-1)
        key   = torch.chunk(key, self.modal_num, dim=-1)
        value = torch.chunk(value, self.modal_num, dim=-1)

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query[0].size(0)

        _query_list = []
        _key_list = []
        _value_list = []
        for i in range(self.modal_num):
            _query_list.append(self.mm_linears[i][0](query[i]).view(nbatches, -1, self.h, self.d_k))
            _key_list.append(self.mm_linears[i][1](key[i]).view(nbatches, -1, self.h, self.d_k))
            _value_list.append(self.mm_linears[i][2](value[i]).view(nbatches, -1, self.h, self.d_k))

        mm_query = torch.stack(_query_list, dim=-2)
        mm_key = torch.stack(_key_list, dim=-2)
        mm_value = torch.stack(_value_list, dim=-2)
        x, _ = attention(mm_query, mm_key, mm_value, mask=mask, dropout=self.dropout)

        x = x.transpose(-2, -3).contiguous().view(nbatches, -1, self.modal_num, self.h * self.d_k)
        _x = torch.chunk(x, self.modal_num, dim=-2)
        _x_list = []
        for i in range(self.modal_num):
            _x_list.append(self.mm_linears[i][-1](_x[i].squeeze()))
        x = torch.cat(_x_list, dim=-1)

        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SEmbeddings(nn.Module):
    def __init__(self, d_model, dim):
        super(SEmbeddings, self).__init__()
        self.lut = nn.Linear(dim, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.lut(x)
        x = x * math.sqrt(self.d_model)
        return x


class TEmbeddings(nn.Module):
    def __init__(self, opts, dim):
        super(TEmbeddings, self).__init__()
        self.levels = opts.levels
        self.ksize = opts.ksize
        self.d_model = opts.d_model
        self.dropout = opts.dropout

        self.channel_sizes = [self.d_model] * self.levels
        self.lut = TemporalConvNet(dim, self.channel_sizes, kernel_size=self.ksize, dropout=self.dropout)

    def forward(self, x):
        x = self.lut(x.transpose(1, 2)).transpose(1, 2) * math.sqrt(self.d_model)
        # x = torch.transpose(self.lut(torch.transpose(x, 1, 2)), 1, 2) * math.sqrt(self.d_model)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        v = torch.arange(0, d_model, 2).type(torch.float)
        v = v * -(math.log(1000.0) / d_model)
        div_term = torch.exp(v)
        pe[:, 0::2] = torch.sin(position.type(torch.float) * div_term)
        pe[:, 1::2] = torch.cos(position.type(torch.float) * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class ProcessInput(nn.Module):
    def __init__(self, opts, dim):
        super(ProcessInput, self).__init__()

        if opts.embed == 'spatial':
            self.Embeddings = SEmbeddings(opts.d_model, dim)
        elif opts.embed == 'temporal':
            self.Embeddings = TEmbeddings(opts, dim)
        self.PositionEncoding = PositionalEncoding(opts.d_model, opts.dropout_position, max_len=5000)

    def forward(self, x):
        return self.PositionEncoding(self.Embeddings(x))


class TE(nn.Module):

    def __init__(self, opts, num_features):
        super(TE, self).__init__()

        self.modal_num = opts.modal_num
        assert self.modal_num == 1, 'TE model is only used for single feature streams ...'

        # self.mask_a_length = int(opts.mask_a_length)
        # self.mask_b_length = int(opts.mask_b_length)

        self.N = opts.block_num
        self.dropout = opts.dropout
        self.h = opts.h
        self.d_model = opts.d_model
        self.d_ff = opts.d_ff

        self.input = ProcessInput(opts, num_features)

        self.regress = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, opts.ntarget)
        )
        self.dropout_embed = nn.Dropout(p=opts.dropout_embed)

        encoder_layer = nn.ModuleList()
        for i in range(self.N):
            atten = MultiHeadedAttention(self.h, self.d_model, self.dropout)
            ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
            encoder_layer.append(EncoderLayer(self.d_model, atten, ff, self.dropout))
        self.te = Encoder(encoder_layer, self.N)

        for p in self.te.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.input.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.regress.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input(x)
        x = self.dropout_embed(x)

        # mask = generate_mask_bidirectional(x.shape[1], self.mask_a_length, self.mask_b_length)
        x = self.te(x, mask=None)

        return self.regress(x)


class JTMA(nn.Module):

    def __init__(self, opts, num_features, len_feature):
        super(JTMA, self).__init__()

        self.modal_num = len(num_features)

        # self.mask_a_length = [int(l) for l in opts.mask_a_length.split(',')]
        # self.mask_b_length = [int(l) for l in opts.mask_b_length.split(',')]

        self.num_features = num_features    # features 各自的维度
        self.N = opts.block_num
        self.dropout_mmatten = opts.dropout_mmatten
        self.dropout_mtatten = opts.dropout_mtatten
        self.dropout_ff = opts.dropout_ff
        self.dropout_subconnect = opts.dropout_subconnect
        self.h = opts.h
        self.h_mma = opts.h_mma
        self.d_model = opts.d_model
        self.d_ff = opts.d_ff

        self.input = nn.ModuleList()
        for i in range(self.modal_num):
            self.input.append(ProcessInput(opts, num_features[i]))
        self.dropout_embed = nn.Dropout(p=opts.dropout_embed)

        multimodal_encoder_layer = nn.ModuleList()
        for i in range(self.N):
            mm_atten = MultiModalAttention(self.h_mma, self.d_model, self.modal_num, self.dropout_mmatten)
            mt_atten = nn.ModuleList()
            ff = nn.ModuleList()
            for j in range(self.modal_num):
                mt_atten.append(MultiHeadedAttention(self.h, self.d_model, self.dropout_mtatten))
                ff.append(PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout_ff))
            multimodal_encoder_layer.append(MultiModalEncoderLayer(self.d_model, self.modal_num, mm_atten, mt_atten, ff, self.dropout_subconnect))

        self.temma = MultiModalEncoder(multimodal_encoder_layer, self.N, self.modal_num)
        self.regress = nn.Sequential(
            nn.Linear(self.d_model * self.modal_num * len_feature, self.d_model * self.modal_num // 2),
            nn.ReLU(),
            nn.Linear(self.d_model * self.modal_num // 2, opts.ntarget),
        )
        self.final_activation = ACTIVATION_FUNCTIONS[opts.task]()


        for p in self.temma.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.input.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.regress.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # _x = torch.chunk(x, self.modal_num, dim=-1)
        _x_list = []
        for i in range(self.modal_num):
            _x_list.append(self.input[i](x[i]))
        x = torch.cat(_x_list, dim=-1)

        x = self.dropout_embed(x)

        # mask = []
        # for i in range(self.modal_num):
        #     mask.append(generate_mask_bidirectional(x.shape[1], self.mask_a_length[i], self.mask_b_length[i]))
        x = self.temma(x, mask=None)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        y = self.regress(x)
        return self.final_activation(y), y

class JTMA_Ensemble(nn.Module):
    def __init__(self,opts,models,model_num) :
        super(JTMA_Ensemble,self).__init__()
        # 
        # self.modal_num = opts.modal_num
        # self.d_model = opts.d_model
        
        self.final_activation = ACTIVATION_FUNCTIONS[opts.task]()

        self.regress=nn.Sequential(
            nn.Linear(in_features=model_num,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=1)
        )

        self.models=models
        self.model_num=model_num
        
    def forward(self,x):
        with torch.no_grad():
            preds=[]
            for i in range(self.model_num):
                model=torch.load(self.models[i], map_location=device)
                pred=model(x)
                preds.append(pred)

        emsenble_pred=torch.cat((pred for pred in preds),dim=1)  # ori:preds[0-4]
        #emsenble_pred=torch.Tensor(preds)
        emsenble_input=self.regress(emsenble_pred)

        return self.final_activation(emsenble_input), emsenble_input
    

# 
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, n_to_1=False):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)
        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.rnn(x_packed)

        if self.n_to_1:
            # hiddenstates, h_n, only last layer
            return last_item_from_packed(rnn_enc[0], x_len)
            #batch_size = x.shape[0]
            #h_n = h_n.view(self.n_layers, self.n_directions, batch_size, self.d_out) # (NL, ND, BS, dim)
            #last_layer = h_n[-1].permute(1,0,2) # (BS, ND, dim)
            #x_out = last_layer.reshape(batch_size, self.n_directions * self.d_out) # (BS, ND*dim)

        else:
            x_out = rnn_enc[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out

#https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/7
def last_item_from_packed(packed, lengths):
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    )).to(device)
    sorted_lengths = lengths[packed.sorted_indices].to(device)
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0)).to(device)
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items


class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y

class SubModel(nn.Module):
    def __init__(self, params, id, feature_out = False):
        super(SubModel, self).__init__()
        self.params = params

        self.inp = nn.Linear(getattr(params,f"d_in_{id}"), getattr(params,f"model_dim_{id}"), bias=False)
        # self.atten = nn.MultiheadAttention(getattr(params,f"model_dim_{id}"),4,batch_first=True)
        # self.atten = nn.MultiheadAttention(getattr(params,f"d_in_{id}"),4,batch_first=True)

        self.encoder = RNN(getattr(params,f"model_dim_{id}"), getattr(params,f"model_dim_{id}"), n_layers=getattr(params,f"rnn_n_layers_{id}"), bi=getattr(params,f"rnn_bi_{id}"),
                           dropout=getattr(params,f"rnn_dropout_{id}"), n_to_1=params.n_to_1)

        self.feature_out = feature_out
        if not feature_out:
            d_rnn_out = getattr(params, f"model_dim_{id}") * 2 if getattr(params, f"rnn_bi_{id}") and getattr(params,
                                                                                                              f"rnn_n_layers_{id}") > 0 else getattr(
                params, f"model_dim_{id}")

            self.out = OutLayer(d_rnn_out, getattr(params,f"d_fc_out_{id}"), params.n_targets, dropout=getattr(params,f"linear_dropout_{id}"))
            self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = self.inp(x)
        # padding_mask = self.generate_sent_masks(x.shape[0], x.shape[1], x_len).cuda() == 0
        # x = self.atten(x, x, x, key_padding_mask=padding_mask)[0]
        x = self.encoder(x, x_len)
        if self.feature_out:
            return x
        y = self.out(x)
        activation = self.final_activation(y)
        return activation

    def generate_sent_masks(self, batch_size, max_seq_length, source_lengths):
        enc_masks = torch.zeros(batch_size, max_seq_length, dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, :src_len] = 1
        return enc_masks

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()

        self.submodels = nn.ModuleList()
        self.dense = nn.ModuleList()
        self.num_feature = len(params.feature)    # len(args.feature)=3
        fusion_size = getattr(params, f"fusion_size")    # 64
        for id in range(self.num_feature):
            self.submodels.append(SubModel(params, id, feature_out=True))
            self.dense.append(nn.Linear(getattr(params,f"model_dim_{id}")*2 if getattr(params, f"rnn_bi_{id}") else getattr(params,f"model_dim_{id}"),
                                        fusion_size
                                        ))

        self.k_project = nn.ModuleList()
        self.k_project = nn.ModuleList()


        self.out = OutLayer( fusion_size * (self.num_feature + self.num_feature * (self.num_feature - 1)) ,d_hidden=params.fusion_hidden,d_out=1,dropout=params.fusion_fc_drop)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()    # sigmoid.
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, x_len):
        sub_pred = []
        for index, feat in enumerate(x):
            sub_pred.append(self.dense[index](self.submodels[index](feat, x_len[index])))

        for i in range(self.num_feature):
            for j in range(self.num_feature):
                if i == j:
                    continue
                atten1 = self.softmax(torch.mm(sub_pred[i],sub_pred[j].T)/math.sqrt(sub_pred[0].shape[-1]))
                f1 = torch.mm(atten1,sub_pred[j])
                sub_pred.append(f1)

        feature = torch.cat(sub_pred, dim=-1)
        y = self.out(feature)
        # y = self.final_activation(y)
        return self.final_activation(y), y