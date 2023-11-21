# -*- coding:utf-8 -*-
"""
Author:
    Yuef Zhang
Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""
import torch

from .basemodel import BaseModel
from ..inputs import *
from ..layers import *
from ..layers.sequence import AttentionSequencePoolingLayer
import math


class MultiAspect(BaseModel):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return:  A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, history_feature_list, dnn_use_bn=False,
                 aspect=4, short_long_length=(16, 256),
                 save_param=False, load_param=False,
                 dnn_hidden_units=(200, 80), dnn_activation='prelu', att_hidden_size=(80, 40),
                 att_activation='sigmoid', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.1,
                 seed=1024, task='binary', device='cpu', gpus=None):
        super(MultiAspect, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                          init_std=init_std, seed=seed, task=task, device=device, gpus=gpus,
                                          save_param=save_param, load_param=load_param)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()

        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)

        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

        self.aspect = aspect
        self.Q = nn.Parameter(torch.randn(aspect, att_emb_dim))
        self.multiHead = MultiheadAttention(att_emb_dim, 1, 0)
        # self.multiHead = nn.MultiheadAttention(att_emb_dim, 1)
        self.aspect_linear = nn.Sequential(
            nn.Linear(att_emb_dim, att_emb_dim//aspect, bias=False),  # 输入层
            nn.LayerNorm(att_emb_dim//aspect),  # 添加Layer Normalization
        )
        self.fft_block = AFNO1D(att_emb_dim, num_blocks=aspect)
        self.short_length = short_long_length[0]
        self.long_length = short_long_length[1]
        self.to(device)

    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.history_feature_list, to_list=True)
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)  # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)  # [B, T, E]
        B, T, E = keys_emb.size()
        A = self.aspect

        # interest extract
        emb = torch.cat((query_emb, keys_emb), dim=1)
        Q_reshape = self.Q.reshape(1, A, E).expand(B, -1, -1)
        # aspect_emb_output = self.multiHead(Q_reshape.permute(1, 0, 2), emb.permute(1, 0, 2),
        #                                    emb.permute(1, 0, 2))[0].permute(1, 0, 2)
        interest_emb = self.multiHead(Q_reshape, emb, emb)
        # [B, A, E]
        # self.aspect_loss_fn(aspect_emb_output)

        # short
        short_keys_length = torch.where(keys_length > self.short_length, self.short_length, keys_length)
        short_keys_emb = keys_emb[:, -self.short_length:, :]

        # long
        long_keys_length = keys_length - self.short_length
        long_keys_length = torch.where(long_keys_length > 0, long_keys_length, 0)
        long_keys_emb = keys_emb[:, :-self.short_length, :]
        # retrieval
        topk_token, topk_length = self.retrieval(interest_emb, long_keys_emb, long_keys_length)

        # interest interaction
        token = torch.cat((topk_token, short_keys_emb), dim=1)
        token_multi_intestest = token.unsqueeze(2) * interest_emb.unsqueeze(1)

        # fft
        fft_input = self.aspect_linear(token_multi_intestest).reshape(B, -1, E)
        keys_emb_fft = self.fft_block(fft_input)

        # din
        keys_length_fft = short_keys_length + topk_length
        hist = self.attention(query_emb, keys_emb_fft, keys_length_fft)  # [B, 1, E]

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)
        # y_pred = self.out(dnn_logit) + 0.00000001

        return y_pred

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim

    def retrieval(self, interest, K, long_keys_length):
        score = torch.matmul(K, interest.permute(0, 2, 1))
        score_max, _ = torch.max(score, dim=-1)
        # score_max = torch.sum(score, dim=-1)
        mask_length = self.long_length - self.short_length - long_keys_length
        mask = torch.arange(self.long_length - self.short_length, device=self.device).repeat(len(mask_length), 1)
        mask = mask >= mask_length.view(-1, 1)
        score_max = torch.where(mask, score_max, -100)

        _, index = torch.topk(score_max, 8, dim=-1, largest=True)
        index = torch.sort(index)[0]

        topk_length = (index >= mask_length.view(-1, 1)).sum(-1)
        index = index.reshape(index.shape[0], index.shape[1], 1).expand(-1, -1, K.shape[-1])
        attentive_token = torch.gather(K, 1, index)
        return attentive_token, topk_length

    def aspect_loss_fn(self, emb):
        mse_loss = nn.MSELoss(reduction='mean')
        combinations = torch.combinations(torch.arange(self.aspect), 2)
        embedding_combinations = emb[:, combinations]

        # 计算所有组合的MSE损失
        losses = -mse_loss(embedding_combinations[:, :, 0], embedding_combinations[:, :, 1])
        self.add_auxiliary_loss(losses, 1)



import math


class MultiheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = math.sqrt(hid_dim // n_heads)

    def forward(self, query, key, value, mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x


class AFNO1D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        # 下面四个量是用来声明可学习的权重，2代表实部和虚部，w * x + b的方式来线性组合傅里叶模态下的值
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, L, E = x.shape
        k = self.num_blocks
        e = E // k
        x = x.reshape(B, L, k, e)
        x = torch.fft.rfft(x, dim=1, norm="ortho")  # 对L做FFT变换
        # 0初始化线性组合后的结果，后面可以看出0初始化只需要部分赋值即可实现截断
        o1_real = torch.zeros([B, x.shape[1], k, e], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], k, e], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = L // 2 + 1
        # kept_modes = int(total_modes * self.hard_thresholding_fraction)
        # 由于是0初始化，所以第三个维度的0:kept_modes被赋了值，kept_modes:-1就被截断了
        o1_real = F.relu(
            torch.einsum('...bi,bio->...bo', x.real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x.real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real = (
                torch.einsum('...bi,bio->...bo', o1_real, self.w2[0]) - \
                torch.einsum('...bi,bio->...bo', o1_imag, self.w2[1]) + \
                self.b2[0]
        )

        o2_imag = (
                torch.einsum('...bi,bio->...bo', o1_imag, self.w2[0]) + \
                torch.einsum('...bi,bio->...bo', o1_real, self.w2[1]) + \
                self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)  # softshrink是一种激活函数
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], E)
        x = torch.fft.irfft(x, n=L, dim=1, norm='ortho')  # 傅里叶逆变换，恢复回原来的域
        x = x.type(dtype)
        return x + bias  # shortcut


if __name__ == '__main__':
    pass

'''
先得到interest 检索 interaction afno din
'''