import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.utils import generate_mask, get_last_visit


class FinalAttentionQKV(nn.Module):
    def __init__(
        self,
        attention_input_dim: int,
        attention_hidden_dim: int,
        attention_type: str = "add",
        dropout: float = 0.5,
    ):
        super(FinalAttentionQKV, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(
            torch.zeros(
                1,
            )
        )
        self.b_out = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(
            torch.randn(2 * attention_input_dim, attention_hidden_dim)
        )
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        (
            batch_size,
            time_step,
            lab_dim,
        ) = x.size()  # batch_size * lab_dim + 1 * hidden_dim(i)
        input_q = self.W_q(x[:, -1, :])  # b h
        input_k = self.W_k(x)  # b t h
        input_v = self.W_v(x)  # b t h

        if self.attention_type == "add":  # B*T*I  @ H*I

            q = torch.reshape(
                input_q, (batch_size, 1, self.attention_hidden_dim)
            )  # B*1*H
            h = q + input_k + self.b_in  # b t h
            h = self.tanh(h)  # B*T*H
            e = self.W_out(h)  # b t 1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "mul":
            q = torch.reshape(
                input_q, (batch_size, self.attention_hidden_dim, 1)
            )  # B*h 1
            e = torch.matmul(input_k, q).squeeze(-1)  # b t

        elif self.attention_type == "concat":
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        else:
            raise ValueError(
                "Unknown attention type: {}, please use add, mul, concat".format(
                    self.attention_type
                )
            )

        a = self.softmax(e)  # B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze(1)  # B*I

        return v, a


class PositionwiseFeedForward(nn.Module):  # new added
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x)))), None


class PositionalEncoding(nn.Module):  # new added / not use anymore
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pos = self.pe[:, : x.size(1)].clone().requires_grad_(False)
        x = x + pos
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, self.d_k * self.h) for _ in range(3)]
        )
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)  # b h t d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # b h t t
        if mask is not None:  # 1 1 t t
            scores = scores.masked_fill(mask == 0, -1e9)  # b h t t 下三角
        p_attn = torch.softmax(scores, dim=-1)  # b h t t
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  # b h t v (d_k)

    def cov(self, m, y=None):
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        if x.size(1) <= 1:
            return torch.zeros(
                x.size(0), x.size(0), device=x.device, dtype=x.dtype
            )
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 1 1 t t

        nbatches = query.size(0)  # b
        lab_dim = query.size(1)  # i+1
        feature_dim = query.size(1)  # i+1

        # x size -> # batch_size * d_input * hidden_dim

        # d_model => h * d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]  # b num_head d_input d_k

        x, self.attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )  # b num_head d_input d_v (d_k)

        x = (
            x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        )  # batch_size * d_input * hidden_dim

        # DeCov
        DeCov_contexts = x.transpose(0, 1).transpose(1, 2)  # I+1 H B
        Covs = self.cov(DeCov_contexts[0, :, :])
        DeCov_loss = 0.5 * (
            torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2
        )
        for i in range(feature_dim - 1):
            Covs = self.cov(DeCov_contexts[i + 1, :, :])
            DeCov_loss += 0.5 * (
                torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2
            )

        return self.final_linear(x), DeCov_loss


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]), returned_value[1]


class SingleAttention(nn.Module):
    def __init__(
        self,
        attention_input_dim,
        attention_hidden_dim,
        attention_type="add",
        time_aware=False,
    ):
        super(SingleAttention, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.time_aware = time_aware

        # batch_time = torch.arange(0, batch_mask.size()[1], dtype=torch.float32).reshape(1, batch_mask.size()[1], 1)
        # batch_time = batch_time.repeat(batch_mask.size()[0], 1, 1)

        if attention_type == "add":
            if self.time_aware:
                # self.Wx = nn.Parameter(torch.randn(attention_input_dim+1, attention_hidden_dim))
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.bh = nn.Parameter(
                torch.zeros(
                    attention_hidden_dim,
                )
            )
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "mul":
            self.Wa = nn.Parameter(
                torch.randn(attention_input_dim, attention_input_dim)
            )
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "concat":
            if self.time_aware:
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim + 1, attention_hidden_dim)
                )
            else:
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim, attention_hidden_dim)
                )

            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == "new":
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.Wx = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )

            self.rate = nn.Parameter(torch.zeros(1) + 0.8)
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        else:
            raise RuntimeError(
                "Wrong attention type. Please use 'add', 'mul', 'concat' or 'new'."
            )

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, mask, device):

        (
            batch_size,
            time_step,
            lab_dim,
        ) = x.size()  # batch_size * time_step * hidden_dim(i)

        time_decays = (
            torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32)
            .unsqueeze(-1)
            .unsqueeze(0)
            .to(device=device)
        )  # 1*t*1
        b_time_decays = time_decays.repeat(batch_size, 1, 1) + 1  # b t 1

        if self.attention_type == "add":  # B*T*I  @ H*I
            last_visit = get_last_visit(x, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            if self.time_aware == True:
                k = torch.matmul(x, self.Wx)  # b t h
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)  # b t h
            else:
                k = torch.matmul(x, self.Wx)  # b t h
            h = q + k + self.bh  # b t h
            if self.time_aware:
                h += time_hidden
            h = self.tanh(h)  # B*T*H
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        elif self.attention_type == "mul":
            last_visit = get_last_visit(x, mask)
            e = torch.matmul(last_visit, self.Wa)  # b i
            e = (
                torch.matmul(e.unsqueeze(1), x.permute(0, 2, 1)).reshape(
                    batch_size, time_step
                )
                + self.ba
            )  # b t
        elif self.attention_type == "concat":
            last_visit = get_last_visit(x, mask)
            q = last_visit.unsqueeze(1).repeat(1, time_step, 1)  # b t i
            k = x
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            if self.time_aware:
                c = torch.cat((c, b_time_decays), dim=-1)  # B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "new":
            last_visit = get_last_visit(x, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            k = torch.matmul(x, self.Wx)  # b t h
            dot_product = torch.matmul(q, k.transpose(1, 2)).reshape(
                batch_size, time_step
            )  # b t
            denominator = self.sigmoid(self.rate) * (
                torch.log(2.72 + (1 - self.sigmoid(dot_product)))
                * (b_time_decays.reshape(batch_size, time_step))
            )
            e = self.relu(self.sigmoid(dot_product) / (denominator))  # b * t
        else:
            raise ValueError(
                "Wrong attention type. Plase use 'add', 'mul', 'concat' or 'new'."
            )

        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        a = self.softmax(e)  # B*T
        v = torch.matmul(a.unsqueeze(1), x).reshape(batch_size, lab_dim)  # B*I

        return v, a


class ConCareLayer(nn.Module):
    """ConCare layer.

    Paper: Liantao Ma et al. Concare: Personalized clinical feature embedding via capturing the healthcare context. AAAI 2020.

    This layer is used in the ConCare model. But it can also be used as a
    standalone layer.

    Args:
        lab_dim: dynamic feature size.
        demo_dim: static feature size, if 0, then no static feature is used.
        hidden_dim: hidden dimension of the channel-wise GRU, default 128.
        transformer_hidden: hidden dimension of the transformer, default 128.
        num_head: number of heads in the transformer, default 4.
        pe_hidden: hidden dimension of the positional encoding, default 64.
        dropout: dropout rate, default 0.5.

    Examples:
        >>> from pyhealth.models import ConCare
        >>> x = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = ConCare(64)
        >>> c, _ = layer(x)
        >>> c.shape
        torch.Size([3, 128])
    """

    def __init__(
        self,
        lab_dim: int,
        demo_dim: int = 0,
        hidden_dim: int = 128,
        num_head: int = 4,
        pe_hidden: int = 64,
        dropout: int = 0.5,
        use_checkpoint: bool = True,
    ):
        super(ConCareLayer, self).__init__()

        # hyperparameters
        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim  # d_model
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
        # self.output_dim = output_dim
        self.dropout = dropout
        self.demo_dim = demo_dim
        self.use_checkpoint = use_checkpoint

        # layers
        self.PositionalEncoding = PositionalEncoding(
            self.transformer_hidden, dropout=0, max_len=400
        )

        self.GRUs = nn.ModuleList(
            [
                nn.GRU(1, self.hidden_dim, batch_first=True)
                for _ in range(self.lab_dim)
            ]
        )
        self.LastStepAttentions = nn.ModuleList(
            [
                SingleAttention(
                    self.hidden_dim,
                    8,
                    attention_type="new",
                    time_aware=True,
                )
                for _ in range(self.lab_dim)
            ]
        )

        self.FinalAttentionQKV = FinalAttentionQKV(
            self.hidden_dim,
            self.hidden_dim,
            attention_type="mul",
            dropout=self.dropout,
        )

        self.MultiHeadedAttention = MultiHeadedAttention(
            self.num_head, self.transformer_hidden, dropout=self.dropout
        )
        self.SublayerConnection = SublayerConnection(
            self.transformer_hidden, dropout=self.dropout
        )

        self.PositionwiseFeedForward = PositionwiseFeedForward(
            self.transformer_hidden, self.pe_hidden, dropout=0.1
        )

        if self.demo_dim > 0:
            self.demo_proj_main = nn.Linear(self.demo_dim, self.hidden_dim)

        self.dropout = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def _run_feature_encoder(
        self,
        feature_idx: int,
        feature_x: torch.tensor,
        mask: Optional[torch.tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.tensor:
        def feature_forward(cur_x, cur_mask, init_hidden):
            gru_out = self.GRUs[feature_idx](cur_x.unsqueeze(-1), init_hidden)[0]
            return self.LastStepAttentions[feature_idx](
                gru_out, cur_mask, cur_x.device
            )[0]

        init_hidden = torch.zeros(batch_size, self.hidden_dim).to(device=device).unsqueeze(0)
        if self.training and self.use_checkpoint:
            return checkpoint(
                feature_forward,
                feature_x,
                mask,
                init_hidden,
                use_reentrant=False,
            )
        return feature_forward(feature_x, mask, init_hidden)

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, lab_dim].
            static: a tensor of shape [batch size, demo_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
        """
        # rnn will only apply dropout between layers
        if self.demo_dim > 0:
            demo_main = self.tanh(self.demo_proj_main(static)).unsqueeze(
                1
            )  # b hidden_dim

        batch_size = x.size(0)
        time_step = x.size(1)
        feature_dim = x.size(2)

        if self.transformer_hidden % self.num_head != 0:
            raise ValueError("transformer_hidden must be divisible by num_head")

        # Encode each lab feature independently, then concatenate once to reduce
        # peak memory from repeated cat operations.
        attention_inputs = []
        for i in range(feature_dim):
            encoded_feature = self._run_feature_encoder(
                i, x[:, :, i], mask, batch_size, x.device
            )
            attention_inputs.append(encoded_feature.unsqueeze(1))

        Attention_embeded_input = torch.cat(attention_inputs, dim=1)

        if self.demo_dim > 0:
            Attention_embeded_input = torch.cat(
                (Attention_embeded_input, demo_main), 1
            )  # b i+1 h
        posi_input = self.dropout(
            Attention_embeded_input
        )  # batch_size * d_input+1 * hidden_dim

        contexts = self.SublayerConnection(
            posi_input,
            lambda _: self.MultiHeadedAttention(
                posi_input, posi_input, posi_input, None
            ),
        )  # # batch_size * d_input * hidden_dim

        DeCov_loss = contexts[1]
        contexts = contexts[0]

        contexts = self.SublayerConnection(
            contexts, lambda _: self.PositionwiseFeedForward(contexts)
        )[0]

        weighted_contexts, feature_attn = self.FinalAttentionQKV(contexts)
        return weighted_contexts, DeCov_loss, feature_attn

class ConCare(nn.Module):
    def __init__(
        self,
        lab_dim: int,
        demo_dim: int = 0,
        hidden_dim: int = 128,
        num_head: int = 4,
        pe_hidden: int = 64,
        dropout: int = 0.0,
        use_checkpoint: bool = True,
        max_seq_len: int = 512,
        **kwargs
    ):
        super(ConCare, self).__init__()

        # hyperparameters
        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim  # d_model
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
        self.use_checkpoint = use_checkpoint
        self.max_seq_len = max_seq_len
        # self.output_dim = output_dim
        self.dropout = nn.Dropout(p=dropout)
        self.demo_dim = demo_dim
        self.concare_layer = ConCareLayer(
            lab_dim=lab_dim,
            demo_dim=demo_dim,
            hidden_dim=hidden_dim,
            num_head=num_head,
            pe_hidden=pe_hidden,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

    def _compress_sequence(self, x, mask):
        if mask is None:
            return x, mask

        lengths = mask.sum(dim=1).long()
        max_length = int(lengths.max().item())
        x = x[:, :max_length, :]
        mask = mask[:, :max_length]

        if max_length <= self.max_seq_len:
            return x, mask

        target_lengths = torch.clamp(lengths, max=self.max_seq_len)
        max_target_length = int(target_lengths.max().item())
        compressed = x.new_zeros(x.size(0), max_target_length, x.size(2))
        compressed_mask = mask.new_zeros(mask.size(0), max_target_length)

        for idx, (src_len, dst_len) in enumerate(zip(lengths.tolist(), target_lengths.tolist())):
            valid_input = x[idx, :src_len, :].transpose(0, 1).unsqueeze(0)
            pooled_input = F.adaptive_avg_pool1d(valid_input, dst_len).squeeze(0).transpose(0, 1)
            compressed[idx, :dst_len, :] = pooled_input
            compressed_mask[idx, :dst_len] = 1

        return compressed, compressed_mask

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, lab_dim].
            static: a tensor of shape [batch size, demo_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
        """
        # rnn will only apply dropout between layers
        x, mask = self._compress_sequence(x, mask)
        batch_size, time_steps, _ = x.size()
        # out = torch.zeros((batch_size, time_steps, self.hidden_dim))
        # feature_attn = torch.zeros((batch_size, time_steps, self.lab_dim))
        # decov_loss = 0
        # for cur_time in range(time_steps):
        #     cur_x = x[:, :cur_time+1, :]
        #     cur_mask = mask[:, :cur_time+1]
        #     cur_out, decov, cur_feature_attn = self.concare_layer(cur_x, static, cur_mask)
        #     out[:, cur_time, :] = cur_out
        #     feature_attn[:, cur_time, :] = cur_feature_attn[:, :self.lab_dim]
        #     decov_loss += decov
        # decov_loss /= time_steps
        
        out, decov_loss, feature_attn = self.concare_layer(x, static, mask)
        out = self.dropout(out)
        return out, decov_loss, feature_attn
