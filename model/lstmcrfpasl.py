import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import log_sum_exp_pa, log_sum_exp_pytorch, PAD, UNK, START, STOP


class Bilstm(nn.Module):

    def __init__(self, args, dicts, embedding_table):
        super(Bilstm, self).__init__()
        self.char_size = dicts['char_size']
        self.tag_size = dicts['tag_size']
        self.max_len = args.max_len
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.embed_layer = self.build_embed_layer(embedding_table)
        self.char_drop = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim // 2,
            bidirectional=True,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(self.hidden_dim, self.tag_size)

    def build_embed_layer(self, embedding_table):
        if embedding_table is not None:
            embedding_table = torch.tensor(embedding_table, dtype=torch.float)
            embed_layer = nn.Embedding.from_pretrained(
                embedding_table,
                freeze=False
            )
        else:
            embed_layer = nn.Embedding(self.char_size, self.char_embedding_dim)
        return embed_layer

    def forward(self, data):
        char_seq_tensor = data['char_seq_tensor']
        char_seq_lens = data['char_seq_lens']

        embedding = self.char_drop(self.embed_layer(char_seq_tensor))

        packed = pack_padded_sequence(
            embedding,
            char_seq_lens,
            batch_first=True,
            enforce_sorted=False
        )
        lstm_out, (_, _) = self.lstm(packed, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        out = self.linear(self.dropout(lstm_out))

        return out

    def encode_state(self, data):
        batch_size = data['batch_size']
        char_seq_tensor = data['char_seq_tensor']
        tag_seq_tensor = data['tag_seq_tensor']  # [B x T x C]
        embedding = self.char_drop(self.embed_layer(char_seq_tensor))  # [B x T x E]
        lstm_out, (_, _) = self.lstm(embedding, None)
        # lstm_out = lstm_out.transpose(0, 1).contiguous()  # [B x T x H]
        score = self.linear(self.dropout(lstm_out))  # [B x T x C]
        allow_num = torch.sum(tag_seq_tensor, dim=-1)  # [B x T]
        mask = torch.eq(tag_seq_tensor, 0)
        masked_score = torch.masked_fill(score, mask, 0)
        label_score = torch.sum(masked_score, dim=-1)  # [B x T]
        label_rep = torch.div(label_score, allow_num)
        sent_rep = lstm_out.reshape(batch_size, -1)
        state_rep = torch.cat([sent_rep, label_rep], dim=-1)  # [B, T * H + C]

        return state_rep


class LstmCrfPaSl(nn.Module):

    def __init__(self, args, dicts, embedding_table):
        super(LstmCrfPaSl, self).__init__()
        self.tag_size = dicts['tag_size']
        self.tag_to_idx = dicts['tag_to_idx']
        self.PAD_IDX = self.tag_to_idx[PAD]
        self.START_IDX = self.tag_to_idx[START]
        self.STOP_IDX = self.tag_to_idx[STOP]
        self.UNK_IDX = self.tag_to_idx[UNK]
        self.bilstm = Bilstm(args, dicts, embedding_table)

        self.trans = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size)
        )
        self.trans.data[self.START_IDX, :] = -10000
        self.trans.data[:, self.STOP_IDX] = -10000
        self.trans.data[:, self.PAD_IDX] = -10000
        self.trans.data[self.PAD_IDX, :] = -10000
        self.trans.data[self.UNK_IDX, :] = -10000
        self.trans.data[:, self.UNK_IDX] = -10000

    def encode(self, data):
        return self.bilstm.encode_state(data)

    def _forward_alg(self, lstm_out, mask):
        max_seq_len = lstm_out.size(1)

        trans = self.trans.unsqueeze(0)  # [1 x C x C]
        alpha = lstm_out[:, 0, :] + trans[:, :, self.START_IDX]  # [B x C]
        for t in range(1, max_seq_len):
            mask_t = mask[:, t].unsqueeze(1)
            emit_score = lstm_out[:, t, :].unsqueeze(2)  # [B x C x 1]
            alpha_t = alpha.unsqueeze(1) + emit_score + trans
            alpha_t = log_sum_exp_pytorch(alpha_t)  # [B x C]
            alpha = mask_t * alpha_t + (1 - mask_t) * alpha
        alpha = alpha + self.trans[self.STOP_IDX, :]  # [B x C]
        return log_sum_exp_pytorch(alpha)

    def _gold_score_pa(self, lstm_out, tag_seq_tensor, dicts, mask):
        batch_size = lstm_out.size(0)
        max_seq_len = lstm_out.size(1)
        tag_size = dicts['tag_size']
        mask_start = tag_seq_tensor[:, 0, :]  # [B x C]
        emit_start = lstm_out[:, 0, :]  # [B x C]
        gold_score = (emit_start + self.trans[:, self.START_IDX].unsqueeze(0)) * mask_start  # [B x C]
        mask_pre = mask_start
        for t in range(1, max_seq_len):
            mask_t = mask[:, t].unsqueeze(-1)  # [B x 1]
            mask_cur = tag_seq_tensor[:, t, :]  # [B x C]
            emit_t = (lstm_out[:, t, :] * mask_cur).unsqueeze(-1)  # [B x C x 1]
            trans = self.trans.unsqueeze(0).expand(batch_size, tag_size, tag_size)  # [B x C x C]
            mask_pre_ = mask_pre.unsqueeze(1)  # [B x 1 x C]
            mask_cur_ = mask_cur.unsqueeze(2)  # [B x C x 1]
            mask_all = torch.ones(batch_size, tag_size, tag_size) * mask_pre_ * mask_cur_
            trans = trans * mask_all  # [B x C x C]
            emit_t = emit_t.expand(batch_size, tag_size, tag_size) * mask_all
            before_logsumexp = gold_score.unsqueeze(1).expand(batch_size, tag_size, tag_size) * mask_all + \
                      trans + emit_t  # [B x C x C]
            score_t = log_sum_exp_pa(before_logsumexp, mask_all)  # [B x C]

            gold_score = score_t * mask_t + gold_score * (1 - mask_t)
            mask_pre = mask_cur
        mask_stop = torch.eq(torch.eq(gold_score, 0), False).float()
        gold_score = gold_score + self.trans[self.STOP_IDX, :].unsqueeze(0) * mask_stop  # [B x C]
        return log_sum_exp_pa(gold_score, mask_stop)  # [B]

    def neg_log_likelihood(self, data, dicts):
        lstm_out = self.bilstm(data)
        char_seq_tensor = data['char_seq_tensor']
        tag_seq_tensor = data['tag_seq_tensor']
        mask = torch.gt(char_seq_tensor, self.PAD_IDX).float()
        forward_score = self._forward_alg(lstm_out, mask)
        gold_score = self._gold_score_pa(lstm_out, tag_seq_tensor, dicts, mask)
        return torch.sum(forward_score - gold_score)

    def forward(self, data):
        lstm_out = self.bilstm(data)
        char_seq_tensor = data['char_seq_tensor']
        mask = torch.gt(char_seq_tensor, self.PAD_IDX).float()
        best_paths = self.decode(lstm_out, mask)
        return best_paths

    def decode(self, lstm_out, mask):
        batch_size = lstm_out.size(0)
        max_seq_len = lstm_out.size(1)
        score = torch.full((batch_size, self.tag_size), -10000, dtype=torch.float)
        score[:, self.START_IDX] = 0
        bptrs = torch.zeros((batch_size, max_seq_len, self.tag_size), dtype=torch.long)

        trans = self.trans.unsqueeze(0)  # [1 x C x C]
        for t in range(max_seq_len):
            mask_t = mask[:, t].unsqueeze(1)  # [B x 1]
            score_t = score.unsqueeze(1) + trans  # [B x C x C]
            max_scores, best_tag_ids = torch.max(score_t, dim=2)  # [B x C]
            max_scores += lstm_out[:, t, :]
            bptrs[:, t, :] = best_tag_ids
            score = mask_t * max_scores + (1 - mask_t) * score
        last_to_stop = self.trans[self.STOP_IDX, :].unsqueeze(0)  # [1 x C]
        score = score + last_to_stop

        best_scores, best_tags = torch.max(score, dim=1)  # [B]
        bptrs = bptrs.tolist()
        best_paths = []

        for i in range(batch_size):
            best_path = [best_tags[i].item()]
            bptr = bptrs[i]
            cur_seq_len = int(torch.sum(mask[i]).item())
            for j in reversed(range(1, cur_seq_len)):
                best_path.append(bptr[j][best_path[-1]])
            best_path.reverse()
            best_paths.append(best_path)

        return best_paths
