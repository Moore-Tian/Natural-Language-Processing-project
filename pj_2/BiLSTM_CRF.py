import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMCRF(nn.Module):
    def __init__(self, tag_size, vocab_size, dropout_rate=0.5, embed_size=256, hidden_size=256):
        super(BiLSTMCRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.tag_size = tag_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size + 1, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)
        self.hidden2emit_score = nn.Linear(hidden_size * 2, tag_size)
        self.transition = nn.Parameter(torch.randn(tag_size, tag_size))

    def forward(self, sentences, tags, sen_lengths):
        mask = (sentences != self.vocab_size - 1).to(self.device)
        sentences = self.embedding(sentences.transpose(0, 1))
        emit_score = self.bilstm(sentences, sen_lengths)
        loss = self.crf(tags, mask, emit_score)
        return loss

    def bilstm(self, sentences, sent_lengths):
        padded_sentences = pack_padded_sequence(sentences, sent_lengths)
        hidden_states, _ = self.lstm(padded_sentences)
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True)
        emit_score = self.hidden2emit_score(hidden_states)
        emit_score = self.dropout(emit_score)
        return emit_score

    def crf(self, tags, mask, emit_score):
        batch_size, sentence_len = tags.shape
        real_score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)
        real_score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        real_score = (real_score * mask.type(torch.float)).sum(dim=1)

        temp = torch.unsqueeze(emit_score[:, 0], dim=1)
        for i in range(1, sentence_len):
            num_unfinished = mask[:, i].sum()
            temp_unfinished = temp[: num_unfinished]
            emit_plus_transition = emit_score[: num_unfinished, i].unsqueeze(dim=1) + self.transition
            forward_score = temp_unfinished.transpose(1, 2) + emit_plus_transition
            max_score = forward_score.max(dim=1)[0].unsqueeze(dim=1)
            forward_score -= max_score
            temp_unfinished = max_score + torch.logsumexp(forward_score, dim=1).unsqueeze(dim=1)
            temp = torch.cat((temp_unfinished, temp[num_unfinished:]), dim=0)
        temp = temp.squeeze(dim=1)
        max_temp = temp.max(dim=-1)[0]
        logsumexp_all_scores = max_temp + torch.logsumexp(temp - max_temp.unsqueeze(dim=1), dim=1)

        log_likelihood = real_score - logsumexp_all_scores
        loss = -log_likelihood
        return loss

    def predict(self, sentences, sen_lengths):
        batch_size = sentences.shape[0]
        mask = (sentences != self.vocab_size - 1)
        sentences = sentences.transpose(0, 1)
        sentences = self.embedding(sentences)
        emit_score = self.bilstm(sentences, sen_lengths)

        tags = [[[i] for i in range(self.tag_size)]] * batch_size
        temp = torch.unsqueeze(emit_score[:, 0], dim=1)
        for i in range(1, sen_lengths[0]):
            num_unfinished = mask[:, i].sum()
            temp_unfinished = temp[: num_unfinished]
            emit_plus_transition = self.transition + emit_score[: num_unfinished, i].unsqueeze(dim=1)
            new_temp_unfinished = temp_unfinished.transpose(1, 2) + emit_plus_transition
            temp_unfinished, max_idx = torch.max(new_temp_unfinished, dim=1)
            max_idx = max_idx.tolist()
            tags[: num_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(num_unfinished)]
            temp = torch.cat((torch.unsqueeze(temp_unfinished, dim=1), temp[num_unfinished:]), dim=0)
        temp = temp.squeeze(dim=1)

        _, max_idx = torch.max(temp, dim=1)
        max_idx = max_idx.tolist()
        tags = [tags[batch_idx][tag_idx] for batch_idx, tag_idx in enumerate(max_idx)]
        return tags

    def save(self, filepath):
        params = {
            'tag_size': self.tag_size,
            'vocab_size': self.vocab_size,
            'args': dict(dropout_rate=self.dropout_rate, embed_size=self.embed_size, hidden_size=self.hidden_size),
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = BiLSTMCRF(params['tag_size'], params['vocab_size'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model

    @property
    def device(self):
        return self.embedding.weight.device