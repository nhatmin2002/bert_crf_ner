#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import List, Optional


class CRF(nn.Module):
    def __init__(self, num_tags, batch_first=False):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask=None, reduction="mean"):
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == "none":
            return llh
        if reduction == "mean":
            return llh.mean()
        if reduction == "sum":
            return llh.sum()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions, mask=None, nbest=None, pad_tag=None):
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.uint8, device=emissions.device
            )
        if mask.dtype != torch.uint8:
            mask = mask.byte()

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpoe(0, 1)

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode(emissions, mask, nbest, pad_tag)

    def _compute_score(self, emissions, tags, mask):
        seq_length, batch_size = tags.shape
        mask = mask.float()

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions, mask):
        seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, mask, pad_tag=None):
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros(
            (seq_length, batch_size, self.num_tags), dtype=torch.long, device=device
        )
        oor_idx = torch.zeros(
            (batch_size, self.num_tags), dtype=torch.long, device=device
        )
        oor_tag = torch.full(
            (batch_size, self.num_tags), pad_tag, dtype=torch.long, device=device
        )

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at colum j stores the score of the tag sequence so far that ends with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used when we
        # trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions where
        # mask is 0, i.e. out of range

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission

            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)
        seq_end = mask.long().sum(dim=0) - 1
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(
            -1,
            seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
            end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags),
        )
        history_idx = history_idx.transpose(1, 0).contiguous()

        best_tags_arr = torch.zeros(
            (seq_length, batch_size), dtype=torch.long, device=device
        )
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)
        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions, mask, nbest, pad_tag):
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros(
            (seq_length, batch_size, self.num_tags, nbest),
            type=torch.long,
            device=device,
        )
        oor_idx = torch.zeros(
            (batch_size, self.num_tags, nbest), dtype=torch.long, device=device
        )
        oor_tag = torch.full(
            (seq_length, batch_size, nbest), pad_tag, dtype=torch.long, device=device
        )

        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                next_score = (
                    broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission
                )

            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(
                nbest, dim=1
            )
            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        seq_ends = mask.long().sum(dim=0) - 1

        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(
            1,
            seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
            end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest),
        )
        history_idx = history_idx.transpose(1, 0).contiguous()
        best_tags_arr = torch.zeros(
            (seq_length, batch_size, nbest), dtype=torch.long, device=device
        )
        best_tags = (
            torch.arange(nbest, dtype=torch.long, device=device)
            .view(1, -1)
            .expand(batch_size, -1)
        )
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(
                history_idx[idx].view(batch_size, -1), 1, best_tags
            )
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest
        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)
