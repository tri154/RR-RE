from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F

from models import RobertaWithLastAttention


small_positive = 1e-10

class Encoder(nn.Module):
    result_path: str
    device: str
    log_path: str

    name: str
    transformer_type: str
    attn_impl: str
    lazy: bool

    def __init__(self, encoder_cfg):
        super().__init__()

        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, encoder_cfg.get(name))

        self.config = AutoConfig.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if not self.lazy:
            self.load_model()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.max_num_tokens = 512
        if self.transformer_type == "bert":
            self.start_token_ids = torch.tensor([self.tokenizer.cls_token_id], dtype=torch.long, device=device)
            self.end_token_ids = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long, device=device)
        elif self.transformer_type == "roberta":
            self.start_token_ids = torch.tensor([self.tokenizer.cls_token_id], dtype=torch.long, device=device)
            self.end_token_ids = torch.tensor([self.tokenizer.sep_token_id, self.tokenizer.sep_token_id], dtype=torch.long, device=device)
        else:
            raise NotImplementedError()

        self.start_token_len = len(self.start_token_ids)
        self.end_token_len = len(self.end_token_ids)
        self.pad_token_ids = self.tokenizer.pad_token_id


    def load_model(self):
        if not hasattr(self, "encoder"):
            if self.attn_impl is not None:
                self.encoder = RobertaWithLastAttention(
                    self.name,
                    config=self.config,
                    attn_implementation=self.attn_impl,
                    # add_pooling_layer=False
                )
            else:
                self.encoder = RobertaWithLastAttention(
                    self.name,
                    config=self.config,
                    # add_pooling_layer=False
                )


    def forward_sliding_window_with_attention(self, batch_token_seqs, batch_token_masks, stride=128):
        # CHANGED: remove token type.
        # if 'roberta' in self.transformer.config._name_or_path:
        #     batch_token_types = torch.zeros_like(batch_token_types)

        batch_size, max_doc_length = batch_token_seqs.shape

        if max_doc_length <= self.max_num_tokens:
            # batch_output = self.transformer(input_ids=batch_token_seqs, attention_mask=batch_token_masks, token_type_ids=batch_token_types, output_attentions=True)
            batch_output = self.encoder(
                input_ids=batch_token_seqs,
                attention_mask=batch_token_masks,
                # output_attentions=True
            )
            batch_token_embs = batch_output[0]
            batch_token_atts = batch_output[-1][-1]
            return batch_token_embs, batch_token_atts

        num_token_per_doc = batch_token_masks.sum(1).int().tolist()

        token_seqs = list()
        token_masks = list()
        # token_types = list()
        num_seg_per_doc = list()
        valids = list()


        for did, num_token in enumerate(num_token_per_doc):
            if num_token <= self.max_num_tokens:
                token_seqs.append(batch_token_seqs[did, :self.max_num_tokens])
                token_masks.append(batch_token_masks[did, :self.max_num_tokens])
                # token_types.append(batch_token_types[did, :self.max_num_tokens])
                num_seg_per_doc.append(1)
                valids.append((-1, -1)) # not contribute
                continue

            start = 0
            end = self.max_num_tokens - self.end_token_len
            num_seg = 1

            sequence = torch.cat([batch_token_seqs[did, start:end], self.end_token_ids], dim=-1)
            mask = batch_token_masks[did, start:end + self.end_token_len]
            # type = torch.cat([batch_token_types[did, start:end],
                             # batch_token_types[did, end - 1].repeat(self.end_token_len)], dim=-1)

            token_seqs.append(sequence)
            token_masks.append(mask)
            # token_types.append(type)
            valids.append((start, end))

            while True:
                start = end - stride
                end = start + self.max_num_tokens - self.end_token_len - self.start_token_len
                num_seg += 1
                if end >= num_token:
                    end = min(end, num_token)

                    sequence = torch.cat([self.start_token_ids,
                                         batch_token_seqs[did, start:end]], dim=-1)
                    mask = batch_token_masks[did, start - self.start_token_len:end]
                    # type = torch.cat([batch_token_types[did, start].repeat(self.start_token_len),
                    #                  batch_token_types[did, start:end]], dim=-1)

                    pad_len = self.max_num_tokens - sequence.shape[-1]
                    sequence = F.pad(sequence, (0, pad_len), value=self.pad_token_ids)
                    mask = F.pad(mask, (0, pad_len), value=0)
                    # type = F.pad(type, (0, pad_len), value=0)

                    token_seqs.append(sequence)
                    token_masks.append(mask)
                    # token_types.append(type)
                    valids.append((start, end))
                    num_seg_per_doc.append(num_seg)
                    break

                sequence = torch.cat([self.start_token_ids,
                                     batch_token_seqs[did, start:end],
                                     self.end_token_ids], dim=-1)
                mask = batch_token_masks[did, start - self.start_token_len:end + self.end_token_len]
                # type = torch.cat([batch_token_types[did, start].repeat(self.start_token_len),
                #                  batch_token_types[did, start:end],
                #                  batch_token_types[did, end - 1].repeat(self.end_token_len)], dim=-1)

                token_seqs.append(sequence)
                token_masks.append(mask)
                # token_types.append(type)
                valids.append((start, end))

        batch_token_seqs = torch.stack(token_seqs).long()
        # CHANGED: to bool saving mem.
        batch_token_masks = torch.stack(token_masks).bool()
        # batch_token_types = torch.stack(token_types).long()

        batch_output = self.encoder(
            input_ids=batch_token_seqs,
            attention_mask=batch_token_masks,
            # token_type_ids=batch_token_types,
            # output_attentions=True
        )
        token_embs = batch_output[0]
        token_atts = batch_output[-1][-1]

        batch_token_embs = list()
        batch_token_atts = list()
        seg_id = 0
        for num_seg, num_token in zip(num_seg_per_doc, num_token_per_doc):
            if num_seg == 1:
                emb = F.pad(token_embs[seg_id], (0, 0, 0, max_doc_length - self.max_num_tokens))
                att = F.pad(token_atts[seg_id], (0, max_doc_length - self.max_num_tokens, 0, max_doc_length - self.max_num_tokens))
                batch_token_embs.append(emb)
                batch_token_atts.append(att)
            else:
                t_embs = list()
                t_atts = list()
                t_masks = list()
                for i in range(num_seg):
                    valid = valids[seg_id + i]
                    num_valid = valid[1] - valid[0]
                    if i == 0: #valid = 511
                        sl = (0, num_valid)
                    elif i == num_seg - 1: #valid = ??
                        sl = (self.start_token_len, self.start_token_len + num_valid)
                    else: #valid = 512
                        sl = (self.start_token_len, self.start_token_len + num_valid)

                    emb = F.pad(token_embs[seg_id + i, sl[0]:sl[1]],
                                pad=(0, 0, valid[0], max_doc_length - valid[1]))
                    att = F.pad(token_atts[seg_id + i, :, sl[0]:sl[1], sl[0]:sl[1]],
                                pad=(valid[0], max_doc_length - valid[1], valid[0], max_doc_length - valid[1]))
                    mask = F.pad(batch_token_masks[seg_id + i, sl[0]:sl[1]],
                                 pad=(valid[0], num_token - valid[1])) # should be num_token.

                    t_embs.append(emb)
                    t_atts.append(att)
                    t_masks.append(mask)
                t_embs = torch.stack(t_embs, dim=0)
                t_atts = torch.stack(t_atts, dim=0)
                t_masks = torch.stack(t_masks, dim=0)

                t_masks = t_masks.sum(dim=0)
                pad_len = max_doc_length - t_masks.shape[0]
                if pad_len > 0:
                    t_masks = F.pad(t_masks, pad=(0, max_doc_length - t_masks.shape[0]), value=1.0)
                doc_token_embs = t_embs.sum(0) / t_masks.unsqueeze(-1)
                doc_token_atts = t_atts.sum(0)
                doc_token_atts = doc_token_atts / (doc_token_atts.sum(-1, keepdim=True) + small_positive)
                batch_token_embs.append(doc_token_embs)
                batch_token_atts.append(doc_token_atts)
            seg_id += num_seg

        batch_token_embs = torch.stack(batch_token_embs)
        batch_token_atts = torch.stack(batch_token_atts)

        return batch_token_embs, batch_token_atts

    def forward_sliding_window_without_attention(self, batch_token_seqs, batch_token_masks, stride=128):
        batch_size, max_doc_length = batch_token_seqs.shape

        if max_doc_length <= self.max_num_tokens:
            # batch_output = self.transformer(input_ids=batch_token_seqs, attention_mask=batch_token_masks, token_type_ids=batch_token_types, output_attentions=True)
            batch_output = self.encoder(input_ids=batch_token_seqs,
                                        attention_mask=batch_token_masks)
            batch_token_embs = batch_output[0]
            return batch_token_embs, None

        num_token_per_doc = batch_token_masks.sum(1).int().tolist()

        token_seqs = list()
        token_masks = list()
        # token_types = list()
        num_seg_per_doc = list()
        valids = list()


        for did, num_token in enumerate(num_token_per_doc):
            if num_token <= self.max_num_tokens:
                token_seqs.append(batch_token_seqs[did, :self.max_num_tokens])
                token_masks.append(batch_token_masks[did, :self.max_num_tokens])
                # token_types.append(batch_token_types[did, :self.max_num_tokens])
                num_seg_per_doc.append(1)
                valids.append((-1, -1)) # not contribute
                continue

            start = 0
            end = self.max_num_tokens - self.end_token_len
            num_seg = 1

            sequence = torch.cat([batch_token_seqs[did, start:end], self.end_token_ids], dim=-1)
            mask = batch_token_masks[did, start:end + self.end_token_len]
            # type = torch.cat([batch_token_types[did, start:end],
                             # batch_token_types[did, end - 1].repeat(self.end_token_len)], dim=-1)

            token_seqs.append(sequence)
            token_masks.append(mask)
            # token_types.append(type)
            valids.append((start, end))

            while True:
                start = end - stride
                end = start + self.max_num_tokens - self.end_token_len - self.start_token_len
                num_seg += 1
                if end >= num_token:
                    end = min(end, num_token)

                    sequence = torch.cat([self.start_token_ids,
                                         batch_token_seqs[did, start:end]], dim=-1)
                    mask = batch_token_masks[did, start - self.start_token_len:end]
                    # type = torch.cat([batch_token_types[did, start].repeat(self.start_token_len),
                    #                  batch_token_types[did, start:end]], dim=-1)

                    pad_len = self.max_num_tokens - sequence.shape[-1]
                    sequence = F.pad(sequence, (0, pad_len), value=self.pad_token_ids)
                    mask = F.pad(mask, (0, pad_len), value=0)
                    # type = F.pad(type, (0, pad_len), value=0)

                    token_seqs.append(sequence)
                    token_masks.append(mask)
                    # token_types.append(type)
                    valids.append((start, end))
                    num_seg_per_doc.append(num_seg)
                    break

                sequence = torch.cat([self.start_token_ids,
                                     batch_token_seqs[did, start:end],
                                     self.end_token_ids], dim=-1)
                mask = batch_token_masks[did, start - self.start_token_len:end + self.end_token_len]
                # type = torch.cat([batch_token_types[did, start].repeat(self.start_token_len),
                #                  batch_token_types[did, start:end],
                #                  batch_token_types[did, end - 1].repeat(self.end_token_len)], dim=-1)

                token_seqs.append(sequence)
                token_masks.append(mask)
                # token_types.append(type)
                valids.append((start, end))

        batch_token_seqs = torch.stack(token_seqs).long()
        # CHANGED: to bool saving mem.
        batch_token_masks = torch.stack(token_masks).bool()
        # batch_token_types = torch.stack(token_types).long()

        batch_output = self.encoder(input_ids=batch_token_seqs,
                                        attention_mask=batch_token_masks,
                                        # token_type_ids=batch_token_types,
                                        )
        token_embs = batch_output[0]

        batch_token_embs = list()
        seg_id = 0
        for num_seg, num_token in zip(num_seg_per_doc, num_token_per_doc):
            if num_seg == 1:
                emb = F.pad(token_embs[seg_id], (0, 0, 0, max_doc_length - self.max_num_tokens))
                batch_token_embs.append(emb)
            else:
                t_embs = list()
                t_masks = list()
                for i in range(num_seg):
                    valid = valids[seg_id + i]
                    num_valid = valid[1] - valid[0]
                    if i == 0: #valid = 511
                        sl = (0, num_valid)
                    elif i == num_seg - 1: #valid = ??
                        sl = (self.start_token_len, self.start_token_len + num_valid)
                    else: #valid = 512
                        sl = (self.start_token_len, self.start_token_len + num_valid)

                    emb = F.pad(token_embs[seg_id + i, sl[0]:sl[1]],
                                pad=(0, 0, valid[0], max_doc_length - valid[1]))
                    mask = F.pad(batch_token_masks[seg_id + i, sl[0]:sl[1]],
                                 pad=(valid[0], num_token - valid[1])) # should be num_token.

                    t_embs.append(emb)
                    t_masks.append(mask)
                t_embs = torch.stack(t_embs, dim=0)
                t_masks = torch.stack(t_masks, dim=0)

                t_masks = t_masks.sum(dim=0)
                pad_len = max_doc_length - t_masks.shape[0]
                if pad_len > 0:
                    t_masks = F.pad(t_masks, pad=(0, max_doc_length - t_masks.shape[0]), value=1.0)
                doc_token_embs = t_embs.sum(0) / t_masks.unsqueeze(-1)
                batch_token_embs.append(doc_token_embs)
            seg_id += num_seg

        batch_token_embs = torch.stack(batch_token_embs)

        return batch_token_embs, None

    def forward(self, input_ids, input_mask, output_attentions):
        self.load_model()
        if output_attentions:
            output = self.forward_sliding_window_with_attention(input_ids, input_mask, stride=128)
        else:
            output = self.forward_sliding_window_without_attention(input_ids, input_mask, stride=128)
        return output
