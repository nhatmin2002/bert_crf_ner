#!/usr/bin/env python3

import csv
import torch
import logging
import os
import copy
import json
import csv


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError

    def get_dev_examples(self, data_dir):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    @classmethod
    def _read_text(self, input_file, delim='\t', word_idx=0, label_idx=-1):
        tokens, labels = [], []
        tmp_tok, tmp_lab = [], []
        label_set = []
        lines = []
        
        with open(input_file, 'r', encoding='utf8') as reader:
            for line in reader:
                if "IMGID" in line: 
                    a=1
                else:
                    line = line.strip()
                    cols = line.split(delim)
                    if len(cols) < 2:
                        if len(tmp_tok) > 0:
                            tokens.append(tmp_tok)
                            labels.append(tmp_lab)
                            lines.append({"words": tmp_tok, "labels": tmp_lab})
                        tmp_tok = []
                        tmp_lab = []
                    else:
                        tmp_tok.append(cols[word_idx])
                        tmp_lab.append(cols[label_idx])
                        label_set.append(cols[label_idx])
        print(lines[0])
        
        return lines


class InputExample(object):
    def __init__(self, guid, text_a, labels):
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self) -> str:
        return str(self.to_json_string())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self) -> str:
        return str(self.to_json_string())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def collate_fn(batch):
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(
        torch.stack, zip(*batch)
    )
    max_len = torch.max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


def convert_example_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = example.text_a
        label_ids = [label_map[x] for x in example.labels]
        if len(tokens) > max_seq_length - 2:  # cls+sep
            tokens = tokens[: (max_seq_length - 2)]
            label_ids = label_ids[: (max_seq_length - 2)]

        tokens += [sep_token]
        label_ids += [label_map["O"]]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [label_map["O"]] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_list)
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token] * padding_length
            label_ids += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids, input_mask, input_len, segment_ids, label_ids)
        )
    return features


class CnerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_text(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_text(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_text(os.path.join(data_dir, "test.txt")), "test"
        )

    def get_labels(self):
        return [
            'X',
            'O', 'B-QUANTITY-NUM', 'B-EVENT-CUL', 'B-PRODUCT-AWARD', 'B-MISCELLANEOUS', 'I-QUANTITY', 'B-EVENT-GAMESHOW', 'B-QUANTITY-AGE', 'I-EVENT-NATURAL', 'I-DATETIME-DURATION', 'B-DATETIME-DURATION', 'I-URL', 'B-PERSONTYPE', 'B-ORGANIZATION', 'B-QUANTITY-ORD', 'I-QUANTITY-NUM', 'I-DATETIME-SET', 'I-EVENT', 'I-ORGANIZATION', 'I-ORGANIZATION-SPORTS', 'I-LOCATION-GPE', 'I-DATETIME', 'I-LOCATION-STRUC', 'B-SKILL', 'I-QUANTITY-TEM', 'B-LOCATION-GEO', 'I-DATETIME-TIME', 'I-PHONENUMBER', 'B-LOCATION-STRUC', 'I-PRODUCT-AWARD', 'I-QUANTITY-PER', 'B-QUANTITY-TEM', 'I-ORGANIZATION-MED', 'B-PERSON', 'I-PRODUCT-COM', 'I-PERSONTYPE', 'B-DATETIME-DATE', 'B-QUANTITY-DIM', 'I-IP', 'B-QUANTITY', 'B-EVENT-NATURAL', 'B-IP', 'B-LOCATION-GPE', 'I-EVENT-GAMESHOW', 'I-ADDRESS', 'B-PRODUCT-COM', 'I-DATETIME-DATERANGE', 'I-PERSON', 'B-EMAIL', 'I-EMAIL', 'B-ORGANIZATION-SPORTS', 'I-EVENT-SPORT', 'I-QUANTITY-AGE', 'B-DATETIME-DATERANGE', 'B-PRODUCT', 'B-ORGANIZATION-STOCK', 'B-LOCATION-GPE-GEO', 'B-DATETIME-TIME', 'B-ORGANIZATION-MED', 'B-LOCATION', 'B-DATETIME-SET', 'I-DATETIME-DATE', 'B-EVENT', 'I-QUANTITY-DIM', 'I-QUANTITY-CUR', 'B-DATETIME', 'I-DATETIME-TIMERANGE', 'B-URL', 'B-EVENT-SPORT', 'I-QUANTITY-ORD', 'I-EVENT-CUL', 'I-PRODUCT', 'I-MISCELLANEOUS', 'I-SKILL', 'I-LOCATION', 'B-DATETIME-TIMERANGE', 'B-PRODUCT-LEGAL', 'B-ADDRESS', 'B-QUANTITY-PER', 'I-ORGANIZATION-STOCK', 'I-LOCATION-GEO', 'B-PHONENUMBER', 'B-QUANTITY-CUR', 
            'I-PRODUCT-LEGAL',

            '[START]',
            '[END]',
        ]

    def _create_examples(self, lines, set_type):
        # {'words': [word1, word2, ...], 'labels': [label1, label2, ...]}
        examples = []

        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line["words"]
            labels = []
            for x in line["labels"]:
                # if "M-" in x:
                #     labels.append(x.replace("M-", "I-"))
                # elif "E-" in x:
                #     labels.append(x.replace("E-", "I-"))
                # else:
                labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples


def get_entities(seq, id2label, markup="bios"):
    assert markup in ["bio", "bios"]
    if markup == "bio":
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


def get_entity_bio(seq, id2label):
    chunks = []
    chunk = [-1, -1, -1]  # label, b-index, i-index
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split("-")[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith("I-") and chunk[1] != -1:
            _type = tag.split("-")[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bios(seq, id2label):
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split("-")[1]
            chunks.append(chunk)
            chunk = [-1, -1, -1]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split("-")[1]
        elif tag.startswith("I-") and chunk[1] != -1:
            _type = tag.split("-")[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


if __name__ == "__main__":
    cner = CnerProcessor()
    data_dir = "./data"
    cner.get_train_examples(data_dir)
