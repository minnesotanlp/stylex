import json
import copy
import torch
import pandas as pd
from torch.utils.data import Dataset

from .data_utils import ScoreToken, process_perception_score


class InputDataPoint(object):
    """
        A single data point consisting of text, text's style label,
        and the human perception score for each word in the text.
    """

    def __init__(self, orig_text, processed_text, style, perception_scores):
        self.orig_text = orig_text
        self.processed_text = processed_text
        self.style = style
        self.perception_scores = perception_scores

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class FeatureDataPoint(object):
    """A single set of features of one data point."""

    def __init__(self, input_ids, attention_mask, style_label_id, perception_scores,
                 perception_mask=None, perception_type=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.style_label_id = style_label_id
        self.perception_scores = perception_scores
        self.perception_mask = perception_mask
        self.perception_type = perception_type  # 0: gt exist 1: no gt

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class StylexDataset(Dataset):
    def __init__(self, features):
        self.len = len(features)
        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        self.all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        self.all_styles = torch.tensor([f.style_label_id for f in features], dtype=torch.long)
        self.all_perception_scores = torch.tensor([f.perception_scores for f in features], dtype=torch.float)
        self.all_perception_mask = torch.tensor([f.perception_mask for f in features], dtype=torch.float)
        self.all_perception_type = torch.tensor([f.perception_type for f in features], dtype=torch.long)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.all_input_ids[idx], self.all_attention_mask[idx],
                self.all_styles[idx], self.all_perception_scores[idx],
                self.all_perception_mask[idx], self.all_perception_type[idx])

    def label_perception_score(self, new_perception_scores):
        self.all_perception_scores[self.all_perception_type == 1] = new_perception_scores[self.all_perception_type == 1]


def read_dataset_from_path(path, split='train'):
    data = pd.read_csv(path, sep="\t")
    if split == 'train':
        gold_labels = data["human_label"].values
        perception_score_list = data["perception_scores"].values
    else:
        gold_labels = data["label"].values
        perception_score_list = None
    orig_text_list = data["orig_text"].values
    processed_text_list = data["processed_text"].values
    instances = []
    for idx, orig_text in enumerate(orig_text_list):
        if split == 'train':
            perception_scores = [float(score) for score in perception_score_list[idx].strip().split(" ")]
        else:
            perception_scores = None

        gold_label = gold_labels[idx]
        if gold_label == 0.5:
            continue

        input_instance = InputDataPoint(orig_text=orig_text_list[idx], processed_text=processed_text_list[idx],
                                        style=gold_label, perception_scores=perception_scores)
        instances.append(input_instance)

    print(f"Num of {split} instances: {len(instances)}")
    return instances


def convert_instances_to_features(hp, instances, tokenizer, data_type):
    # Setting based on the current model type
    score_token = ScoreToken(hp.pretrained_model, tokenizer)
    
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
    cls_sep_perception_id = 0
    pad_perception_id = -100

    features = []

    for idx, instance in enumerate(instances):
        processed_text = instance.processed_text  # already lower-cased and punctuations are removed (,?.!;")
        style = instance.style
        perception_scores = instance.perception_scores
        if hp.sufficiency_test and data_type == 'train':
            word_tokens = processed_text.split(' ')
            perception_scores_for_bert = perception_scores
        else:
            try:
                word_tokens = tokenizer.tokenize(processed_text)
            except TypeError as e:
                continue
        
            # Word tokens
            if not word_tokens:
                word_tokens = [unk_token]
    
            # Perception score
            if data_type == "train":
                perception_scores_for_bert = score_token(processed_text, word_tokens, perception_scores)
            else:
                perception_scores_for_bert = [0] * len(word_tokens)

        # for [CLS] and [SEP]
        num_of_special_tokens = 2
        if len(word_tokens) > (hp.max_seq_len - num_of_special_tokens):
            word_tokens = word_tokens[:(hp.max_seq_len - num_of_special_tokens)]
            perception_scores_for_bert = perception_scores_for_bert[:(hp.max_seq_len - num_of_special_tokens)]

        # Add word tokens and perception scores for feature vector which already contains [CLS]
        token_list = [cls_token] + word_tokens + [sep_token]

        perception_score_list = [cls_sep_perception_id] + perception_scores_for_bert + [cls_sep_perception_id]

        # Convert token to token ids
        input_ids = tokenizer.convert_tokens_to_ids(token_list)

        input_len = len(input_ids)
        attention_mask = [1] * input_len

        # Add padding for input_ids and perception_scores
        padding_length = hp.max_seq_len - input_len
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)

        perception_score_list = perception_score_list + ([pad_perception_id] * padding_length)
        perception_score_list = process_perception_score(perception_score_list,
                                                         hp.perception_mode, hp.negative_perception)

        perception_mask = [0] + ([1] * (input_len - 2)) + ([0] * (padding_length + 1))
        perception_size = 2 if hp.negative_perception else 1
        perception_mask = [[p for _ in range(perception_size)] for p in perception_mask]

        perception_type = int(perception_scores is None or all(score == 0 for score in perception_scores))

        assert len(input_ids) == hp.max_seq_len, "Error with input length {} vs {}".format(len(input_ids), hp.max_seq_len)
        assert len(attention_mask) == hp.max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), hp.max_seq_len)
        assert len(perception_score_list) == hp.max_seq_len, "Error with perception score length {} vs {}".format(
            len(perception_score_list), hp.max_seq_len)
        assert len(perception_mask) == hp.max_seq_len

        feature_vector = FeatureDataPoint(input_ids=input_ids, attention_mask=attention_mask, style_label_id=style,
                                          perception_scores=perception_score_list, perception_mask=perception_mask,
                                          perception_type=perception_type)

        features.append(feature_vector)

    return features


def load_dataset(hp, path, split, style, suffix, tokenizer):
    data_path = path + style + f"_{suffix}.tsv"

    print(f"Load {split} {style} dataset from {data_path} ")

    training_instances = read_dataset_from_path(data_path, split)
    features = convert_instances_to_features(hp, training_instances, tokenizer, data_type=split)

    print(f"Converting {split} data to tensors")

    dataset = StylexDataset(features)

    return dataset
