import json
import torch
from tqdm import tqdm

from captum.attr import LayerIntegratedGradients


class Predictor:
    def __init__(self, hp, tokenizer):
        self.hp = hp
        self.tokenizer = tokenizer

    def predict_sample(self, model, loader, filename=None):
        """ Generate prediction file with model / dataset(loader) """
        model.eval()

        all_result = {}
        sample_idx = 0
        # Evaluate data for one epoch
        pbar = tqdm(loader, total=len(loader), desc=f'Prediction idx {self.hp.exp_idx}')
        for batch in pbar:
            # Add batch to GPU
            batch = tuple(t.to(self.hp.get_device()) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_perception_scores = batch[:4]

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                losses, logits, outputs = model(b_input_ids, attention_mask=b_input_mask, style_label_ids=b_labels,
                                                perception_scores=None, perception_mask=None, is_eval=True)
                style_logits, perception_logits = logits

            batch_size = b_input_ids.size(0)

            for batch_idx in range(batch_size):
                input_len = b_input_mask[batch_idx].sum()
                style_pred = torch.softmax(style_logits[batch_idx, :input_len], dim=0)
                perception_pred = torch.sigmoid(perception_logits[batch_idx, :input_len])
                perception_target = b_perception_scores[batch_idx, :input_len]

                tokens = self.tokenizer.convert_ids_to_tokens(b_input_ids[batch_idx, :input_len])
                label = b_labels[batch_idx].item()

                sample_pred = perception_pred[:, 0]
                sample_target = perception_target[:, 0]

                if self.hp.negative_perception:
                    negative_pred = perception_pred[:, 1]
                    for i in range(input_len):
                        pos_perception = sample_pred[i]
                        neg_perception = negative_pred[i]
                        if neg_perception > pos_perception:
                            sample_pred[i] = -neg_perception
                            
                    sample_target -= perception_target[:, 1]

                result_dict = {
                    'idx': sample_idx,
                    'text': tokens,
                    'style_prediction': style_pred.tolist(),
                    'style_label': label,
                    'perception_pred': sample_pred.tolist(),
                    'perception_label': sample_target.tolist(),
                }
                all_result[sample_idx] = result_dict

                sample_idx += 1

        save_name = self.hp.pred_file if filename is None else filename
        with open(save_name, 'w') as f:
            json.dump(all_result, f, indent=4, ensure_ascii=False)

        return all_result

    # Captum
    def predict_captum(self, model, loader, filename=None):
        """ Generate Captum prediction file from model / dataset (loader) """
        model.eval()

        all_result = {}
        sample_idx = 0
        # Evaluate data for one epoch
        pbar = tqdm(loader, total=len(loader), desc=f'Prediction captum: idx {self.hp.exp_idx}')
        for idx, batch in enumerate(pbar):
            # Add batch to GPU
            batch = tuple(t.to(self.hp.get_device()) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_perception_scores = batch[:4]

            attributions = self.compute_captum_score(model, b_input_ids, b_input_mask, b_labels)
            
            for batch_idx in range(len(b_input_ids)):
                input_len = b_input_mask[batch_idx].sum()
                attr = attributions[batch_idx, :input_len]
                perception_label = b_perception_scores[batch_idx, :input_len]
                if perception_label.size(-1) == 2:
                    perception_label = perception_label[:, 0] - perception_label[:, 1]
                else:
                    perception_label = perception_label[:, 0]

                tokens = self.tokenizer.convert_ids_to_tokens(b_input_ids[batch_idx, :input_len])

                result_dict = {
                    'idx': sample_idx,
                    'text': tokens,
                    'perception_pred': attr.tolist(),
                    'perception_label': perception_label.tolist()
                }
                all_result[sample_idx] = result_dict
                sample_idx += 1

        save_name = self.hp.pred_file if filename is None else filename
        with open(save_name, 'w') as f:
            json.dump(all_result, f, indent=4, ensure_ascii=False)

        return all_result

    def compute_captum_score(self, model, input_ids, attention_mask, style_label_ids):
        def forward_func(inputs, mask, labels):
            loss, logits, outputs = model(inputs, attention_mask=mask, style_label_ids=labels,
                                          perception_scores=None, perception_mask=None, is_eval=True)
            style_logits = logits[0]
            return style_logits.max(1).values

        ref_token_id = self.tokenizer.pad_token_id
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        batch_size = len(input_ids)
        ref_input_ids = []
        for i in range(batch_size):
            input_len = attention_mask[i].sum().item()

            ref_ids = [cls_token_id] + [ref_token_id] * (input_len - 2) + [sep_token_id] + \
                      (self.hp.max_seq_len - input_len) * [ref_token_id]
            ref_input_ids.append(ref_ids)
            
        ref_input_ids = torch.tensor(ref_input_ids).to(self.hp.get_device())

        model.zero_grad()

        lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
        attributions = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                     additional_forward_args=(attention_mask, style_label_ids),
                                     internal_batch_size=batch_size)

        attributions = attributions.sum(dim=-1)
        if model.num_style_labels == 2:
            attributions[style_label_ids == 1] *= -1
            
        return attributions
