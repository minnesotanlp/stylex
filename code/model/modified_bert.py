import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from .module import StyleClassifier, PerceptionPredictor, Pooler


class StyLEx(nn.Module):
    def __init__(self, config, hp, num_style_labels=2, bert_model=None):
        super(StyLEx, self).__init__()
        self.hp = hp
        self.config = config

        self.num_style_labels = num_style_labels

        self.bert = AutoModel.from_config(config) if bert_model is None else bert_model
        if self.hp.pretrained_model == 't5-base':
            self.bert = self.bert.get_encoder()
        self.pooler = Pooler(config=config)

        self.perception_size = 2 if self.hp.negative_perception else 1

        if self.hp.use_perception_logits:
            if self.hp.perception_method == 'aggregate':
                style_hidden_size = config.hidden_size + self.perception_size
            elif self.hp.perception_method == 'concat_hidden':
                style_hidden_size = config.hidden_size + config.hidden_size * self.perception_size
            else:   # concat_score
                style_hidden_size = config.hidden_size + self.hp.max_seq_len * self.perception_size  # config.hidden_size = 768 + 512 * K
        else:
            style_hidden_size = config.hidden_size
        
        self.style_classifier = StyleClassifier(style_hidden_size, self.num_style_labels, self.hp.dropout_rate)
        self.perception_predictor = PerceptionPredictor(config.hidden_size, self.perception_size, self.hp.dropout_rate)
        
        self.style_loss_fct = nn.CrossEntropyLoss()

        if self.hp.perception_loss_fcn == 'mse':
            self.perception_loss_fct = nn.MSELoss(reduction='none')  # Use 0 or 1 for token level, take it if >= 2 people agree
        elif self.hp.perception_loss_fcn == 'bce':
            self.perception_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, input_ids, attention_mask, style_label_ids=None, perception_scores=None,
                perception_mask=None, perception_type=None, is_eval=False):
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)

        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)

        # Compute perception logits
        perception_logits = self.perception_predictor(sequence_output)  # B * 512 (maximum length of the tokens) * 1 # torch.Size([B, 512, 1])
        
        if self.hp.use_perception_logits:
            if self.hp.perception_method == 'aggregate':
                if perception_mask is None:
                    tmp_perception_mask = attention_mask.clone()
                    for i in range(len(input_ids)):
                        input_len = attention_mask[i].sum()
                        tmp_perception_mask[i, 0] = 0
                        tmp_perception_mask[i, input_len - 1] = 0
                else:
                    tmp_perception_mask = perception_mask[:, :, 0]

                aggregated_perception = torch.cat([sequence_output, perception_logits], dim=-1)   # B * 512 * (768 + 1or2)
                aggregated_perception *= tmp_perception_mask.unsqueeze(-1)
                
                aggregated_perception = torch.max(aggregated_perception, dim=1)[0] # B * (768 + 1or2)
                style_logits = self.style_classifier(aggregated_perception)
            
            else:
                # Mask perception score to exclude pad
                mask = torch.stack([attention_mask for _ in range(self.perception_size)], dim=-1) if perception_mask is None else perception_mask
                masked_perception_logits = perception_logits * mask # B * 512 * 2
                
                if self.hp.perception_method == 'concat_hidden':
                    # [CLS; summed_hidden]; summed_hidden = sum(perception * hidden for each token)
                    hidden_list = []
                    for i in range(self.perception_size):
                        weighted_hidden = torch.mul(sequence_output.transpose(0, -1),
                                                    masked_perception_logits[:, :, i].transpose(-1, -2)).transpose(0, -1)
                        summed_hidden = torch.sum(weighted_hidden, dim=1)
                        hidden_list.append(summed_hidden)
                    perception_hidden = torch.cat(hidden_list, dim=1)
                else:   # concat_score
                    # [CLS; perception_logits]
                    perception_hidden = torch.flatten(masked_perception_logits, start_dim=1) # B * max_len * n_class -> B * (max_len * n_class)
    
                # use pooled_output concatenated with word_importance_logits to predict style
                concatenated_logits = torch.cat((pooled_output, perception_hidden), dim=1)
                style_logits = self.style_classifier(concatenated_logits)

        else:
            style_logits = self.style_classifier(pooled_output)
            
        logits = (style_logits, perception_logits)
            
        # Style loss calculation
        total_loss = 0
        
        # ------------ Sentence level -------------
        assert style_label_ids is not None  # if adding style_label_ids is None, then insert if statement here
        style_loss = self.style_loss_fct(style_logits, style_label_ids)
        total_loss += style_loss
        losses = (total_loss, )

        # ------------ Perception level -------------
        if self.hp.get_perception_loss and not is_eval:
            assert perception_mask is not None
            masked_perception_scores = perception_scores * perception_mask
            perception_loss_matrix = self.perception_loss_fct(perception_logits, masked_perception_scores)
            perception_loss = torch.sum(perception_loss_matrix * perception_mask, dim=1)

            if perception_type is not None:
                if self.hp.label == 'pseudo':
                    perception_loss[perception_type == 1] *= self.hp.pseudo_label_penalty
                elif self.hp.label == 'semi-supervised':
                    perception_loss[perception_type == 1] *= 0

            mask_sum = torch.sum(perception_mask, dim=1)
            mask_sum = torch.maximum(mask_sum, torch.ones_like(mask_sum))
            perception_loss = perception_loss / mask_sum
            
            perception_loss = perception_loss.sum()
                
            total_loss += self.hp.perception_lambda * perception_loss  # use a regularization with very small lambda rather than weight for loss
            losses += (style_loss, perception_loss)
        
        return losses, logits, outputs

    def save_pretrained(self, output_dir=None):
        if output_dir is None:
            output_dir = self.hp.output_dir
        
        self.config.save_pretrained(output_dir)
        
        state_dict = self.state_dict()
        output_model_file = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(state_dict, output_model_file)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, hp, config=None):
        # Load config
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Load model
        if os.path.isdir(pretrained_model_name_or_path):
            archive_file = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
            state_dict = torch.load(archive_file, map_location='cpu')

            model = cls(config, hp)
            model.load_state_dict(state_dict)
        else:
            bert_model = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config)
            model = cls(config, hp, bert_model=bert_model)
        
        return model
