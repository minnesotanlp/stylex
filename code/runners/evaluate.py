import torch

from sklearn.metrics import f1_score

class Evaluator:
    def __init__(self, hp):
        self.hp = hp

    def eval(self, model, dev_dataloader, mode='Dev'):
        model.eval()
    
        # Tracking variables
        perception_f1_score, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_preds, all_labels = [], []
    
        # Evaluate data for one epoch
        for batch in dev_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.hp.get_device()) for t in batch)
    
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_perception_scores = batch[:4]
    
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                losses, logits, outputs = model(b_input_ids, attention_mask=b_input_mask, style_label_ids=b_labels,
                                                perception_scores=None, perception_mask=None, is_eval=True)
                style_logits, perception_logits = logits

            # Compute style accuracy & perception F1 score
            style_preds = torch.argmax(style_logits, dim=1)
            step_eval_accuracy = torch.eq(style_preds, b_labels).to(torch.float32).mean()

            all_preds.extend(style_preds.cpu().tolist())
            all_labels.extend(b_labels.cpu().tolist())

            perception_preds = torch.round(torch.sigmoid(perception_logits)).detach().cpu().numpy()
            perception_scores = torch.round(b_perception_scores).cpu().numpy()
            step_f1_score = 0
            for i in range(len(b_labels)):
                input_len = b_input_mask[i].sum()
                y_true = perception_scores[i, :input_len]
                y_pred = perception_preds[i, :input_len]
                tmp_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
                step_f1_score += tmp_f1

            perception_f1_score += step_f1_score / len(b_labels)
            eval_accuracy += step_eval_accuracy
            nb_eval_steps += 1
    
        final_acc = eval_accuracy / nb_eval_steps
        final_perception_f1 = perception_f1_score / nb_eval_steps
        final_style_f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(f"{mode} | Style Accuracy: {final_acc:.5f} | Perception F1: {final_perception_f1:.5f} | Style F1: {final_style_f1:.5f}")
    
        return final_acc, final_perception_f1, final_style_f1
            

