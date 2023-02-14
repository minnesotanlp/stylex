import json
import random
import torch
import numpy as np

from transformers import AdamW


def set_seed(n_gpu, the_seed):
    random.seed(the_seed)
    np.random.seed(the_seed)
    torch.manual_seed(the_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(the_seed)


def get_optimizer(model, lr=2e-5):
    # BERT fine-tuning parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)  # , warmup=.1)
    return optimizer


def get_captum_label(idx, max_len=512, negative_label=False):
    with open(f'predictions/{idx}_orig_train_captum.json', 'r') as f:
        captum_data = json.load(f)
    
    captum_labels = []
    for i in range(len(captum_data)):
        label = captum_data[str(i)]['perception_pred']
        label = label + [0] * (max_len - len(label))
        
        processed_label = []
        for score in label:
            if negative_label:
                if score > 0:
                    processed_label.append([score, 0])
                else:
                    processed_label.append([0, abs(score)])
            else:
                processed_label.append([score])
        
        captum_labels.append(processed_label)
        
    captum_labels = torch.tensor(captum_labels)
    
    return captum_labels
    