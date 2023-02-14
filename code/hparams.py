import os
import json
import torch
import glob
import dataclasses
from argparse import Namespace
from typing import Optional
from dataclasses import dataclass

@dataclass
class Hparams:
    # General hyperparameters
    pretrained_model: str = 'bert-base-uncased' # roberta-base, t5-base, xlnet-base-cased, bert-base-uncased
    batch_size: int = 16
    train_epoch: int = 5
    dropout_rate: float = 0.1
    n_gpu: int = 1
    seed: int = 23
    device: str = 'cuda:0'
    patience: int = 10
    early_stopping: bool = False
    
    # Dataset & Preprocessing
    style: str = 'offensiveness'
    
    # Preprocessing configuration
    perception_mode: str = 'majority'   # majority | min1 | full | ''
    max_seq_len: int = 512

    # Dataset configuration
    label: bool = 'pseudo'  # semi-supervised | pseudo | captum | baseline | None
    pseudo_label_penalty: float = 0.5
    
    # Only valid when label is None
    dataset_suffix: str = 'train_captum'   # 'train_orig' | 'train_combined' | 'hummingbird' || Only valid when label is None
    split_train: bool = False    # Split train set for dev set or use predefined dev set
    
    split_train_ratio: float = 0.8

    # Pseudo labeling with smaller data
    pseudo_label_train_proportion: float = 0.25
    
    # when label == captum
    captum_idx: int = 249
    
    # Low Resource setting
    data_usage: float = 1  # use [data_usage] x 100% of original dataset
    
    sufficiency_test: bool = False
    
    # Perception prediction options
    use_perception_logits: bool = True  # pwi, concat perception with CLS hidden for style classification
    get_perception_loss: bool = True   # wi, compute & backward perception loss
    perception_lambda: float = 0.05       # weight_coef, loss = style_loss + perception_lambda * perception_loss
    perception_method: str = 'aggregate'    # concat_score [CLS; perception_logit] | concat_hidden [CLS; weighted_summed_hidden] | aggregate [aggregated_hidden]
    perception_loss_fcn: str = 'bce'    # bce, mse
    perception_only: bool = False
    
    # Set during initialization
    negative_perception: bool = None
    logdir: str = None
    data_path: str = None
    output_dir: str = None
    exp_idx: int = None

    def __init__(self,
                 args: Optional[Namespace] = None,
                 pseudo_hparam: Optional[bool] = False,
                 **kwargs):

        exp_idx = [int(i.split('/')[1]) for i in glob.glob('logdir/*') if os.path.isdir(i)]
        self.exp_idx = max(exp_idx + [-1])
        if not pseudo_hparam:
            self.exp_idx += 1

        if args is not None:
            for key, val in vars(args).items():
                if val is not None:
                    setattr(self, key, val)

        for key, val in kwargs.items():
            if val is not None:
                setattr(self, key, val)
                
        if pseudo_hparam:
            self.output_dir = f'pseudo_labels/checkpoints/{self.exp_idx}/'
        else:
            self.logdir = f"logdir/{self.exp_idx}"
            self.output_dir = f'checkpoints/{self.exp_idx}/'

        self.data_path = f'data/{self.style}/'
        self.negative_perception = self.style in ['politeness', 'sentiment']

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs('./predictions', exist_ok=True)
        
    def to_dict(self):
        return dataclasses.asdict(self)

    def save(self):
        with open(os.path.join(self.output_dir, "hparams.json"), 'w') as f:
            json.dump(self.to_dict(), f)
            
    @classmethod
    def load(cls, model_dir):
        hp_path = os.path.join(model_dir, "hparams.json")
        with open(hp_path, 'r') as f:
            obj_dict = json.load(f)
            
        hp = cls(**obj_dict)
        return hp
    
    def get_device(self):
        return torch.device(self.device if torch.cuda.is_available() and self.n_gpu > 0 else "cpu")
