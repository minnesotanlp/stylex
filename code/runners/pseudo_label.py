import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from hparams import Hparams
from utils.data_loader import load_dataset
from utils.utils import get_optimizer
from model.modified_bert import StyLEx
from runners.evaluate import Evaluator
from runners.train import Trainer
    

class PseudoLabeler:
    def __init__(self, tokenizer, args):
        self.hp = self.get_hparams(args)
        self.tokenizer = tokenizer

    def train_pseudo_labeler(self):
        train_data = load_dataset(self.hp, self.hp.data_path,
                                  split="train", style=self.hp.style, suffix='hummingbird', tokenizer=self.tokenizer)
        train_data, dev_data = train_test_split(train_data, train_size=self.hp.split_train_ratio,
                                                random_state=23, shuffle=True)
        
        train_data = train_data[:int(len(train_data) * self.hp.pseudo_label_train_proportion)]

        print(f'Pseudo Labeling dataset loaded | train: {len(train_data)} | dev: {len(dev_data)}')

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.hp.batch_size)
        dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=self.hp.batch_size)

        model = StyLEx.from_pretrained(self.hp.pretrained_model, hp=self.hp).to(self.hp.get_device())

        # BERT fine-tuning parameters
        optimizer = get_optimizer(model)
        evaluator = Evaluator(self.hp)

        Trainer().run_train(self.hp, model, optimizer, evaluator, train_dataloader, dev_dataloader, criterion='perception_f1')

    def pseudo_label(self, run_train=True):
        if run_train:
            self.train_pseudo_labeler()

        label_data = load_dataset(self.hp, self.hp.data_path,
                                  split="train", style=self.hp.style, suffix='train_orig', tokenizer=self.tokenizer)
        label_dataloader = DataLoader(label_data, shuffle=False, batch_size=self.hp.batch_size)

        model = StyLEx.from_pretrained(self.hp.output_dir, hp=self.hp).to(self.hp.get_device())
        model.eval()

        # Tracking variables
        all_preds = []

        # Evaluate data for one epoch
        for batch in label_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.hp.get_device()) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, _ = batch[:4]

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                losses, logits, outputs = model(b_input_ids, attention_mask=b_input_mask, style_label_ids=b_labels,
                                                perception_scores=None, perception_mask=None, is_eval=True)
                perception_logits = logits[1]

            # Move logits and labels to CPU
            preds = torch.sigmoid(perception_logits).detach().cpu()
            all_preds.append(preds)

        all_preds = torch.cat(all_preds, dim=0)
        return all_preds

    @staticmethod
    def get_hparams(args):
        hp = Hparams(args=args, pseudo_hparam=True, batch_size=16, train_epoch=50)
        return hp
