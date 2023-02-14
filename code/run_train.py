import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from runners import PseudoLabeler, Evaluator, Trainer
from utils.data_loader import load_dataset
from model.modified_bert import StyLEx
from hparams import Hparams
from utils.utils import set_seed, get_optimizer, get_captum_label


def get_dataset(hp, args):
    # Load tokenizer
    print(f'Loading {hp.pretrained_model} tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(hp.pretrained_model, do_lower_case=True)
    if hp.pretrained_model == 't5-base':
        tokenizer.cls_token = '<s>'
        tokenizer.sep_token = '</s>'
    
    # Train dataset
    if hp.label is not None:    # pseudo, captum, semi-supervised, baseline
        # Orig label
        if hp.label == 'pseudo':
            labels = PseudoLabeler(tokenizer, args).pseudo_label()
        elif hp.label == 'captum':
            labels = get_captum_label(hp.captum_idx, max_len=hp.max_seq_len, negative_label=hp.negative_perception)
        else:
            labels = None
            
        # Orig dataset
        orig_data = load_dataset(hp, hp.data_path, split='train', style=hp.style, suffix='train_orig',
                                 tokenizer=tokenizer)
        if hp.label in ['pseudo', 'captum']:
            orig_data.label_perception_score(labels)

        if hp.data_usage < 1:
            orig_data, _ = train_test_split(orig_data, train_size=hp.data_usage, random_state=23, shuffle=True)
            
        # Hummingbird dataset
        pl_data = load_dataset(hp, hp.data_path, split='train', style=hp.style, suffix='hummingbird',
                               tokenizer=tokenizer)
        pl_data, _ = train_test_split(pl_data, train_size=hp.split_train_ratio, random_state=23, shuffle=True)

        # Concat dataset
        train_data = ConcatDataset([pl_data, orig_data])
    else:
        train_data = load_dataset(hp, hp.data_path, split="train", style=hp.style, suffix=hp.dataset_suffix,
                                  tokenizer=tokenizer)
        
    # Dev dataset
    if hp.label is None and hp.split_train:
        train_data, dev_data = train_test_split(train_data, train_size=hp.split_train_ratio, random_state=23,
                                                shuffle=True)
    else:
        dev_data = load_dataset(hp, hp.data_path, split="dev", style=hp.style, suffix='dev', tokenizer=tokenizer)

    print(f'Training dataset loaded | train: {len(train_data)} | dev: {len(dev_data)}')
    
    return train_data, dev_data
    

def train(hp, args):
    # Logger
    writer = SummaryWriter(hp.logdir)
    
    # Seed & HP save
    set_seed(hp.n_gpu, hp.seed)
    hp.save()

    train_data, dev_data = get_dataset(hp, args)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=hp.batch_size)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=hp.batch_size)
    
    print("Finish processing data")
    
    # Load model
    model = StyLEx.from_pretrained(hp.pretrained_model, hp=hp).to(hp.get_device())
    
    # Load evaluator
    evaluator = Evaluator(hp)
    
    # BERT fine-tuning parameters
    optimizer = get_optimizer(model)
    
    criterion = 'perception_f1' if args.only_pseudo_labeler else 'style_f1'
    best_dev_acc, best_epoch = Trainer().run_train(hp, model, optimizer, evaluator,
                                                   train_dataloader, dev_dataloader, writer, criterion=criterion)
    
    writer.add_hparams(hp.to_dict(),
                       {'hparams/dev_acc': best_dev_acc, 'hparams/best_epoch': best_epoch})
    
    writer.flush()
    writer.close()
    
    print(f'{hp.exp_idx}\t{best_dev_acc:.5f}\t{best_epoch}')


if __name__ == '__main__':
    def str2none(s):
        return None if s == 'none' else s
    
    def str2bool(s):
        return s.lower() == 'true'
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, help='pretrained model')
    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--exp_idx', type=int, help='experiment idx')
    parser.add_argument('--style', type=str, help='experiment style')
    parser.add_argument('--label', type=str2none, help='experiment label',
                        choices=['baseline', 'pseudo', 'semi-supervised', 'none', 'captum'])
    parser.add_argument('--use_perception_logits', type=str2bool, help='use perception for style logit')
    parser.add_argument('--get_perception_loss', type=str2bool, help='get perception loss?')
    parser.add_argument('--perception_lambda', type=float, help='perception lambda')
    parser.add_argument('--perception_method', type=str, help='perception method')
    parser.add_argument('--perception_mode', type=str, help='perception mode')
    parser.add_argument('--data_usage', type=float, help='data usage ratio')
    parser.add_argument('--perception_loss_fcn', type=str, help='perception loss function', choices=['bce', 'mse'])
    parser.add_argument('--dataset_suffix', type=str, help='dataset suffix')
    parser.add_argument('--only_pseudo_labeler', action='store_true')
    
    args = parser.parse_args()
    
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Total number of GPU: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    hp = Hparams(args)
    print(hp)
    
    train(hp, args)
