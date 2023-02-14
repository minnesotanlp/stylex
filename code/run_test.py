import argparse

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.utils import set_seed
from utils.data_loader import load_dataset
from model.modified_bert import StyLEx
from hparams import Hparams
from runners import Tester, Predictor


def run_test(args):
    model_dir = f'checkpoints/{args.idx}/'
    hp = Hparams.load(model_dir)
    print(hp)

    if args.run_prediction or args.predict_captum:
        set_seed(hp.n_gpu, hp.seed)

        tokenizer = AutoTokenizer.from_pretrained(hp.pretrained_model, do_lower_case=True)
        if hp.pretrained_model == 't5-base':
            tokenizer.cls_token = '<s>'
            tokenizer.sep_token = '</s>'
    
        if args.dataset == 'hummingbird':
            data = load_dataset(hp, hp.data_path, split='train', style=hp.style, suffix='hummingbird', tokenizer=tokenizer)
            _, test_data = train_test_split(data, train_size=hp.split_train_ratio, random_state=23,
                                            shuffle=True)
        elif args.dataset == 'orig':
            test_data = load_dataset(hp, hp.data_path, split='test', style=hp.style, suffix='test',
                                     tokenizer=tokenizer)
        elif args.dataset == 'orig_dev':
            test_data = load_dataset(hp, hp.data_path, split='dev', style=hp.style, suffix='dev',
                                     tokenizer=tokenizer)
        elif args.dataset == 'orig_train':
            test_data = load_dataset(hp, hp.data_path, split='train', style=hp.style, suffix='train_orig',
                                     tokenizer=tokenizer)
        elif args.dataset == 'ood':   # OOD
            test_data = load_dataset(hp, hp.data_path, split='ood', style=hp.style, suffix='ood', tokenizer=tokenizer)
        elif args.dataset == 'ood2':   # OOD
            test_data = load_dataset(hp, hp.data_path, split='ood', style=hp.style, suffix='ood_new', tokenizer=tokenizer)
        else:
            raise RuntimeError('Wrong dataset')
        
        print(f'{args.dataset} loaded | size: {len(test_data)}')
        
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
    
        model = StyLEx.from_pretrained(model_dir, hp=hp).to(hp.get_device())
    
        predictor = Predictor(hp, tokenizer)
    
        print('Generating Prediction Files...')
        if args.run_prediction:
            pred_filename = f'predictions/{args.idx}_{args.dataset}.json'
            predictor.predict_sample(model, test_dataloader, pred_filename)
    
        if args.predict_captum:
            captum_filename = f'predictions/{args.idx}_{args.dataset}_captum.json'
            predictor.predict_captum(model, test_dataloader, captum_filename)

    tester = Tester()

    print('Running Test...')
    if args.run_vis:
        prediction_path1 = f'predictions/{args.idx}_{args.dataset}.json'
        
        if args.idx2 is None:
            sample_name = f'html/{args.idx}_{args.dataset}.html'
            prediction_path2 = None
        else:
            sample_name = f'html/{args.idx}_{args.idx2}_{args.dataset}.html'
            prediction_path2 = f'predictions/{args.idx2}_{args.dataset}.json'
        
        if args.captum_idx is not None:
            captum_filename = f'predictions/{args.captum_idx}_{args.dataset}_captum.json'
            sample_name = sample_name.replace('.html', f'_c{args.captum_idx}.html')
        else:
            captum_filename = None
            
        baseline_path = None if args.baseline_idx is None else f'predictions/{args.baseline_idx}_{args.dataset}.json'
            
        tester.print_test(sample_name,
                          prediction_path1, prediction_path2=prediction_path2,
                          name1=args.idx, name2=args.idx2,
                          num_sample=args.num_samples,
                          captum=captum_filename, ground_truth=args.ground_truth,
                          baseline_path=baseline_path, num_changed=args.num_changed)

    if args.run_corr:
        tester.correlation_test(args.idx, suffix=args.dataset)

    if args.run_acc:
        tester.accuracy_test(args.idx, suffix=args.dataset)
        
    if args.run_captum_corr:
        tester.correlation_test(args.idx, suffix='hummingbird_captum')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', required=True, help='Experiment idx')
    parser.add_argument('--dataset', type=str, default='orig', # choices=['hummingbird', 'orig', 'ood', 'orig_train', 'train_captum'],
                        help='Test Dataset for test')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for test')
    parser.add_argument('--run_prediction', action='store_true', help='Generate prediction file or not')
    parser.add_argument('--predict_captum', action='store_true', help='captum record filename')

    # Test args
    parser.add_argument('--run_acc', action='store_true', help='Run accuracy test?')
    parser.add_argument('--run_corr', action='store_true', help='Run correlation test?')
    parser.add_argument('--run_captum_corr', action='store_true', help='Run correlation of captum score?')

    # print test args
    parser.add_argument('--run_vis', action='store_true', help='Visualize or not?')
    parser.add_argument('--idx2', default=None, help='Second experiment idx for paired test')
    parser.add_argument('--captum_idx', help='Captum idx for visualize')
    parser.add_argument('--num_samples', type=int, default=300, help='Number of samples to visualize')
    parser.add_argument('--ground_truth', action='store_true', help='visualize GT')
    parser.add_argument('--baseline_idx', default=None, help='baseline idx for reference')
    parser.add_argument('--num_changed', type=int, default=10, help='num_changed for baseline comparison')

    args = parser.parse_args()

    run_test(args)