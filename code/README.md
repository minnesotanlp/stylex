# StyLEx train/test code

### Training Model
1. Go to code directory
2. Fix hparams.py - check hparams.py
3. Run following script:
```shell
python run_train.py
```

### Testing Model
```shell
python run_test.py --idx [idx] --dataset [hummingbird/orig] [--run_prediction] [--predict_captum] \
[--run_acc] [--run_corr] [--run_vis] [--idx2 idx2] [--captum_idx idx] [--num_samples n_samples]
```