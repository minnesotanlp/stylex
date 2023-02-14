# StyLEx: Explaining Style Using Human Lexical Annotations

This repository provides datasets and code for preprocessing, training and testing models for 
style classification with human lexical annotations 
with the official Hugging Face implementation of the following paper:

> [StyLEx: Explaining Style Using Human Lexical Annotations](https://arxiv.org/abs/2210.07469) \
> [Shirley A. Hayati](https://www.shirley.id/), 
> [Kyumin Park](https://github.com/Kyumin-Park), 
> [Dheeraj Rajagopal](https://dheerajrajagopal.github.io/), 
> [Lyle Ungar](https://www.cis.upenn.edu/~ungar/), [Dongyeop Kang](https://github.com/dykang)\
> [EACL 2023](https://2023.eacl.org/)

## Installation
The following command installs all necessary packages:
```shell
pip install -r requirements.txt
```
The project was tested using Python 3.8.


## Models
#### Model Checkpoints

Model | Style      | F1 (Orig) | F1 (Hummingbird) | F1 (OOD)
----- |------------|-----------|------------------| -----
[BERT](https://huggingface.co/kyuminpark/stylex-politeness) | Politeness | 0.96      | 0.91             | 0.87
[BERT](https://huggingface.co/kyuminpark/stylex-sentiment) | Sentiment  | 0.67      | 0.91             | 0.75
[BERT](https://huggingface.co/kyuminpark/stylex-joy) | Joy        | 0.88      | 0.92             | 0.73
[BERT](https://huggingface.co/kyuminpark/stylex-sadness) | Sadness    | 0.89      | 0.94             | 0.78
[BERT](https://huggingface.co/kyuminpark/stylex-fear) | Fear       | 0.96      | 0.92             | 0.80
[BERT](https://huggingface.co/kyuminpark/stylex-disgust) | Disgust    | 0.86      | 0.81             | 0.74
[BERT](https://huggingface.co/kyuminpark/stylex-anger) | Anger      | 0.89      | 0.82             | 0.78
[BERT](https://huggingface.co/kyuminpark/stylex-offensiveness) | Offensiveness | 0.97      | 0.87             | 0.88


## Citation
If you find this work useful for your research, please cite our papers:
```
@article{hayati2022stylex,
  title={StyLEx: Explaining Styles with Lexicon-Based Human Perception},
  author={Hayati, Shirley Anugrah and Park, Kyumin and Rajagopal, Dheeraj and Ungar, Lyle and Kang, Dongyeop},
  journal={arXiv preprint arXiv:2210.07469},
  year={2022}
}
```


