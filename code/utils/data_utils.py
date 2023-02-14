

class ScoreToken:
    model_dict = {
        'bert-base-uncased': 'default_func',
        'roberta-base': 'roberta_func',
        'xlnet-base-cased': 'default_func',
        't5-base': 'default_func',
    }
    
    def __init__(self, pretrained_model, tokenizer):
        self.tokenizer = tokenizer
        
        if pretrained_model in self.model_dict:
            self.score_func = getattr(self, self.model_dict[pretrained_model])
        else:
            self.score_func = self.default_func

    def register_func(self, model, func):
        if isinstance(func, str):
            try:
                getattr(self, func)
            except AttributeError:
                raise AttributeError('Invaild function is registered')
            
            self.model_dict[model] = func
        else:
            func_name = func.__name__
            setattr(self, func_name, func)
            self.model_dict[model] = func_name
        
    def __call__(self, *args, **kwargs):
        return self.score_func(*args, **kwargs)

    def default_func(self, orig_text, tokens, scores):
        orig_text_list = orig_text.split(' ')
        assert len(orig_text_list) == len(scores)
        
        token_scores = []
        for i, word in enumerate(orig_text_list):
            word_tokens = self.tokenizer.tokenize(word)
            score = scores[i]
            
            token_scores.extend([score for _ in word_tokens])
        
        assert len(token_scores) == len(tokens)
        
        return token_scores
    
    @staticmethod
    def roberta_func(orig_text, tokens, scores):
        # Check valid score
        orig_text_list = orig_text.split(' ')
        assert len(orig_text_list) == len(scores)
        
        token_scores = []
        score_idx = 0
        for token in tokens:
            if token.startswith('Ġ'):
                score_idx += 1
            
            assert score_idx < len(scores)
            token_scores.append(scores[score_idx])
        
        return token_scores


def process_perception_score(perception_score, perception_mode, negative_perception):
    pad_perception_score = -100
    
    processed_perception = []
    for score in perception_score:
        if score != pad_perception_score:
            if perception_mode == 'majority':
                score = round(score)
            elif perception_mode == 'full':
                score = int(score)
            elif perception_mode == 'min1':
                score = 0 if score == 0 else score / abs(score)
        
        if negative_perception:
            if score == pad_perception_score:
                new_score = [score, score]
            else:
                new_score = [0, abs(score)] if score < 0 else [score, 0]
        else:
            new_score = [score]
        
        processed_perception.append(new_score)
    
    return processed_perception