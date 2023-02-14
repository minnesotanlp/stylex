import torch.nn as nn

class PerceptionPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.):
        super(PerceptionPredictor, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class StyleClassifier(nn.Module):
    def __init__(self, input_dim, num_style_labels, dropout_rate=0.):
        super(StyleClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)        
        self.linear = nn.Linear(input_dim, num_style_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
