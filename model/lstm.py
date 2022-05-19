import torch


class Gate(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, actvation="sigmoid"):
        super(Gate, self).__init__()

        actvation = actvation.lower()
        assert actvation in ["sigmoid", "tanh"], f"'activation' ({actvation}) not in ['sigmoid', 'tanh']"

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear_input = torch.nn.Linear(
            in_features=self.in_dim,
            out_features=self.out_dim
        )

        self.linear_hidden = torch.nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.out_dim
        )

        if actvation == "sigmoid":
            self.actvation = torch.nn.Sigmoid()
        elif actvation == "tanh":
            self.actvation = torch.nn.Tanh()
    
    def forward(self, x, h):
        return  self.actvation(self.linear_input(x) + self.linear_hidden(h))

class FFN(torch.nn.Module):
    def __init__(self, in_dim, ff_dim):
        super(FFN, self).__init__()
        self.up = torch.nn.Linear(in_features=in_dim, out_features=ff_dim)
        self.down = torch.nn.Linear(in_features=ff_dim, out_features=in_dim)
        self.activation_ffn = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.up(x)
        x = self.activation_ffn(x)
        x = self.down(x)
        return x

class LSTM(torch.nn.Module):

    def __init__(self, emb_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.input_gate = Gate(
            in_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim
        )

        self.forget_gate = Gate(
            in_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim
        )

        self.output_gate = Gate(
            in_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim
        )

        self.cell_gate = Gate(
            in_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            actvation='tanh'
        )

        self.actvation = torch.nn.Tanh()

    def forward(self, x):
        bs = x.shape[0]
        max_len = x.shape[1]
        h = torch.zeros(bs, self.hidden_dim)
        c = torch.zeros(bs, self.hidden_dim)

        for step in range(max_len):
            forget = self.forget_gate(x=x[:, step], h=h)
            input = self.input_gate(x=x[:, step], h=h)
            candidate_c = self.cell_gate(x=x[:, step], h=h)
            output = self.output_gate(x=x[:, step], h=h)
            c = forget * c + input * candidate_c
            h = output * self.actvation(c)
            
        return h, c

class LSTMClassifier(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, ff_dim, vocab_size, padding_idx, num_classes):
        super(LSTMClassifier, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.num_embeddings = vocab_size
        self.num_classes = num_classes
        self.padding_idx = padding_idx

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.emb_dim,
            padding_idx=self.padding_idx
        )

        self.lstm = LSTM(
            emb_dim=self.emb_dim, 
            hidden_dim=self.hidden_dim
        )

        self.ffn = FFN(
            in_dim=self.hidden_dim, 
            ff_dim=self.ff_dim
        )
        self.layer_norm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-12)

        self.dense = torch.nn.Linear(
            in_features=self.hidden_dim, 
            out_features=num_classes
        )

    def forward(self, x, return_feature=False):
        x = self.embedding(x)
        h_last, c_last = self.lstm(x)
        
        h_ffn = self.ffn(h_last)
        h = self.layer_norm(h_last + h_ffn)
        
        scores = self.dense(h)

        if return_feature:
            return scores, h.detach()
        else:
            return scores