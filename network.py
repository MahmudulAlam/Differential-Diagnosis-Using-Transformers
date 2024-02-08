import math
import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size: int, features: int):
        super(WordEmbedding, self).__init__()
        self.we = nn.Embedding(num_embeddings=vocab_size, embedding_dim=features)

    def forward(self, x):
        return self.we(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, features: int):
        super(PositionalEmbedding, self).__init__()
        self.positions = torch.arange(0, max_seq_len).cuda()
        self.pe = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=features)

    def forward(self, x):
        return x + self.pe(self.positions)[None, 0:x.size(1)]


# class PositionalEncoding(nn.Module):
#     def __init__(self, max_seq_len: int, features: int):
#         super(PositionalEncoding, self).__init__()
#         position = torch.arange(max_seq_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, features, 2) * (-math.log(10000.0) / features))
#         pe = torch.zeros(1, max_seq_len, features)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return x + self.pe[:, 0:x.size(1)]


class MHAttention(nn.Module):
    def __init__(self, features: int, heads: int, dropout_rate: float):
        super(MHAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=features, num_heads=heads, dropout=dropout_rate, batch_first=True)

    def forward(self, query, key, value, mask, look_ahead_mask):
        x, w = self.mha(query=query, key=key, value=value, key_padding_mask=mask, attn_mask=look_ahead_mask)
        return x


class MLP(nn.Module):
    def __init__(self, features: int, dropout_rate: float):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(features, features * 4)
        self.linear2 = nn.Linear(features * 4, features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    def __init__(self, features, heads, dropout_rate):
        super(Encoder, self).__init__()
        self.ln = nn.LayerNorm(features, eps=1e-6)
        self.attn = MHAttention(features=features, heads=heads, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp = MLP(features=features, dropout_rate=dropout_rate)

    def forward(self, x, mask):
        lnx = self.ln(x)
        attention = self.attn(query=lnx, key=lnx, value=lnx, mask=mask, look_ahead_mask=None)
        attention = self.dropout(attention)
        x = x + attention

        lnx = self.ln(x)
        feed_forward = self.mlp(lnx)
        return x + feed_forward


class Decoder(nn.Module):
    def __init__(self, features, heads, dropout_rate):
        super(Decoder, self).__init__()
        self.ln = nn.LayerNorm(features, eps=1e-6)
        self.attn_mask = MHAttention(features=features, heads=heads, dropout_rate=dropout_rate)
        self.attn_en_de = MHAttention(features=features, heads=heads, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp = MLP(features=features, dropout_rate=dropout_rate)

    @staticmethod
    def causal_mask(x):
        n = x.shape[1]
        return (1. - torch.tril(torch.ones(n, n).cuda())) * -1e9

    def forward(self, decoder_input, encoder_output, en_mask, de_mask):
        de_in = self.ln(decoder_input)
        attention = self.attn_mask(query=de_in, key=de_in, value=de_in, mask=de_mask,
                                   look_ahead_mask=self.causal_mask(de_in))
        attention = self.dropout(attention)
        x = decoder_input + attention

        en_out = self.ln(encoder_output)
        attention = self.attn_en_de(query=x, key=en_out, value=en_out, mask=en_mask, look_ahead_mask=None)
        attention = self.dropout(attention)
        x = x + attention

        lnx = self.ln(x)
        feed_forward = self.mlp(lnx)
        return x + feed_forward


class Network(nn.Module):
    def __init__(self, vocab_size: int, en_seq_len: int, de_seq_len: int, features: int, heads: int,
                 n_layer: int, output_size: int, dropout_rate: float = 0.1):
        super(Network, self).__init__()
        self.en_we = WordEmbedding(vocab_size=vocab_size, features=features)
        self.en_pe = PositionalEmbedding(max_seq_len=en_seq_len, features=features)

        self.de_we = WordEmbedding(vocab_size=vocab_size, features=features)
        self.de_pe = PositionalEmbedding(max_seq_len=de_seq_len, features=features)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.encoder = nn.ModuleList([
            Encoder(features=features, heads=heads, dropout_rate=dropout_rate) for _ in range(n_layer)
        ])
        self.decoder = nn.ModuleList([
            Decoder(features=features, heads=heads, dropout_rate=dropout_rate) for _ in range(n_layer)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(features, eps=1e-6),
            nn.Linear(features, output_size)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(features * 2, eps=1e-6),
            nn.Linear(features * 2, features),
            nn.GELU(),
            nn.LayerNorm(features, eps=1e-6),
            nn.Linear(features, output_size)
        )

    def forward(self, en_input, de_input):
        encoder_mask = torch.where(en_input > 0, False, True)
        decoder_mask = torch.where(de_input > 0, False, True)

        en_input = self.en_we(en_input)
        en_input = self.en_pe(en_input)
        en_input = self.dropout1(en_input)

        de_input = self.de_we(de_input)
        de_input = self.de_pe(de_input)
        de_input = self.dropout2(de_input)

        x = en_input
        for layer in self.encoder:
            x = layer(x, encoder_mask)
        en_output = x

        x = de_input
        for layer in self.decoder:
            x = layer(x, en_output, encoder_mask, decoder_mask)
        de_output = x

        # differential diagnoses
        ddx = self.head(x)

        # pathology classification
        x = torch.cat([torch.mean(en_output, dim=1), torch.mean(de_output, dim=1)], dim=-1)
        x = self.classifier(x)
        return ddx, x


if __name__ == '__main__':
    from torchinfo import summary

    encoder_inputs_ = torch.tensor([[1, 5, 6, 4, 3, 2], [1, 3, 7, 4, 2, 0]]).cuda()
    decoder_inputs_ = torch.tensor([[1, 3, 4, 6, 5], [1, 4, 7, 3, 2]]).cuda()

    network = Network(vocab_size=10,
                      en_seq_len=6,
                      de_seq_len=5,
                      features=32,
                      heads=4,
                      n_layer=2,
                      output_size=10,
                      dropout_rate=0.1).cuda()
    ddx_, cls_ = network(encoder_inputs_, decoder_inputs_)

    print(ddx_.shape)
    print(cls_.shape)

    summary(network, input_data={"en_input": encoder_inputs_,
                                 "de_input": decoder_inputs_}, dtypes=[torch.IntTensor, torch.IntTensor])
    # summary()
