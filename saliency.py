import csv
import torch
import torch.nn as nn
from network import Network
from vocab import build_vocab
from preprocess import parse_patient
from utils import evaluate_ddx

batch_size = 64
vocab_size = 436
en_seq_len = 80
de_seq_len = 40
features = 128
heads = 4
layers = 6
output_size = 54
drop_rate = 0.1

network = Network(vocab_size=vocab_size,
                  en_seq_len=en_seq_len,
                  de_seq_len=de_seq_len,
                  features=features,
                  heads=heads,
                  n_layer=layers,
                  output_size=output_size,
                  dropout_rate=drop_rate).cuda()

network.load_state_dict(torch.load('./weights/model.h5'))

# loading inference sample
en_vocab, de_vocab = build_vocab()

filename = 'data/release_test_patients.csv'
with open(filename, mode='r', encoding='utf-8') as f:
    loader = list(csv.DictReader(f))

en_input, de, gt = parse_patient(loader[2], en_max_len=80, de_max_len=41)
print(en_input)
# print(de)
# print(gt)

en_input = list(map(lambda x: en_vocab.get(x, en_vocab['<unk>']), en_input.split(' ')))
de = list(map(lambda x: de_vocab.get(x, de_vocab['<unk>']), de.split(' ')))
de_input = de[0:-1]
de_output = de[1:]
pathology = de_vocab.get(gt)
print(f'encoder input: {en_input}')
print(f'decoder input: {de_input}')

# inference

network.train()
# network.zero_grad()

en_input = torch.tensor([en_input]).long().cuda()
de_input = torch.tensor([de_input]).long().cuda()


x = network.en_we(en_input)
x = network.en_pe(x)
embed = torch.tensor(x, requires_grad=True)


network.train()
# if I set model.eval(), an error occur: RuntimeError: cudnn RNN backward can only be called in training mode


def model(en_input, de_input, embed):
    encoder_mask = torch.where(en_input > 0, False, True)
    decoder_mask = torch.where(de_input > 0, False, True)

    en_input = network.dropout1(embed)

    de_input = network.de_we(de_input)
    de_input = network.de_pe(de_input)
    de_input = network.dropout2(de_input)

    x = en_input
    for layer in network.encoder:
        x = layer(x, encoder_mask)
    en_output = x

    x = de_input
    for layer in network.decoder:
        x = layer(x, en_output, encoder_mask, decoder_mask)
    de_output = x

    # differential diagnoses
    ddx = network.head(x)

    # pathology classification
    x = torch.cat([torch.mean(en_output, dim=1), torch.mean(de_output, dim=1)], dim=-1)
    x = network.classifier(x)
    return ddx, x


_, pred = model(en_input, de_input, embed)
'''
backward function on score_max performs the backward pass in the computation graph and calculates the gradient of
score_max with respect to nodes in the computation graph
'''
score, indices = torch.max(pred, 1)
score.backward()
'''
Saliency would be the gradient with respect to the input now.
But note that the input has 100 dim embdeddings.
To derive a single class saliency value for each word (i, j),
we take the maximum magnitude across all embedding dimensions.
'''
# saliency, _ = torch.max(embed.grad.data.abs(), dim=2)  # AttributeError: 'NoneType' object has no attribute 'data'
print(embed.grad.shape)
slc, _ = torch.max(torch.abs(embed.grad[0]), dim=-1)
# slc = (slc - slc.min()) / (slc.max() - slc.min())
slc = nn.Softmax(dim=-1)(slc)
print(slc.shape)

print(slc)