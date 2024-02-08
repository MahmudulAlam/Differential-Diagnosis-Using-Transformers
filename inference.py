import csv
import torch
from network import Network
from vocab import build_vocab
from preprocess import parse_patient
from utils import evaluate_ddx

batch_size = 64
vocab_size = 436
max_seq_len = 80
features = 512
heads = 8
layers = 6
output_size = 54
drop_rate = 0.1

network = Network(vocab_size=vocab_size,
                  max_seq_len=max_seq_len,
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
print(de)
print(gt)

en_input = list(map(lambda x: en_vocab.get(x, en_vocab['<unk>']), en_input.split(' ')))
de = list(map(lambda x: de_vocab.get(x, de_vocab['<unk>']), de.split(' ')))
de_input = de[0:-1]
de_output = de[1:]
pathology = de_vocab.get(gt)
print(f'Ground truth encoder input: {en_input}')
print(f'Ground truth decoder input: {de_input}')
print(f'Ground truth decoder output: {de_output}')
print(f'Ground truth pathology: {pathology}')

# inference
network.eval()
with torch.no_grad():
    en_input = torch.tensor([en_input]).long().cuda()
    de_input = torch.tensor([de_input]).long().cuda()
    de_output = torch.tensor([de_output]).long().cuda()
    de_in_ = torch.zeros((1, 40)).long().cuda()
    de_in_[:, 0] = 1  # start decoder with <bos> token
    # out = None

    for i in range(40 - 1):
        print(i, de_in_.tolist())
        y_pred, cls_ = network(en_input=en_input, de_input=de_in_)
        p_ = torch.argmax(y_pred, dim=-1)
        de_in_[:, i + 1] = p_[:, i]
        if p_[:, i] == 0:
            break

            # if torch.eq():
    acc = evaluate_ddx(true=de_output, pred=y_pred)
    print('accuracy', acc)
    print(torch.argmax(cls_, dim=-1))
