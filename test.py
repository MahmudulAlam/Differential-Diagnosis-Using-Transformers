import time
import torch
import numpy as np
from network import Network
from dataset import load_dataset
from utils import mean, evaluate_ddx, evaluate_cls

batch_size = 64
vocab_size = 436
en_seq_len = 80
de_seq_len = 40
features = 128
heads = 4
layers = 6
output_size = 54
drop_rate = 0.1

print('Loading data & network ...')
_, test_loader = load_dataset(batch_size=batch_size, num_workers=0)

network = Network(vocab_size=vocab_size,
                  en_seq_len=en_seq_len,
                  de_seq_len=de_seq_len,
                  features=features,
                  heads=heads,
                  n_layer=layers,
                  output_size=output_size,
                  dropout_rate=drop_rate).cuda()

network.load_state_dict(torch.load('./weights/model.h5'))

print('Start testing ...')

# test
network.eval()
test_acc_ddx, test_acc_cls = [], []
tic = time.time()

np_true_ddx = []
np_pred_ddx = []

np_true_cls = []
np_pred_cls = []

with torch.no_grad():
    for n, (en_in, de_in, de_out, path) in enumerate(test_loader):
        en_in, de_in, de_out, path = en_in.cuda(), de_in.cuda(), de_out.cuda(), path.cuda()
        # de_out = one_hot(de_out, output_size)

        # forward
        de_out_pred, path_pred = network(en_input=en_in, de_input=de_in)

        # store
        np_true_ddx.append(de_out.detach().cpu().numpy())
        np_pred_ddx.append(torch.argmax(de_out_pred, dim=-1).detach().cpu().numpy())
        np_true_cls.append(path.detach().cpu().numpy())
        np_pred_cls.append(torch.argmax(path_pred, dim=-1).detach().cpu().numpy())

        # evaluate
        ddx_acc = evaluate_ddx(true=de_out, pred=de_out_pred)
        cls_acc = evaluate_cls(true=path, pred=path_pred)
        test_acc_ddx.append(ddx_acc.item())
        test_acc_cls.append(cls_acc.item())

test_acc_ddx = mean(test_acc_ddx) * 100
test_acc_cls = mean(test_acc_cls) * 100
toc = time.time()

print(f'test ddx acc: {test_acc_ddx:.2f}%, test cls acc: {test_acc_cls:.2f}%, eta: {toc - tic:.2}s')

np_true_ddx = np.concatenate(np_true_ddx, dtype=np.float32)
np_pred_ddx = np.concatenate(np_pred_ddx, dtype=np.float32)
np_true_cls = np.concatenate(np_true_cls, dtype=np.float32)
np_pred_cls = np.concatenate(np_pred_cls, dtype=np.float32)

print(np_true_ddx.shape)
print(np_pred_ddx.shape)
print(np_true_cls.shape)
print(np_pred_cls.shape)

# save file
np.save('results/true_ddx.npy', np_true_ddx)
np.save('results/pred_ddx.npy', np_pred_ddx)
np.save('results/true_cls.npy', np_true_cls)
np.save('results/pred_cls.npy', np_pred_cls)

print('All Done!')
