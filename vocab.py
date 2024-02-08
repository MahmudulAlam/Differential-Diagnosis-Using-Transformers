from read_utils import *


def scan_duplicate(x):
    x_ = set(list(x.keys()))
    if len(x) == len(x_):
        print('No duplicates found.')
    else:
        print('List contains duplicates')


def build_vocab():
    specials = ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>']

    input_vocab = specials + read_age() + read_sex() + read_evidences()
    output_vocab = specials + read_conditions()

    input_vocab = {key: value for value, key in enumerate(input_vocab)}
    output_vocab = {key: value for value, key in enumerate(output_vocab)}

    return input_vocab, output_vocab


if __name__ == '__main__':
    in_vocab, out_vocab = build_vocab()

    scan_duplicate(in_vocab)
    scan_duplicate(out_vocab)
    print('')

    print(in_vocab)
    print(out_vocab)
    print(len(in_vocab))
    print(len(out_vocab))
    print('')

    # test
    line = '<bos> age_15-29 <sep> F <sep> douleurxx_carac lancinante_/_choc_électrique <eos> <pad>'.split(' ')
    s2i = list(map(lambda x: in_vocab.get(x, in_vocab['<unk>']), line))
    print(s2i)

    line = '<bos> Bronchite RGO Possible_NSTEMI_/_STEMI Angine_instable <eos> <pad> <pad>'.split(' ')
    s2i = list(map(lambda x: out_vocab.get(x, out_vocab['<unk>']), line))
    print(s2i)
