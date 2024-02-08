import csv
import torch


def mean(x):
    return sum(x) / len(x)


def evaluate_ddx(true, pred):
    """
    evaluates differential diagnosis accuracy
    :param true: ground truth sequence of labels
    :param pred: decoder sequence of predictions
    :return: accuracy
    """
    mask = torch.where(true > 0, 1., 0.)
    pred = torch.argmax(pred, dim=-1)
    acc = (true == pred).float() * mask
    acc = torch.sum(acc) / torch.sum(mask)
    return acc


def evaluate_cls(true, pred):
    """
    evaluates accuracy of pathology classification
    :param true: ground truth labels of pathology
    :param pred: predicted one-hot approximation of classifier
    :return:
    """
    pred = torch.argmax(pred, dim=-1)
    acc = (true == pred).float().mean()
    return acc


def save_history(file, history, mode='w'):
    """
    writes history to a csv file
    :param file: name of the file
    :param history: list of history
    :param mode: writing mode
    :return: None
    """
    with open(file, mode) as f:
        writer = csv.writer(f)
        history = [line.replace(':', ',').split(',') for line in history]
        [writer.writerow(line) for line in history]
