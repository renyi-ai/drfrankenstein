import numpy as np
import torch
from tqdm import tqdm

from src.dataset.utils import _get_data_loader


def eval_net(model_trainer, str_dataset, verbose=False):
    model_trainer.model.eval()

    # loss_func = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()
    batch_size = 50 if str_dataset == 'celeba' else 5000
    val_data_loader = _get_data_loader(str_dataset, 'val', batch_size=batch_size, seed=0)
    true_labels = []
    pred_labels = []
    losses = []

    val_data_loader = tqdm(val_data_loader) if verbose else val_data_loader

    for inputs, labels in val_data_loader:
        preds, loss = model_trainer.evaluate(inputs, labels)
        losses.append(loss.detach().cpu())
        true_labels.extend(list(labels.detach().cpu().numpy().flatten()))
        pred_labels.extend(list(preds.detach().cpu().numpy().flatten()))

    # Calcualte mean values for loss and accuracy
    mean_loss = np.mean(losses)
    hits = np.array(true_labels) == np.array(pred_labels)
    n_equal_values = (hits).sum()
    n_examples = len(true_labels)
    mean_acc = n_equal_values / n_examples

    return mean_loss, mean_acc, hits


def frank_m2_similarity(frank_model, str_dataset, verbose=False):
    m1 = frank_model
    m2 = frank_model.end_model
    return relative_eval_net(m1, m2, str_dataset, verbose=verbose)

def relative_eval_net(m1, m2, str_dataset, verbose=False):
    m1.eval()
    m2.eval()

    multilabel = str_dataset == 'celeba'
    batch_size = 100 if str_dataset == 'celeba' else 500
    val_data_loader = _get_data_loader(str_dataset, 'val', batch_size=batch_size, seed=0)

    m1_out, m2_out, true_labels = [], [], []
    val_data_loader = tqdm(val_data_loader) if verbose else val_data_loader
    for inputs, labels in val_data_loader:
        # Calculate metrics
        inputs = inputs.to(m1.device)
        m1_out.append(m1(inputs).detach().cpu())
        m2_out.append(m2(inputs).detach().cpu())
        true_labels.extend(list(labels.detach().cpu().numpy()))

    m1_out = torch.cat(m1_out, dim=0)
    m2_out = torch.cat(m2_out, dim=0)

    m1_pred = _get_predictions(m1_out, multilabel)
    m2_pred = _get_predictions(m2_out, multilabel)
    
    loss = float(_crossentropy_diff([m1_out, m2_out], multilabel))
    similarity = (m1_pred == m2_pred).numpy().sum() / len(m2_pred)

    m1_hits = np.array(true_labels) == np.array(m1_pred)
    m2_hits = np.array(true_labels) == np.array(m2_pred)
    n = len(m2_hits)
    m1_acc = m1_hits.sum() / n
    m2_acc = m2_hits.sum() / n
    rel_acc = m1_acc / m2_acc

    results = {
        'crossentropy': loss,
        'same_class_out': similarity,
        'rel_acc': rel_acc
    }

    return results


def _get_predictions(outputs, multilabel):
    if multilabel:
        preds = torch.where(torch.nn.Sigmoid()(outputs) > 0.5,
                            torch.ones_like(outputs),
                            torch.zeros_like(outputs))
    else:
        preds = torch.max(outputs.data, 1)[1]
    return preds


def _crossentropy_diff(outputs, multilabel):
    x = _get_p_distributions(outputs[0], multilabel)
    y = _get_p_distributions(outputs[1], multilabel)

    def cross_entropy(x, y):
        return torch.mean(-y * torch.log(x))

    loss_func = torch.nn.BCELoss() if multilabel else cross_entropy
    return loss_func(x, y)


def _cka(outputs, multilabel, norm=True):
    from src.comparators.compare_functions import cka
    if norm:
        x = _get_p_distributions(outputs[0], multilabel).detach().cpu().numpy()
        y = _get_p_distributions(outputs[1], multilabel).detach().cpu().numpy()
    else:
        x = outputs[0].view(-1, 1).detach().cpu().numpy().astype(np.float32)
        y = outputs[1].view(-1, 1).detach().cpu().numpy().astype(np.float32)

    if multilabel:
        step = 4000
        measures = []
        for i in range(0, len(x), step):
            start = i
            end = min(i + step, len(x))
            a = x[start:end]
            b = y[start:end]
            measures.append(cka(a, b))
        return np.mean(measures, axis=0)
    else:
        return cka(x, y)


def _get_p_distributions(x, multilabel):
    if multilabel:
        return torch.nn.functional.sigmoid(x)
    else:
        return torch.nn.functional.softmax(x, dim=1)
