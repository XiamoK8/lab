import torch
from torchmetrics import AUROC
from torchmetrics.classification import BinaryF1Score


def compute_acc(pred_k, labels, use_gpu=False):
    preds = torch.as_tensor(pred_k)
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    if preds.dim() > 1:
        pred_labels = preds.argmax(dim=1)
    else:
        pred_labels = preds.to(torch.long)

    acc = (pred_labels == labels_t).float().mean() * 100
    return acc.item()


def compute_auroc(pred_k, pred_u, use_gpu=False):
    k_scores = torch.as_tensor(pred_k, dtype=torch.float32)
    u_scores = torch.as_tensor(pred_u, dtype=torch.float32)

    if k_scores.dim() > 1:
        k_scores = k_scores.max(dim=1).values
    if u_scores.dim() > 1:
        u_scores = u_scores.max(dim=1).values

    y_true = torch.cat(
        [
            torch.ones_like(k_scores, dtype=torch.long),
            torch.zeros_like(u_scores, dtype=torch.long),
        ]
    )
    y_score = torch.cat([k_scores, u_scores])

    metric = AUROC(task="binary")
    if use_gpu:
        metric = metric.cuda()

    return (metric(y_score, y_true) * 100).item()


def compute_f1(pred_k, pred_u, use_gpu=False):
    k_scores = torch.as_tensor(pred_k, dtype=torch.float32)
    u_scores = torch.as_tensor(pred_u, dtype=torch.float32)

    if k_scores.dim() > 1:
        k_scores = k_scores.max(dim=1).values
    if u_scores.dim() > 1:
        u_scores = u_scores.max(dim=1).values

    threshold = torch.median(k_scores)
    k_binary = (k_scores >= threshold).long()
    u_binary = (u_scores >= threshold).long()

    preds = torch.cat([k_binary, u_binary])
    targets = torch.cat(
        [
            torch.ones_like(k_binary, dtype=torch.long),
            torch.zeros_like(u_binary, dtype=torch.long),
        ]
    )

    metric = BinaryF1Score()
    if use_gpu:
        metric = metric.cuda()

    return (metric(preds, targets) * 100).item()


def evaluate(pred_k, pred_u=None, labels=None, use_gpu=False):
    results = {}
    if labels is not None:
        results["ACC"] = compute_acc(pred_k, labels, use_gpu)

    if pred_u is not None:
        if torch.numel(torch.as_tensor(pred_u)) > 0:
            results["AUROC"] = compute_auroc(pred_k, pred_u, use_gpu)
            results["F1"] = compute_f1(pred_k, pred_u, use_gpu=use_gpu)

    return results
