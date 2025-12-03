import torch
from torchmetrics import AUROC, Accuracy, F1Score
from torchmetrics.classification import BinaryF1Score

def compute_acc(preds, labels):
    preds_t = torch.as_tensor(preds)
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    if preds_t.dim() > 1:
        preds_t = preds_t.argmax(dim=1)
    else:
        preds_t = preds_t.to(torch.long)

    acc = (preds_t == labels_t).float().mean() * 100
    return float(acc.item())

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

    return float(metric(y_score, y_true).item() * 100)

def compute_binary_f1(pred_k, pred_u, use_gpu=False):
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

    return float(metric(preds, targets).item() * 100)

def compute_multiclass_f1(preds, labels, num_classes, device=None):
    preds_t = torch.as_tensor(preds, dtype=torch.long)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    if device:
        preds_t = preds_t.to(device)
        labels_t = labels_t.to(device)
    metric = F1Score(task="multiclass", num_classes=num_classes, average="macro")
    if device:
        metric = metric.to(device)
    return float(metric(preds_t, labels_t).item())

def compute_multiclass_acc(preds, labels, num_classes, device=None):
    preds_t = torch.as_tensor(preds, dtype=torch.long)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    if device:
        preds_t = preds_t.to(device)
        labels_t = labels_t.to(device)
    metric = Accuracy(task="multiclass", num_classes=num_classes)
    if device:
        metric = metric.to(device)
    return float(metric(preds_t, labels_t).item())

def _recon_metrics(true_labels, cls_preds, errors, threshold, num_known_classes, device=None):
    unknown_label = num_known_classes
    true_t = torch.as_tensor(true_labels, dtype=torch.long)
    preds_t = torch.as_tensor(cls_preds, dtype=torch.long)
    errors_t = torch.as_tensor(errors, dtype=torch.float32)
    if device:
        true_t = true_t.to(device)
        preds_t = preds_t.to(device)
        errors_t = errors_t.to(device)

    predicted_labels = torch.where(
        errors_t <= threshold,
        preds_t,
        torch.full_like(preds_t, unknown_label),
    )

    known_mask = true_t != unknown_label
    unknown_mask = ~known_mask
    known_errors = errors_t[known_mask]
    unknown_errors = errors_t[unknown_mask]

    results = {
        "multiclass_f1": compute_multiclass_f1(
            predicted_labels, true_t, num_classes=num_known_classes + 1, device=device
        ),
        "accuracy": compute_multiclass_acc(
            predicted_labels, true_t, num_classes=num_known_classes + 1, device=device
        ),
    }

    if known_errors.numel() > 0 and unknown_errors.numel() > 0:
        # invert errors so higher = more likely known
        results["auroc"] = compute_auroc(-known_errors, -unknown_errors, use_gpu=device == "cuda")
        results["f1"] = compute_binary_f1(-known_errors, -unknown_errors, use_gpu=device == "cuda")

    return results

def _opengan_metrics(train_results, test_results, num_classes, device=None):
    preds = torch.cat(test_results["predictions"])
    gts = torch.cat(test_results["gts"])
    if device:
        preds = preds.to(device)
        gts = gts.to(device)

    results = {
        "multiclass_f1": compute_multiclass_f1(preds, gts, num_classes=num_classes, device=device),
        "accuracy": compute_multiclass_acc(preds, gts, num_classes=num_classes, device=device),
    }

    if train_results:
        num_samples = max(train_results.get("num_samples", 0), 1)
        if "total_lossD" in train_results:
            results["train_lossD"] = float(train_results["total_lossD"] / num_samples)
        if "total_lossG" in train_results:
            results["train_lossG"] = float(train_results["total_lossG"] / num_samples)

    return results

def _csgrl_metrics(train_results, test_results, num_classes, device=None):
    results: dict[str, float] = {}

    if train_results:
        if "predictions" in train_results and "gts" in train_results:
            train_preds = torch.cat(train_results["predictions"])
            train_gts = torch.cat(train_results["gts"])
            if device:
                train_preds = train_preds.to(device)
                train_gts = train_gts.to(device)
            results["train_accuracy"] = compute_multiclass_acc(
                train_preds, train_gts, num_classes=num_classes, device=device
            )
        if "total_lossD" in train_results:
            results["train_lossD"] = float(
                train_results["total_lossD"] / max(train_results.get("num_samples", 1), 1)
            )
        if "total_lossG" in train_results:
            results["train_lossG"] = float(
                train_results["total_lossG"] / max(train_results.get("num_samples", 1), 1)
            )

    test_preds = torch.cat(test_results["predictions"])
    test_gts = torch.cat(test_results["gts"])
    if device:
        test_preds = test_preds.to(device)
        test_gts = test_gts.to(device)

    mask = test_gts < num_classes  # only known classes
    if mask.any():
        results["multiclass_f1"] = compute_multiclass_f1(
            test_preds[mask], test_gts[mask], num_classes=num_classes, device=device
        )
    else:
        results["multiclass_f1"] = 0.0

    return results

def evaluate(model_name, *, num_known_classes, train_results=None, test_results=None,
             true_labels=None, cls_preds=None, errors=None, threshold=None, device=None):
    """
    Unified evaluator for baseline / c2ae / csgrl / opengan.
    """
    model_name = (model_name or "").lower()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model_name in ("baseline", "c2ae"):
        if any(v is None for v in (true_labels, cls_preds, errors, threshold)):
            raise ValueError("baseline/c2ae evaluate requires true_labels, cls_preds, errors, threshold")
        return _recon_metrics(true_labels, cls_preds, errors, threshold, num_known_classes, device=device)

    if model_name == "csgrl":
        if train_results is None or test_results is None:
            raise ValueError("csgrl evaluate requires train_results and test_results")
        return _csgrl_metrics(train_results, test_results, num_classes=num_known_classes, device=device)

    if model_name == "opengan":
        if test_results is None:
            raise ValueError("opengan evaluate requires test_results")
        return _opengan_metrics(train_results, test_results, num_classes=num_known_classes + 1, device=device)

    raise ValueError(f"Unknown model_name '{model_name}'")

class Metrics:
    def __init__(self, args, num_classes: int | None = None, device: str = "cuda"):
        self.args = args
        self.model_name = getattr(args, "model_name", "").lower()
        self.num_known_classes = getattr(args, "num_known_classes", 0)
        self.num_classes = num_classes if num_classes is not None else self.num_known_classes + 1
        self.device = device if torch.cuda.is_available() else "cpu"

    def compute(self, *args):
        if self.model_name in ("baseline", "c2ae"):
            true_labels, cls_preds, errors, threshold = args
            return evaluate(
                self.model_name,
                num_known_classes=self.num_known_classes,
                true_labels=true_labels,
                cls_preds=cls_preds,
                errors=errors,
                threshold=threshold,
                device=self.device,
            )

        if self.model_name == "csgrl":
            train_results, test_results = args
            return evaluate(
                self.model_name,
                num_known_classes=self.num_known_classes,
                train_results=train_results,
                test_results=test_results,
                device=self.device,
            )

        if self.model_name == "opengan":
            if len(args) == 1:
                test_results = args[0]
                train_results = None
            else:
                train_results, test_results = args
            return evaluate(
                self.model_name,
                num_known_classes=self.num_known_classes,
                train_results=train_results,
                test_results=test_results,
                device=self.device,
            )

        raise ValueError(f"Unknown model_name '{self.model_name}'")

    @staticmethod
    def print_results(results: dict, header: str | None = None) -> None:
        if not results:
            return
        parts = []
        for key, value in results.items():
            if isinstance(value, float):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")
        message = " | ".join(parts)
        if header:
            print(f"{header} | {message}")
        else:
            print(message)

    def print_osr_results(self, results, epoch=None):
        if epoch is not None and getattr(self.args, "epoch_num", None) is not None:
            epoch_info = f"Epoch [{epoch + 1}/{self.args.epoch_num}]"
        else:
            epoch_info = "Evaluation"

        train_parts = []
        for key in ("train_lossD", "train_lossG", "train_accuracy"):
            if key in results:
                train_parts.append(f"{key}: {results[key]:.4f}")

        test_parts = []
        for key in ("multiclass_f1", "accuracy"):
            if key in results:
                test_parts.append(f"{key}: {results[key]:.4f}")

        print(f"[TRAIN] {epoch_info}" + (" | " + " | ".join(train_parts) if train_parts else ""))
        print(f"[TEST]  {epoch_info}" + (" | " + " | ".join(test_parts) if test_parts else ""))
