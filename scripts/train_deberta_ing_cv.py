import os, json, random
import numpy as np
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import precision_score, recall_score, f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import inspect

# -----------------------
# CONFIG
# -----------------------
DATA_PATH  = "/home/vanessa/project/labeling/products_evidence_labeled.jsonl"
OUT_DIR    = "/home/vanessa/project/models/deberta_ing_cv"
MODEL_NAME = "microsoft/deberta-v3-base"

LABELS = [
    "acne","comedonal_acne",
    "pigmentation","acne_scars_texture","pores","redness","wrinkles",
]

SEED   = 42
N_FOLDS = 4
MAX_LEN = 512
LR      = 2e-5
EPOCHS  = 3
TRAIN_BS = 8
EVAL_BS  = 16
USE_FP16 = False

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# -----------------------
# Helpers
# -----------------------
def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def multilabel_metrics_tuned_thresholds(logits, y_true):
    probs = sigmoid(logits)

    ts = np.linspace(0.05, 0.95, 19)
    best_t = np.zeros(y_true.shape[1], dtype=np.float32)

    for j in range(y_true.shape[1]):
        best_f1 = -1.0
        best_thr = 0.5
        for t in ts:
            pred_j = (probs[:, j] >= t).astype(int)
            f1_j = f1_score(y_true[:, j], pred_j, zero_division=0)
            if f1_j > best_f1:
                best_f1 = f1_j
                best_thr = float(t)
        best_t[j] = best_thr

    y_pred = (probs >= best_t).astype(int)

    label_accuracy = (y_pred == y_true).mean()
    subset_accuracy = (y_pred == y_true).all(axis=1).mean()

    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_label = {}
    for j, lab in enumerate(LABELS):
        per_label[f"f1_{lab}"] = float(f1_score(y_true[:, j], y_pred[:, j], zero_division=0))

    return {
        "accuracy": float(label_accuracy),
        "subset_acc": float(subset_accuracy),
        "precision": float(micro_p),
        "recall": float(micro_r),
        "f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        **per_label,
    }

def get_ingredients_text(row, sep_token="[SEP]"):
    candidates = [
        "ingredients", "ingredients_list", "ingredient_list",
        "inci", "inci_list", "ingredients_raw"
    ]
    ing = None
    for k in candidates:
        if k in row and row[k]:
            ing = row[k]
            break
    if ing is None:
        return ""

    sep = f" {sep_token} "

    if isinstance(ing, list):
        parts = [str(x).strip() for x in ing if str(x).strip()]
        if not parts:
            return ""
        return sep + sep.join(parts)

    s = str(ing).strip()
    if not s:
        return ""
    return sep + s.replace(",", sep)

def load_dataset(path, sep_token="[SEP]"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = get_ingredients_text(r, sep_token=sep_token)
            if not text.strip():
                continue

            y = []
            for lab in LABELS:
                v = r.get(lab, None)
                if v not in (0, 1):
                    raise ValueError(f"Bad label {lab}={v} for product_url={r.get('product_url')}")
                y.append(float(v))

            rows.append({"text": text, "labels": y})

    return Dataset.from_list(rows)

# -----------------------
# Trainer
# -----------------------
class WeightedBCETrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.shape[-1] != labels.shape[-1]:
            raise ValueError(f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}")

        pos_w = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        loss = loss_fct(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = nn.BCEWithLogitsLoss()(logits, labels)

        return (loss, outputs) if return_outputs else loss

# -----------------------
# Main
# -----------------------
os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
SEP_TOKEN = tokenizer.sep_token or "[SEP]"

ds = load_dataset(DATA_PATH, sep_token=SEP_TOKEN)
Y = np.array(ds["labels"], dtype=int)

print(f"Loaded N={len(ds)} products")
pos_counts = dict(zip(LABELS, Y.sum(axis=0).tolist()))
print("Positives per label:", pos_counts)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
X_dummy = np.zeros((len(ds), 1))
fold_results = []

for fold, (tr_idx, va_idx) in enumerate(mskf.split(X_dummy, Y), start=1):
    print(f"\n=== Fold {fold}/{N_FOLDS} ===")

    ds_train = ds.select(tr_idx).map(tokenize, batched=True, remove_columns=["text"])
    ds_val   = ds.select(va_idx).map(tokenize, batched=True, remove_columns=["text"])
    ds_train.set_format("torch")
    ds_val.set_format("torch")

    Y_train = Y[tr_idx]
    pos = Y_train.sum(axis=0)
    neg = (Y_train.shape[0] - pos)

    pos_weight = np.ones(len(LABELS), dtype=np.float32)
    mask = pos > 0
    pos_weight[mask] = (neg[mask] / pos[mask]).astype(np.float32)

    print("pos_weight:", dict(zip(LABELS, pos_weight.tolist())))
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        torch_dtype=torch.float32,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = labels.astype(int)
        return multilabel_metrics_tuned_thresholds(logits, labels)

    desired_args = {
        "output_dir": os.path.join(OUT_DIR, f"fold_{fold}"),
        "learning_rate": LR,
        "per_device_train_batch_size": TRAIN_BS,
        "per_device_eval_batch_size": EVAL_BS,
        "num_train_epochs": EPOCHS,
        "weight_decay": 0.01,

        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.06,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 2,

        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "logging_steps": 50,

        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,

        "fp16": (torch.cuda.is_available() and USE_FP16),
        "seed": SEED,
        "report_to": "none",
    }

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    supported_args = {k: v for k, v in desired_args.items() if k in ta_params}

    if "evaluation_strategy" not in ta_params and "evaluate_during_training" in ta_params:
        supported_args["evaluate_during_training"] = True
        if "eval_steps" in ta_params and "logging_steps" in supported_args:
            supported_args["eval_steps"] = supported_args["logging_steps"]

    if "save_strategy" not in ta_params and "save_steps" in ta_params:
        supported_args["save_steps"] = supported_args.get("logging_steps", 50)

    if "evaluation_strategy" not in ta_params:
        supported_args.pop("load_best_model_at_end", None)
        supported_args.pop("metric_for_best_model", None)
        supported_args.pop("greater_is_better", None)

    args = TrainingArguments(**supported_args)

    trainer = WeightedBCETrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight_t,
    )

    trainer.train()
    metrics = trainer.evaluate()

    keep_keys = [
        "eval_accuracy", "eval_subset_acc",
        "eval_precision", "eval_recall",
        "eval_f1", "eval_macro_f1"
    ]
    keep = {k: float(metrics[k]) for k in keep_keys if k in metrics}
    print("Fold metrics:", keep)

    per_label_keys = [f"eval_f1_{lab}" for lab in LABELS]
    per_label = {k: float(metrics[k]) for k in per_label_keys if k in metrics}
    print("Per-label F1:", per_label)

    fold_results.append({**keep, **per_label})

def mean_std(key):
    vals = [r[key] for r in fold_results if key in r]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))

print("\n=== 4-Fold CV Summary (mean +/- std) ===")
for k in ["eval_accuracy", "eval_subset_acc", "eval_precision", "eval_recall", "eval_f1", "eval_macro_f1"]:
    m, s = mean_std(k)
    print(f"{k.replace('eval_','')}: {m:.4f} +/- {s:.4f}")

print("\n=== Per-Label F1 (mean +/- std) ===")
for lab in LABELS:
    m, s = mean_std(f"eval_f1_{lab}")
    n = pos_counts.get(lab, 0)
    print(f"  {lab} (n={n}): {m:.4f} +/- {s:.4f}")
