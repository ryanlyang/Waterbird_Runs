import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np


FLOAT_RE = re.compile(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))

def _maybe_import_omegaconf():
    try:
        from omegaconf import OmegaConf  # type: ignore

        return OmegaConf
    except Exception:
        return None


def _infer_dataset_and_attention_dir(config_path):
    OmegaConf = _maybe_import_omegaconf()
    if OmegaConf is None:
        return None, None
    try:
        cfg = OmegaConf.load(config_path)
    except Exception:
        return None, None

    dataset = None
    attention_dir = None
    try:
        dataset = str(cfg.DATA.DATASET)
    except Exception:
        dataset = None
    try:
        attention_dir = str(cfg.DATA.ATTENTION_DIR)
    except Exception:
        attention_dir = None
    return dataset, attention_dir


def write_row(csv_path, row, header):
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _parse_metric_line(line):
    # Matches lines like: "balanced_val_acc: 0.8123" or "test_acc: 0.901"
    parts = line.strip().split(":")
    if len(parts) < 2:
        return None
    key = parts[0].strip()
    m = FLOAT_RE.search(line)
    if not key or m is None:
        return None
    return key, float(m.group(1))


def run_one_trial(
    trial_id,
    *,
    config,
    data_root,
    waterbirds_dir,
    dataset_name,
    train_seed,
    base_lr,
    classifier_lr,
    grad_weight,
    weight_decay,
    python_exe,
    logs_dir,
):
    run_name = f"gals_trial_{trial_id:03d}"
    dataset_root = os.path.join(data_root, waterbirds_dir)

    cmd = [
        python_exe,
        "-u",
        "main.py",
        "--config",
        config,
        "--name",
        run_name,
        f"DATA.ROOT={data_root}",
        f"DATA.WATERBIRDS_DIR={waterbirds_dir}",
        f"SEED={train_seed}",
        f"EXP.BASE.LR={base_lr}",
        f"EXP.CLASSIFIER.LR={classifier_lr}",
        f"EXP.LOSSES.GRADIENT_OUTSIDE.WEIGHT={grad_weight}",
        f"EXP.WEIGHT_DECAY={weight_decay}",
    ]

    os.makedirs(logs_dir, exist_ok=True)
    trial_log = os.path.join(logs_dir, f"{run_name}.log")

    best_balanced_val = None
    test_metrics = {}
    checkpoint_used = None

    run_dir = os.path.join("trained_weights", dataset_name, run_name)
    start = time.time()
    with open(trial_log, "w") as lf:
        lf.write(f"[CMD] {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()

            if "BALANCED VAL ACC:" in line:
                m = FLOAT_RE.search(line)
                if m:
                    v = float(m.group(1))
                    if best_balanced_val is None or v > best_balanced_val:
                        best_balanced_val = v

            if line.startswith("TEST SET RESULTS FOR CHECKPOINT"):
                # example: TEST SET RESULTS FOR CHECKPOINT path
                checkpoint_used = line.strip().split("CHECKPOINT", 1)[-1].strip()

            parsed = _parse_metric_line(line)
            if parsed:
                k, v = parsed
                if k.endswith("_test_acc") or k in ("test_acc", "balanced_test_acc", "balanced_val_acc"):
                    test_metrics[k] = v

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Trial {trial_id} failed with exit code {rc}. See {trial_log}")

    # Compute per-group and worst-group on test if available
    group_keys = [k for k in test_metrics.keys() if k.endswith("_test_acc") and k not in ("balanced_test_acc",)]
    group_vals = [test_metrics[k] for k in group_keys]
    per_group = float(np.mean(group_vals)) if group_vals else None
    worst_group = float(np.min(group_vals)) if group_vals else None

    # Normalize accs to %
    def to_pct(x):
        if x is None:
            return None
        return x * 100.0 if x <= 1.0 else x

    elapsed_s = time.time() - start

    return {
        "trial": trial_id,
        "name": run_name,
        "seed": train_seed,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "grad_weight": grad_weight,
        "weight_decay": weight_decay,
        "best_balanced_val_acc": best_balanced_val,
        "test_acc": to_pct(test_metrics.get("test_acc")),
        "balanced_test_acc": to_pct(test_metrics.get("balanced_test_acc")),
        "per_group": to_pct(per_group),
        "worst_group": to_pct(worst_group),
        "checkpoint": checkpoint_used,
        "log_path": trial_log,
        "seconds": int(elapsed_s),
        "run_dir": run_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/waterbirds_95_gals.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--waterbirds-dir", default="waterbird_complete95_forest2water2")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling hyperparameters")
    parser.add_argument("--train-seed", type=int, default=0, help="Training seed used for every trial")
    parser.add_argument("--output-csv", default="gals_sweep.csv")
    parser.add_argument("--logs-dir", default="gals_sweep_logs")
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument(
        "--keep",
        choices=["best", "all", "none"],
        default="best",
        help="What to keep under trained_weights/: best=keep best trial only, all=keep all, none=delete everything",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name used for run dirs under trained_weights/ (defaults to DATA.DATASET from config)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Optional wallclock limit; stops launching new trials once exceeded",
    )

    parser.add_argument("--weight-min", type=float, default=1.0)
    parser.add_argument("--weight-max", type=float, default=100000.0)
    parser.add_argument("--base-lr-min", type=float, default=1e-4)
    parser.add_argument("--base-lr-max", type=float, default=5e-2)
    parser.add_argument("--cls-lr-min", type=float, default=1e-5)
    parser.add_argument("--cls-lr-max", type=float, default=1e-2)
    parser.add_argument(
        "--tune-weight-decay",
        action="store_true",
        help="If set, also tune EXP.WEIGHT_DECAY (otherwise uses the value from the YAML config).",
    )
    parser.add_argument("--weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay-max", type=float, default=1e-3)
    args = parser.parse_args()

    header = [
        "trial",
        "name",
        "seed",
        "base_lr",
        "classifier_lr",
        "grad_weight",
        "weight_decay",
        "best_balanced_val_acc",
        "test_acc",
        "balanced_test_acc",
        "per_group",
        "worst_group",
        "checkpoint",
        "log_path",
        "seconds",
        "sampler",
        "run_dir",
    ]

    rng = np.random.default_rng(args.seed)
    python_exe = sys.executable

    inferred_dataset, inferred_attention_dir = _infer_dataset_and_attention_dir(args.config)
    dataset_name = args.dataset or inferred_dataset or "waterbirds"

    cfg_weight_decay = None
    OmegaConf = _maybe_import_omegaconf()
    if OmegaConf is not None:
        try:
            _cfg = OmegaConf.load(args.config)
            cfg_weight_decay = float(_cfg.EXP.WEIGHT_DECAY)
        except Exception:
            cfg_weight_decay = None
    if cfg_weight_decay is None:
        cfg_weight_decay = 1e-5

    if inferred_attention_dir and inferred_attention_dir.upper() != "NONE":
        expected_attention_dir = os.path.join(args.data_root, args.waterbirds_dir, inferred_attention_dir)
        if not os.path.isdir(expected_attention_dir):
            raise FileNotFoundError(
                "Missing precomputed attention maps for GALS.\n"
                f"Expected directory: {expected_attention_dir}\n"
                "Run `extract_attention.py` first (stage 1) using the corresponding *_attention.yaml config."
            )

    if args.sampler == "tpe":
        try:
            import optuna  # noqa: F401
        except Exception as exc:
            print(f"[SWEEP] Optuna not available ({exc}); falling back to random search.", flush=True)
            args.sampler = "random"

    best_row = None
    best_dir = None

    def cleanup_run_dir(path):
        if not path:
            return
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

    def run_and_record(trial_id, base_lr, classifier_lr, grad_weight, weight_decay, sampler_name):
        nonlocal best_row, best_dir
        row = run_one_trial(
            trial_id,
            config=args.config,
            data_root=args.data_root,
            waterbirds_dir=args.waterbirds_dir,
            dataset_name=dataset_name,
            train_seed=args.train_seed,
            base_lr=base_lr,
            classifier_lr=classifier_lr,
            grad_weight=grad_weight,
            weight_decay=weight_decay,
            python_exe=python_exe,
            logs_dir=args.logs_dir,
        )
        row["sampler"] = sampler_name
        write_row(args.output_csv, row, header)

        score = row["best_balanced_val_acc"]
        is_new_best = score is not None and (best_row is None or score > best_row["best_balanced_val_acc"])
        run_dir = row.get("run_dir")

        if args.keep == "none":
            cleanup_run_dir(run_dir)
        elif args.keep == "best":
            if is_new_best:
                cleanup_run_dir(best_dir)
                best_dir = run_dir
            else:
                cleanup_run_dir(run_dir)

        if is_new_best:
            best_row = row
        return row

    start_time = time.time()

    if args.sampler == "random":
        for trial_id in range(args.n_trials):
            if args.max_hours is not None and (time.time() - start_time) >= args.max_hours * 3600:
                print(f"[SWEEP] Reached max-hours={args.max_hours}; stopping before trial {trial_id}.", flush=True)
                break
            grad_weight = loguniform(rng, args.weight_min, args.weight_max)
            base_lr = loguniform(rng, args.base_lr_min, args.base_lr_max)
            classifier_lr = loguniform(rng, args.cls_lr_min, args.cls_lr_max)
            weight_decay = (
                loguniform(rng, args.weight_decay_min, args.weight_decay_max)
                if args.tune_weight_decay
                else cfg_weight_decay
            )
            try:
                row = run_and_record(trial_id, base_lr, classifier_lr, grad_weight, weight_decay, "random")
                print(
                    f"[SWEEP] Trial {trial_id} done. best_balanced_val_acc={row['best_balanced_val_acc']}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[SWEEP] Trial {trial_id} failed: {exc}", flush=True)
    else:
        import optuna

        def objective(trial):
            grad_weight = float(trial.suggest_float("grad_weight", args.weight_min, args.weight_max, log=True))
            base_lr = float(trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
            classifier_lr = float(trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
            weight_decay = (
                float(trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max, log=True))
                if args.tune_weight_decay
                else cfg_weight_decay
            )
            row = run_and_record(trial.number, base_lr, classifier_lr, grad_weight, weight_decay, "tpe")
            print(f"[SWEEP] Trial {trial.number} done. best_balanced_val_acc={row['best_balanced_val_acc']}", flush=True)
            return row["best_balanced_val_acc"] if row["best_balanced_val_acc"] is not None else -1.0

        callbacks = []
        if args.max_hours is not None:
            def _time_limit_cb(study, _trial):
                if (time.time() - start_time) >= args.max_hours * 3600:
                    print(f"[SWEEP] Reached max-hours={args.max_hours}; stopping study.", flush=True)
                    study.stop()
            callbacks.append(_time_limit_cb)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials, catch=(RuntimeError,), callbacks=callbacks)

    if best_row is not None:
        print("[SWEEP] Best trial:", flush=True)
        for k in header:
            if k in best_row:
                print(f"  {k}: {best_row[k]}", flush=True)


if __name__ == "__main__":
    main()
