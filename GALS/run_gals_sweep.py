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
    run_name=None,
    method="gals",
    config,
    data_root,
    waterbirds_dir,
    dataset_name,
    train_seed,
    base_lr,
    classifier_lr,
    grad_weight=None,
    cam_weight=None,
    abn_cls_weight=None,
    abn_att_weight=None,
    weight_decay,
    python_exe,
    logs_dir,
    extra_overrides=None,
):
    if run_name is None:
        run_name = f"{method}_trial_{trial_id:03d}"

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
        f"EXP.WEIGHT_DECAY={weight_decay}",
    ]
    if method in ("gals", "rrr"):
        if grad_weight is None:
            raise ValueError("grad_weight must be provided when method is gals/rrr")
        cmd.append(f"EXP.LOSSES.GRADIENT_OUTSIDE.WEIGHT={grad_weight}")
    elif method == "gradcam":
        if cam_weight is None:
            raise ValueError("cam_weight must be provided when method='gradcam'")
        cmd.append(f"EXP.LOSSES.GRADCAM.WEIGHT={cam_weight}")
    elif method == "abn_cls":
        if abn_cls_weight is None:
            raise ValueError("abn_cls_weight must be provided when method='abn_cls'")
        cmd.append(f"EXP.LOSSES.ABN_CLASSIFICATION.WEIGHT={abn_cls_weight}")
    elif method == "abn_att":
        if abn_att_weight is None:
            raise ValueError("abn_att_weight must be provided when method='abn_att'")
        cmd.append(f"EXP.LOSSES.ABN_SUPERVISION.WEIGHT={abn_att_weight}")
    elif method == "upweight":
        pass
    else:
        raise ValueError(f"Unknown method: {method}")

    if extra_overrides:
        cmd.extend(list(extra_overrides))

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
        "method": method,
        "seed": train_seed,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "grad_weight": grad_weight,
        "cam_weight": cam_weight,
        "abn_cls_weight": abn_cls_weight,
        "abn_att_weight": abn_att_weight,
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
    parser.add_argument(
        "--method",
        choices=["gals", "rrr", "gradcam", "abn_cls", "abn_att", "upweight"],
        default="gals",
        help="Which 'classifier attention method' to sweep. gals/rrr=GRADIENT_OUTSIDE, gradcam=GRADCAM, "
        "abn_cls=ABN_CLASSIFICATION, abn_att=ABN_SUPERVISION, upweight=no attention loss.",
    )
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

    parser.add_argument("--weight-min", type=float, default=1.0, help="Min for GRADIENT_OUTSIDE weight (gals/rrr)")
    parser.add_argument("--weight-max", type=float, default=100000.0, help="Max for GRADIENT_OUTSIDE weight (gals/rrr)")
    parser.add_argument("--cam-weight-min", type=float, default=0.1, help="Min for GRADCAM weight (gradcam)")
    parser.add_argument("--cam-weight-max", type=float, default=100.0, help="Max for GRADCAM weight (gradcam)")
    parser.add_argument("--abn-cls-weight-min", type=float, default=0.1, help="Min for ABN_CLASSIFICATION weight (abn_cls)")
    parser.add_argument("--abn-cls-weight-max", type=float, default=10.0, help="Max for ABN_CLASSIFICATION weight (abn_cls)")
    parser.add_argument("--abn-att-weight-min", type=float, default=0.1, help="Min for ABN_SUPERVISION weight (abn_att)")
    parser.add_argument("--abn-att-weight-max", type=float, default=10.0, help="Max for ABN_SUPERVISION weight (abn_att)")
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
    parser.add_argument(
        "--post-seeds",
        type=int,
        default=0,
        help="After the sweep, rerun the best hyperparameters for N different seeds and write a second CSV.",
    )
    parser.add_argument(
        "--post-seed-start",
        type=int,
        default=0,
        help="Starting seed for post-sweep reruns (uses seeds post_seed_start..post_seed_start+post_seeds-1).",
    )
    parser.add_argument(
        "--post-output-csv",
        default=None,
        help="CSV for post-sweep seed reruns (default: <output-csv basename>_best_seeds.csv).",
    )
    parser.add_argument(
        "--post-logs-dir",
        default=None,
        help="Logs dir for post-sweep seed reruns (default: <logs-dir>_best_seeds).",
    )
    parser.add_argument(
        "--post-keep",
        choices=["all", "none"],
        default="all",
        help="What to keep for the post-sweep seed reruns under trained_weights/: all or none.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra OmegaConf overrides passed through to main.py (e.g. DATA.SEGMENTATION_DIR=/path/to/masks).",
    )
    args = parser.parse_args()

    header = [
        "trial",
        "name",
        "method",
        "seed",
        "base_lr",
        "classifier_lr",
        "grad_weight",
        "cam_weight",
        "abn_cls_weight",
        "abn_att_weight",
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

    def run_and_record(
        trial_id,
        base_lr,
        classifier_lr,
        grad_weight,
        cam_weight,
        abn_cls_weight,
        abn_att_weight,
        weight_decay,
        sampler_name,
    ):
        nonlocal best_row, best_dir
        row = run_one_trial(
            trial_id,
            method=args.method,
            config=args.config,
            data_root=args.data_root,
            waterbirds_dir=args.waterbirds_dir,
            dataset_name=dataset_name,
            train_seed=args.train_seed,
            base_lr=base_lr,
            classifier_lr=classifier_lr,
            grad_weight=grad_weight,
            cam_weight=cam_weight,
            abn_cls_weight=abn_cls_weight,
            abn_att_weight=abn_att_weight,
            weight_decay=weight_decay,
            python_exe=python_exe,
            logs_dir=args.logs_dir,
            extra_overrides=args.overrides,
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
            base_lr = loguniform(rng, args.base_lr_min, args.base_lr_max)
            classifier_lr = loguniform(rng, args.cls_lr_min, args.cls_lr_max)
            weight_decay = (
                loguniform(rng, args.weight_decay_min, args.weight_decay_max)
                if args.tune_weight_decay
                else cfg_weight_decay
            )
            grad_weight = None
            cam_weight = None
            abn_cls_weight = None
            abn_att_weight = None
            if args.method in ("gals", "rrr"):
                grad_weight = loguniform(rng, args.weight_min, args.weight_max)
            elif args.method == "gradcam":
                cam_weight = loguniform(rng, args.cam_weight_min, args.cam_weight_max)
            elif args.method == "abn_cls":
                abn_cls_weight = loguniform(rng, args.abn_cls_weight_min, args.abn_cls_weight_max)
            elif args.method == "abn_att":
                abn_att_weight = loguniform(rng, args.abn_att_weight_min, args.abn_att_weight_max)
            try:
                row = run_and_record(
                    trial_id,
                    base_lr,
                    classifier_lr,
                    grad_weight,
                    cam_weight,
                    abn_cls_weight,
                    abn_att_weight,
                    weight_decay,
                    "random",
                )
                print(
                    f"[SWEEP] Trial {trial_id} done. best_balanced_val_acc={row['best_balanced_val_acc']}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[SWEEP] Trial {trial_id} failed: {exc}", flush=True)
    else:
        import optuna

        def objective(trial):
            base_lr = float(trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
            classifier_lr = float(trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
            weight_decay = (
                float(trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max, log=True))
                if args.tune_weight_decay
                else cfg_weight_decay
            )
            grad_weight = None
            cam_weight = None
            abn_cls_weight = None
            abn_att_weight = None
            if args.method in ("gals", "rrr"):
                grad_weight = float(trial.suggest_float("grad_weight", args.weight_min, args.weight_max, log=True))
            elif args.method == "gradcam":
                cam_weight = float(trial.suggest_float("cam_weight", args.cam_weight_min, args.cam_weight_max, log=True))
            elif args.method == "abn_cls":
                abn_cls_weight = float(
                    trial.suggest_float("abn_cls_weight", args.abn_cls_weight_min, args.abn_cls_weight_max, log=True)
                )
            elif args.method == "abn_att":
                abn_att_weight = float(
                    trial.suggest_float("abn_att_weight", args.abn_att_weight_min, args.abn_att_weight_max, log=True)
                )

            row = run_and_record(
                trial.number,
                base_lr,
                classifier_lr,
                grad_weight,
                cam_weight,
                abn_cls_weight,
                abn_att_weight,
                weight_decay,
                "tpe",
            )
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

    if best_row is not None and args.post_seeds > 0:
        post_csv = args.post_output_csv
        if post_csv is None:
            root, ext = os.path.splitext(args.output_csv)
            post_csv = f"{root}_best_seeds{ext or '.csv'}"
        post_logs_dir = args.post_logs_dir or f"{args.logs_dir}_best_seeds"

        post_header = [
            "seed",
            "name",
            "method",
            "base_lr",
            "classifier_lr",
            "grad_weight",
            "cam_weight",
            "abn_cls_weight",
            "abn_att_weight",
            "weight_decay",
            "best_balanced_val_acc",
            "test_acc",
            "balanced_test_acc",
            "per_group",
            "worst_group",
            "checkpoint",
            "log_path",
            "seconds",
            "run_dir",
            "sweep_best_trial",
        ]

        seeds = list(range(args.post_seed_start, args.post_seed_start + args.post_seeds))
        print(f"[POST] Rerunning best hyperparameters for {len(seeds)} seeds: {seeds}", flush=True)

        post_rows = []
        for s in seeds:
            run_name = f"{args.method}_best_seed{s}"
            row = run_one_trial(
                100000 + s,
                run_name=run_name,
                method=args.method,
                config=args.config,
                data_root=args.data_root,
                waterbirds_dir=args.waterbirds_dir,
                dataset_name=dataset_name,
                train_seed=s,
                base_lr=float(best_row["base_lr"]),
                classifier_lr=float(best_row["classifier_lr"]),
                grad_weight=float(best_row["grad_weight"]) if best_row.get("grad_weight") is not None else None,
                cam_weight=float(best_row["cam_weight"]) if best_row.get("cam_weight") is not None else None,
                abn_cls_weight=float(best_row["abn_cls_weight"]) if best_row.get("abn_cls_weight") is not None else None,
                abn_att_weight=float(best_row["abn_att_weight"]) if best_row.get("abn_att_weight") is not None else None,
                weight_decay=float(best_row["weight_decay"]),
                python_exe=python_exe,
                logs_dir=post_logs_dir,
                extra_overrides=args.overrides,
            )

            if args.post_keep == "none":
                cleanup_run_dir(row.get("run_dir"))

            out_row = {
                "seed": s,
                "name": row.get("name"),
                "method": row.get("method"),
                "base_lr": row.get("base_lr"),
                "classifier_lr": row.get("classifier_lr"),
                "grad_weight": row.get("grad_weight"),
                "cam_weight": row.get("cam_weight"),
                "abn_cls_weight": row.get("abn_cls_weight"),
                "abn_att_weight": row.get("abn_att_weight"),
                "weight_decay": row.get("weight_decay"),
                "best_balanced_val_acc": row.get("best_balanced_val_acc"),
                "test_acc": row.get("test_acc"),
                "balanced_test_acc": row.get("balanced_test_acc"),
                "per_group": row.get("per_group"),
                "worst_group": row.get("worst_group"),
                "checkpoint": row.get("checkpoint"),
                "log_path": row.get("log_path"),
                "seconds": row.get("seconds"),
                "run_dir": row.get("run_dir"),
                "sweep_best_trial": int(best_row["trial"]),
            }
            write_row(post_csv, out_row, post_header)
            post_rows.append(out_row)
            print(
                f"[POST] seed={s} best_balanced_val_acc={out_row['best_balanced_val_acc']} "
                f"per_group={out_row['per_group']} worst_group={out_row['worst_group']}",
                flush=True,
            )

        def _mean_std(key):
            vals = [r.get(key) for r in post_rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None, None
            arr = np.array(vals, dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        m_pg, s_pg = _mean_std("per_group")
        m_wg, s_wg = _mean_std("worst_group")
        m_bte, s_bte = _mean_std("balanced_test_acc")
        print("[POST] Summary over seeds:", flush=True)
        if m_pg is not None:
            print(f"  per_group: {m_pg:.2f} +/- {s_pg:.2f}", flush=True)
        if m_wg is not None:
            print(f"  worst_group: {m_wg:.2f} +/- {s_wg:.2f}", flush=True)
        if m_bte is not None:
            print(f"  balanced_test_acc: {m_bte:.2f} +/- {s_bte:.2f}", flush=True)
        print(f"[POST] Wrote: {post_csv}", flush=True)


if __name__ == "__main__":
    main()
