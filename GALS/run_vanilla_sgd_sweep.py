#!/usr/bin/env python3
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


def write_row(csv_path, row, header):
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _parse_metric_line(line):
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
    run_name,
    config,
    data_root,
    food_dir,
    dataset_name,
    train_seed,
    lr,
    weight_decay,
    momentum,
    nesterov,
    python_exe,
    logs_dir,
    extra_overrides=None,
):
    cmd = [
        python_exe,
        "-u",
        "main.py",
        "--config",
        config,
        "--name",
        run_name,
        f"DATA.ROOT={data_root}",
        f"DATA.FOOD_SUBSET_DIR={food_dir}",
        f"DATA.SUBDIR={food_dir}",
        f"SEED={train_seed}",
        f"EXP.BASE.LR={lr}",
        f"EXP.CLASSIFIER.LR={lr}",
        f"EXP.WEIGHT_DECAY={weight_decay}",
        f"EXP.MOMENTUM={momentum}",
        f"EXP.NESTEROV={str(bool(nesterov))}",
        "EXP.AUX_LOSSES_ON_VAL=False",
    ]
    if extra_overrides:
        cmd.extend(list(extra_overrides))

    os.makedirs(logs_dir, exist_ok=True)
    trial_log = os.path.join(logs_dir, f"{run_name}.log")

    best_balanced_val = None
    checkpoint_used = None
    test_metrics = {}
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

            parsed = _parse_metric_line(line)
            if parsed:
                k, v = parsed
                if k == "balanced_val_acc":
                    if best_balanced_val is None or v > best_balanced_val:
                        best_balanced_val = v
                if k.endswith("_test_acc") or k in ("test_acc", "balanced_test_acc"):
                    test_metrics[k] = v

            if line.startswith("TEST SET RESULTS FOR CHECKPOINT"):
                checkpoint_used = line.strip().split("CHECKPOINT", 1)[-1].strip()

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Trial {trial_id} failed with exit code {rc}. See {trial_log}")

    def to_pct(x):
        if x is None:
            return None
        return x * 100.0 if x <= 1.0 else x

    elapsed_s = time.time() - start
    return {
        "trial": trial_id,
        "name": run_name,
        "seed": train_seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "nesterov": bool(nesterov),
        "best_balanced_val_acc": best_balanced_val,
        "test_acc": to_pct(test_metrics.get("test_acc")),
        "balanced_test_acc": to_pct(test_metrics.get("balanced_test_acc")),
        "checkpoint": checkpoint_used,
        "log_path": trial_log,
        "seconds": int(elapsed_s),
        "run_dir": run_dir,
    }


def print_runtime_summary(tag, rows, num_epochs=None):
    secs = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
    if not secs:
        print(f"[TIME] {tag}: no successful trials to summarize.", flush=True)
        return
    arr = np.array(secs, dtype=float)
    med_trial_min = float(np.median(arr) / 60.0)
    total_gpu_hours = float(np.sum(arr) / 3600.0)
    print(
        f"[TIME] {tag}: median min/trial={med_trial_min:.2f} | total tuning GPU-hours={total_gpu_hours:.2f}",
        flush=True,
    )
    if num_epochs is not None and num_epochs > 0:
        med_epoch_min = float(np.median(arr / float(num_epochs)) / 60.0)
        print(
            f"[TIME] {tag}: median min/epoch={med_epoch_min:.4f} (epochs/trial={int(num_epochs)})",
            flush=True,
        )


def _maybe_import_omegaconf():
    try:
        from omegaconf import OmegaConf  # type: ignore

        return OmegaConf
    except Exception:
        return None


def _config_num_epochs(config_path):
    OmegaConf = _maybe_import_omegaconf()
    if OmegaConf is None:
        return 150
    try:
        cfg = OmegaConf.load(config_path)
        return int(cfg.EXP.NUM_EPOCHS)
    except Exception:
        return 150


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/food_vanilla.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--food-dir", default="food-101-redmeat")
    parser.add_argument("--dataset", default="food_subset")

    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="sampler seed")
    parser.add_argument("--train-seed", type=int, default=0, help="train seed during sweep")
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--max-hours", type=float, default=None)

    parser.add_argument("--lr-min", type=float, default=3e-5)
    parser.add_argument("--lr-max", type=float, default=3e-2)
    parser.add_argument("--weight-decay-min", type=float, default=1e-7)
    parser.add_argument("--weight-decay-max", type=float, default=1e-3)
    parser.add_argument("--momentum-min", type=float, default=0.80)
    parser.add_argument("--momentum-max", type=float, default=0.98)

    parser.add_argument("--output-csv", default="vanilla_sgd_sweep.csv")
    parser.add_argument("--logs-dir", default="vanilla_sgd_sweep_logs")
    parser.add_argument("--keep", choices=["best", "all", "none"], default="best")
    parser.add_argument("--run-name-prefix", default=None)

    parser.add_argument("--post-seeds", type=int, default=5)
    parser.add_argument("--post-seed-start", type=int, default=0)
    parser.add_argument("--post-output-csv", default=None)
    parser.add_argument("--post-logs-dir", default=None)
    parser.add_argument("--post-keep", choices=["all", "none"], default="all")

    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra OmegaConf overrides passed through to main.py",
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.data_root, args.food_dir, "meta", "all_images.csv")):
        raise FileNotFoundError(
            f"Missing: {os.path.join(args.data_root, args.food_dir, 'meta', 'all_images.csv')}"
        )

    header = [
        "trial",
        "name",
        "seed",
        "lr",
        "weight_decay",
        "momentum",
        "nesterov",
        "best_balanced_val_acc",
        "test_acc",
        "balanced_test_acc",
        "checkpoint",
        "log_path",
        "seconds",
        "sampler",
        "run_dir",
    ]

    rng = np.random.default_rng(args.seed)
    python_exe = sys.executable
    num_epochs = _config_num_epochs(args.config)

    job_tag = os.environ.get("SLURM_JOB_ID", "nojid")
    if args.run_name_prefix:
        run_name_prefix = args.run_name_prefix
    else:
        run_name_prefix = f"vanilla_sgd_{args.food_dir}_{job_tag}"

    if args.sampler == "tpe":
        try:
            import optuna  # noqa: F401
        except Exception as exc:
            print(f"[SWEEP] Optuna unavailable ({exc}), falling back to random.", flush=True)
            args.sampler = "random"

    best_row = None
    best_dir = None
    sweep_rows = []
    start_time = time.time()

    def cleanup_run_dir(path):
        if not path:
            return
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

    def run_and_record(trial_id, lr, wd, momentum, nesterov, sampler_name):
        nonlocal best_row, best_dir
        run_name = f"{run_name_prefix}_trial_{trial_id:03d}"
        row = run_one_trial(
            trial_id,
            run_name=run_name,
            config=args.config,
            data_root=args.data_root,
            food_dir=args.food_dir,
            dataset_name=args.dataset,
            train_seed=args.train_seed,
            lr=lr,
            weight_decay=wd,
            momentum=momentum,
            nesterov=nesterov,
            python_exe=python_exe,
            logs_dir=args.logs_dir,
            extra_overrides=args.overrides,
        )
        row["sampler"] = sampler_name
        write_row(args.output_csv, row, header)
        sweep_rows.append(row)

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

    if args.sampler == "random":
        for trial_id in range(args.n_trials):
            if args.max_hours is not None and (time.time() - start_time) >= args.max_hours * 3600:
                print(f"[SWEEP] Reached max-hours={args.max_hours}; stopping before trial {trial_id}.", flush=True)
                break
            lr = loguniform(rng, args.lr_min, args.lr_max)
            wd = loguniform(rng, args.weight_decay_min, args.weight_decay_max)
            momentum = float(rng.uniform(args.momentum_min, args.momentum_max))
            nesterov = bool(rng.integers(0, 2))
            try:
                row = run_and_record(trial_id, lr, wd, momentum, nesterov, "random")
                print(
                    f"[SWEEP] Trial {trial_id} done. best_balanced_val_acc={row['best_balanced_val_acc']}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[SWEEP] Trial {trial_id} failed: {exc}", flush=True)
    else:
        import optuna

        def objective(trial):
            lr = float(trial.suggest_float("lr", args.lr_min, args.lr_max, log=True))
            wd = float(trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max, log=True))
            momentum = float(trial.suggest_float("momentum", args.momentum_min, args.momentum_max))
            nesterov = bool(trial.suggest_categorical("nesterov", [False, True]))

            row = run_and_record(trial.number, lr, wd, momentum, nesterov, "tpe")
            print(
                f"[SWEEP] Trial {trial.number} done. best_balanced_val_acc={row['best_balanced_val_acc']}",
                flush=True,
            )
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

    post_rows = []
    if best_row is not None and args.post_seeds > 0:
        post_csv = args.post_output_csv
        if post_csv is None:
            root, ext = os.path.splitext(args.output_csv)
            post_csv = f"{root}_best_seeds{ext or '.csv'}"
        post_logs_dir = args.post_logs_dir or f"{args.logs_dir}_best_seeds"

        post_header = [
            "seed",
            "name",
            "lr",
            "weight_decay",
            "momentum",
            "nesterov",
            "best_balanced_val_acc",
            "test_acc",
            "balanced_test_acc",
            "checkpoint",
            "log_path",
            "seconds",
            "run_dir",
            "sweep_best_trial",
        ]

        seeds = list(range(args.post_seed_start, args.post_seed_start + args.post_seeds))
        print(f"[POST] Rerunning best hyperparameters for {len(seeds)} seeds: {seeds}", flush=True)

        for s in seeds:
            run_name = f"{run_name_prefix}_best_seed{s}"
            row = run_one_trial(
                100000 + s,
                run_name=run_name,
                config=args.config,
                data_root=args.data_root,
                food_dir=args.food_dir,
                dataset_name=args.dataset,
                train_seed=s,
                lr=float(best_row["lr"]),
                weight_decay=float(best_row["weight_decay"]),
                momentum=float(best_row["momentum"]),
                nesterov=bool(best_row["nesterov"]),
                python_exe=python_exe,
                logs_dir=post_logs_dir,
                extra_overrides=args.overrides,
            )
            if args.post_keep == "none":
                cleanup_run_dir(row.get("run_dir"))

            out_row = {
                "seed": s,
                "name": row.get("name"),
                "lr": row.get("lr"),
                "weight_decay": row.get("weight_decay"),
                "momentum": row.get("momentum"),
                "nesterov": row.get("nesterov"),
                "best_balanced_val_acc": row.get("best_balanced_val_acc"),
                "test_acc": row.get("test_acc"),
                "balanced_test_acc": row.get("balanced_test_acc"),
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
                f"test_acc={out_row['test_acc']}",
                flush=True,
            )

        print(f"[POST] Wrote: {post_csv}", flush=True)

    print_runtime_summary("sweep", sweep_rows, num_epochs)
    if post_rows:
        print_runtime_summary("post_best_seeds", post_rows, num_epochs)


if __name__ == "__main__":
    main()

