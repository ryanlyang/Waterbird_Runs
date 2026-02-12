import argparse
import csv
import os
import time
from types import SimpleNamespace

import numpy as np

import run_guided_redmeat as rgw


def loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def write_row(csv_path, row, header):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_runtime_summary(tag, rows, num_epochs):
    secs = [float(r["seconds"]) for r in rows if r.get("seconds") is not None]
    if not secs:
        print(f"[TIME] {tag}: no successful trials to summarize.")
        return
    arr = np.array(secs, dtype=float)
    med_trial_min = float(np.median(arr) / 60.0)
    total_gpu_hours = float(np.sum(arr) / 3600.0)
    print(f"[TIME] {tag}: median min/trial={med_trial_min:.4f} | total tuning GPU-hours={total_gpu_hours:.4f}")
    if num_epochs is not None and num_epochs > 0:
        med_epoch_min = float(np.median(arr / float(num_epochs)) / 60.0)
        print(f"[TIME] {tag}: median min/epoch={med_epoch_min:.4f} (epochs/trial={int(num_epochs)})")
    else:
        print(f"[TIME] {tag}: median min/epoch=N/A")


def run_trial(trial_id, args, rng, sampler_name):
    t0 = time.time()
    if sampler_name == "random":
        attn_epoch = int(rng.integers(args.attn_min, args.attn_max + 1))
        kl_lambda = loguniform(rng, args.kl_min, args.kl_max)
        base_lr = loguniform(rng, args.base_lr_min, args.base_lr_max)
        classifier_lr = loguniform(rng, args.cls_lr_min, args.cls_lr_max)
        lr2_mult = loguniform(rng, args.lr2_mult_min, args.lr2_mult_max)
    else:
        attn_epoch = int(args.trial.suggest_int("attention_epoch", args.attn_min, args.attn_max))
        kl_lambda = float(args.trial.suggest_float("kl_lambda", args.kl_min, args.kl_max, log=True))
        base_lr = float(args.trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
        classifier_lr = float(args.trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))
        lr2_mult = float(args.trial.suggest_float("lr2_mult", args.lr2_mult_min, args.lr2_mult_max, log=True))

    # Fixed policy: KL increment is derived from KL lambda, not swept.
    kl_incr = 0.1 * kl_lambda

    rgw.base_lr = base_lr
    rgw.classifier_lr = classifier_lr
    rgw.lr2_mult = lr2_mult
    rgw.SEED = args.seed

    run_args = SimpleNamespace(
        data_path=args.data_path,
        gt_path=args.gt_path,
        split_col=args.split_col,
        mask_brighten=args.mask_brighten,
    )
    best_balanced_val, test_acc, balanced_test, worst_class, ckpt = rgw.run_single(
        run_args, attn_epoch, kl_lambda, kl_incr
    )

    row = {
        "trial": trial_id,
        "attention_epoch": attn_epoch,
        "kl_lambda": kl_lambda,
        "kl_incr": kl_incr,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "lr2_mult": lr2_mult,
        "best_balanced_val_acc": best_balanced_val,
        "test_acc": test_acc,
        "balanced_test_acc": balanced_test,
        "worst_class_acc": worst_class,
        "checkpoint": ckpt,
        "sampler": sampler_name,
        "seconds": int(time.time() - t0),
    }
    return row


def rerun_best_across_gt_paths(args, best_row):
    post_csv = args.post_output_csv
    if post_csv is None:
        root, ext = os.path.splitext(args.output_csv)
        post_csv = f"{root}_best_seeds{ext or '.csv'}"

    post_header = [
        "phase",
        "seed",
        "attention_epoch",
        "kl_lambda",
        "kl_incr",
        "base_lr",
        "classifier_lr",
        "lr2_mult",
        "best_balanced_val_acc",
        "test_acc",
        "balanced_test_acc",
        "worst_class_acc",
        "checkpoint",
        "sweep_best_trial",
        "sampler",
        "seconds",
        "mask_root",
    ]

    best_attn_epoch = int(best_row["attention_epoch"])
    best_kl_lambda = float(best_row["kl_lambda"])
    best_kl_incr = float(best_row["kl_incr"])
    best_base_lr = float(best_row["base_lr"])
    best_classifier_lr = float(best_row["classifier_lr"])
    best_lr2_mult = float(best_row["lr2_mult"])

    seeds = list(range(args.post_seed_start, args.post_seed_start + args.post_seeds))
    print(f"[POST] Rerunning best hyperparameters for {len(seeds)} seeds: {seeds}")

    gt_paths = [("best5_primary", args.gt_path)] + [
        (f"best5_extra{i+1}", p) for i, p in enumerate(args.extra_gt_path)
    ]

    post_rows = []
    for phase, gt_path in gt_paths:
        print(f"[POST] phase={phase} gt_path={gt_path}")
        for s in seeds:
            rgw.base_lr = best_base_lr
            rgw.classifier_lr = best_classifier_lr
            rgw.lr2_mult = best_lr2_mult
            rgw.SEED = s

            run_args = SimpleNamespace(
                data_path=args.data_path,
                gt_path=gt_path,
                split_col=args.split_col,
                mask_brighten=args.mask_brighten,
            )
            t0 = time.time()
            best_balanced_val, test_acc, balanced_test, worst_class, ckpt = rgw.run_single(
                run_args, best_attn_epoch, best_kl_lambda, best_kl_incr
            )

            out_row = {
                "phase": phase,
                "seed": s,
                "attention_epoch": best_attn_epoch,
                "kl_lambda": best_kl_lambda,
                "kl_incr": best_kl_incr,
                "base_lr": best_base_lr,
                "classifier_lr": best_classifier_lr,
                "lr2_mult": best_lr2_mult,
                "best_balanced_val_acc": best_balanced_val,
                "test_acc": test_acc,
                "balanced_test_acc": balanced_test,
                "worst_class_acc": worst_class,
                "checkpoint": ckpt,
                "sweep_best_trial": int(best_row["trial"]),
                "sampler": args.sampler,
                "seconds": int(time.time() - t0),
                "mask_root": gt_path,
            }
            write_row(post_csv, out_row, post_header)
            post_rows.append(out_row)
            print(
                f"[POST] phase={phase} seed={s} best_balanced_val_acc={best_balanced_val:.4f} "
                f"test_acc={test_acc:.2f} balanced_test={balanced_test:.2f} worst_class={worst_class:.2f}",
                flush=True,
            )
    print_runtime_summary("post_best_seeds", post_rows, rgw.num_epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="RedMeat dataset root with meta/all_images.csv and images/")
    parser.add_argument("gt_path", help="Primary mask root for tuning + post reruns")
    parser.add_argument(
        "--extra-gt-path",
        action="append",
        default=[],
        help="Additional mask root(s) for post best-5 reruns. Repeat flag for multiple paths.",
    )
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--mask-brighten", type=float, default=8.0)

    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default="guided_redmeat_sweep.csv")
    parser.add_argument("--attn-min", type=int, default=0)
    parser.add_argument("--attn-max", type=int, default=rgw.num_epochs - 1)
    parser.add_argument("--kl-min", type=float, default=1.0)
    parser.add_argument("--kl-max", type=float, default=500.0)
    parser.add_argument("--base-lr-min", type=float, default=1e-6)
    parser.add_argument("--base-lr-max", type=float, default=1e-3)
    parser.add_argument("--cls-lr-min", type=float, default=1e-5)
    parser.add_argument("--cls-lr-max", type=float, default=1e-2)
    parser.add_argument("--lr2-mult-min", type=float, default=1e-1)
    parser.add_argument("--lr2-mult-max", type=float, default=3.0)
    parser.add_argument("--sampler", choices=["tpe", "random"], default="tpe")

    parser.add_argument("--post-seeds", type=int, default=5)
    parser.add_argument("--post-seed-start", type=int, default=0)
    parser.add_argument("--post-output-csv", default=None)
    args = parser.parse_args()

    header = [
        "trial",
        "attention_epoch",
        "kl_lambda",
        "kl_incr",
        "base_lr",
        "classifier_lr",
        "lr2_mult",
        "best_balanced_val_acc",
        "test_acc",
        "balanced_test_acc",
        "worst_class_acc",
        "checkpoint",
        "sampler",
        "seconds",
    ]

    rng = np.random.default_rng(args.seed)

    if args.sampler == "tpe":
        try:
            import optuna
        except Exception as exc:
            print(f"[SWEEP] Optuna not available ({exc}); falling back to random search.")
            args.sampler = "random"

    best_row = None
    sweep_rows = []
    if args.sampler == "random":
        for trial_id in range(args.n_trials):
            row = run_trial(trial_id, args, rng, "random")
            write_row(args.output_csv, row, header)
            sweep_rows.append(row)
            if best_row is None or row["best_balanced_val_acc"] > best_row["best_balanced_val_acc"]:
                best_row = row
            print(f"[SWEEP] Trial {trial_id} done. best_balanced_val_acc={row['best_balanced_val_acc']:.4f}")
    else:
        import optuna

        def objective(trial):
            nonlocal best_row
            args.trial = trial
            row = run_trial(trial.number, args, rng, "tpe")
            write_row(args.output_csv, row, header)
            sweep_rows.append(row)
            if best_row is None or row["best_balanced_val_acc"] > best_row["best_balanced_val_acc"]:
                best_row = row
            print(f"[SWEEP] Trial {trial.number} done. best_balanced_val_acc={row['best_balanced_val_acc']:.4f}")
            return row["best_balanced_val_acc"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)

    if best_row is not None:
        print("[SWEEP] Best trial:")
        for k in header:
            print(f"  {k}: {best_row[k]}")
        if args.post_seeds > 0:
            rerun_best_across_gt_paths(args, best_row)
    print_runtime_summary("sweep", sweep_rows, rgw.num_epochs)


if __name__ == "__main__":
    main()
