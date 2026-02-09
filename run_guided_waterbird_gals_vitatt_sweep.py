import argparse
import csv
import os
import time
from types import SimpleNamespace

import numpy as np

import run_guided_waterbird_gals_vitatt as rgw


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
        kl_incr = loguniform(rng, args.kl_incr_min, args.kl_incr_max)
        base_lr = loguniform(rng, args.base_lr_min, args.base_lr_max)
        classifier_lr = loguniform(rng, args.cls_lr_min, args.cls_lr_max)
    else:
        attn_epoch = int(args.trial.suggest_int("attention_epoch", args.attn_min, args.attn_max))
        kl_lambda = float(args.trial.suggest_float("kl_lambda", args.kl_min, args.kl_max, log=True))
        kl_incr = float(args.trial.suggest_float("kl_incr", args.kl_incr_min, args.kl_incr_max, log=True))
        base_lr = float(args.trial.suggest_float("base_lr", args.base_lr_min, args.base_lr_max, log=True))
        classifier_lr = float(args.trial.suggest_float("classifier_lr", args.cls_lr_min, args.cls_lr_max, log=True))

    rgw.base_lr = base_lr
    rgw.classifier_lr = classifier_lr
    rgw.SEED = args.seed

    run_args = SimpleNamespace(
        data_path=args.data_path,
        att_path=args.att_path,
        att_key=args.att_key,
        att_combine=args.att_combine,
        att_norm01=args.att_norm01,
        att_brighten=args.att_brighten,
    )
    best_balanced_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(
        run_args, attn_epoch, kl_lambda, kl_incr
    )

    row = {
        "trial": trial_id,
        "attention_epoch": attn_epoch,
        "kl_lambda": kl_lambda,
        "kl_incr": kl_incr,
        "base_lr": base_lr,
        "classifier_lr": classifier_lr,
        "best_balanced_val_acc": best_balanced_val,
        "test_acc": test_acc,
        "per_group": per_group,
        "worst_group": worst_group,
        "checkpoint": ckpt,
        "sampler": sampler_name,
        "seconds": int(time.time() - t0),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Waterbirds dataset root with metadata.csv")
    parser.add_argument("att_path", help="Folder containing GALS ViT attention .pth files")
    parser.add_argument("--att-key", default="unnormalized_attentions", choices=["unnormalized_attentions", "attentions"])
    parser.add_argument("--att-combine", default="mean", choices=["mean", "max"])
    parser.add_argument("--att-norm01", action="store_true", default=True)
    parser.add_argument("--att-brighten", type=float, default=1.0)

    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default="guided_waterbird_galsvit_sweep.csv")
    parser.add_argument("--attn-min", type=int, default=0)
    parser.add_argument("--attn-max", type=int, default=rgw.num_epochs - 1)
    parser.add_argument("--kl-min", type=float, default=1.0)
    parser.add_argument("--kl-max", type=float, default=300.0)
    parser.add_argument("--kl-incr-min", type=float, default=1e-1)
    parser.add_argument("--kl-incr-max", type=float, default=30.0)
    parser.add_argument("--base-lr-min", type=float, default=1e-5)
    parser.add_argument("--base-lr-max", type=float, default=1e-3)
    parser.add_argument("--cls-lr-min", type=float, default=1e-4)
    parser.add_argument("--cls-lr-max", type=float, default=1e-2)
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
        "best_balanced_val_acc",
        "test_acc",
        "per_group",
        "worst_group",
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

    if best_row is not None and args.post_seeds > 0:
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
            "best_balanced_val_acc",
            "test_acc",
            "per_group",
            "worst_group",
            "checkpoint",
            "sweep_best_trial",
            "sampler",
            "seconds",
        ]

        best_attn_epoch = int(best_row["attention_epoch"])
        best_kl_lambda = float(best_row["kl_lambda"])
        best_kl_incr = float(best_row["kl_incr"])
        best_base_lr = float(best_row["base_lr"])
        best_classifier_lr = float(best_row["classifier_lr"])

        seeds = list(range(args.post_seed_start, args.post_seed_start + args.post_seeds))
        print(f"[POST] Rerunning best hyperparameters for {len(seeds)} seeds: {seeds}")
        post_rows = []
        for s in seeds:
            rgw.base_lr = best_base_lr
            rgw.classifier_lr = best_classifier_lr
            rgw.SEED = s

            run_args = SimpleNamespace(
                data_path=args.data_path,
                att_path=args.att_path,
                att_key=args.att_key,
                att_combine=args.att_combine,
                att_norm01=args.att_norm01,
                att_brighten=args.att_brighten,
            )
            t0 = time.time()
            best_balanced_val, test_acc, per_group, worst_group, ckpt = rgw.run_single(
                run_args, best_attn_epoch, best_kl_lambda, best_kl_incr
            )

            out_row = {
                "phase": "best5",
                "seed": s,
                "attention_epoch": best_attn_epoch,
                "kl_lambda": best_kl_lambda,
                "kl_incr": best_kl_incr,
                "base_lr": best_base_lr,
                "classifier_lr": best_classifier_lr,
                "best_balanced_val_acc": best_balanced_val,
                "test_acc": test_acc,
                "per_group": per_group,
                "worst_group": worst_group,
                "checkpoint": ckpt,
                "sweep_best_trial": int(best_row["trial"]),
                "sampler": args.sampler,
                "seconds": int(time.time() - t0),
            }
            write_row(post_csv, out_row, post_header)
            post_rows.append(out_row)
            print(
                f"[POST] seed={s} best_balanced_val_acc={best_balanced_val:.4f} "
                f"test_acc={test_acc:.2f} worst_group={worst_group:.2f}",
                flush=True,
            )
        print_runtime_summary("post_best_seeds", post_rows, rgw.num_epochs)

    print_runtime_summary("sweep", sweep_rows, rgw.num_epochs)


if __name__ == "__main__":
    main()
