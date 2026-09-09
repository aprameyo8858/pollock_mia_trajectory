import argparse
import inspect
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import torch

from loss_traces.attacks import AttackConfig, RMIAAttack
from loss_traces.config import MODEL_DIR, STORAGE_DIR
from loss_traces.data_processing.data_processing import (
    get_no_shuffle_train_loader,
    get_num_classes,
)
from loss_traces.models.model import load_model
from loss_traces.results.final_model_metrics import get_final_model_metrics
from loss_traces.results.result_processing import (
    get_attackr_scores,
    get_lira_scores,
    get_rmia_scores,
    get_trace_reduction,
)

if os.path.exists("plot_style.mplstyle"):
  plt.style.use("plot_style.mplstyle")


def parse_args():
  parser = argparse.ArgumentParser(
      description="Evaluate MIA Artifacts and Trajectory Metrics"
  )
  parser.add_argument(
      "--exp_id", type=str, default="wrn28-2_CIFAR10", help="Experiment ID"
  )
  parser.add_argument(
      "--dataset", type=str, default="CIFAR10", help="Dataset name"
  )
  parser.add_argument(
      "--arch", type=str, default="wrn28-2", help="Model architecture"
  )
  parser.add_argument(
      "--n_shadows",
      type=int,
      default=16,
      help="Number of shadow models used for baseline score files",
  )
  parser.add_argument(
      "--batchsize",
      type=int,
      default=16,
      help="Batch size for evaluation loaders",
  )
  parser.add_argument(
      "--num_workers", type=int, default=4, help="Data loader workers"
  )
  parser.add_argument(
      "--device",
      type=str,
      default="cuda:0" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument(
      "--augment", action="store_true", default=True, help="Augmentation flag"
  )
  parser.add_argument(
      "--rho",
      type=float,
      default=1.0,
      help="Smoothness Lipschitz parameter for Ghadimi-Lan weights",
  )
  parser.add_argument(
      "--initial_lr",
      type=float,
      default=0.1,
      help="Initial learning rate for Ghadimi-Lan schedule",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="./results_analysis",
      help="Output directory for CSVs and figures",
  )
  return parser.parse_args()


def safe_call(func, *args, **kwargs):
  """Dispatches arguments dynamically according to function parameter signatures."""
  sig = inspect.signature(func)
  filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
  return func(*args, **filtered_kwargs)


def compute_gl_weights(num_epochs, initial_lr=0.1, rho=1.0):
  """Calculates Ghadimi-Lan convergence weights: eta_t - 0.5 * rho * eta_t^2."""
  epochs = np.arange(num_epochs)
  lr_t = 0.5 * initial_lr * (1.0 + np.cos(np.pi * epochs / num_epochs))
  gl_w = np.maximum(lr_t - 0.5 * rho * (lr_t**2), 1e-6)
  return gl_w


def find_parquet_file(directory, exp_id):
  """Locates target parquet trajectories across naming conventions."""
  candidates = [
      os.path.join(directory, f"{exp_id}_target.pq"),
      os.path.join(directory, f"{exp_id}.pq"),
      os.path.join(directory, f"{exp_id}_target.parquet"),
      os.path.join(directory, f"{exp_id}.parquet"),
  ]
  for path in candidates:
    if os.path.exists(path):
      return path
  return None


def load_trajectories(exp_id):
  """Reads per-epoch loss and CSG trajectory parquet data."""
  losses_dir = os.path.join(STORAGE_DIR, "losses")
  loss_path = find_parquet_file(losses_dir, exp_id)

  if loss_path is None:
    raise FileNotFoundError(
        f"Loss trajectory parquet not found for {exp_id} under {losses_dir}"
    )
  losses_df = pd.read_parquet(loss_path)

  csg_dir = os.path.join(STORAGE_DIR, "csg_norms")
  csg_path = find_parquet_file(csg_dir, exp_id)

  csg_df = None
  if csg_path is not None:
    csg_df = pd.read_parquet(csg_path)
  else:
    print(
        f"[Warning] CSG trajectory not found under {csg_dir}. CSG metrics will"
        " default to zeros."
    )

  return losses_df, csg_df


def initialize_model_and_loaders(config):
  attack_loaders = [
      get_no_shuffle_train_loader(
          config["dataset"],
          config["arch"],
          config["batchsize"],
          config["num_workers"],
      )
  ]
  if config["augment"]:
    attack_loaders.append(
        get_no_shuffle_train_loader(
            config["dataset"],
            config["arch"],
            config["batchsize"],
            config["num_workers"],
            mirror_all=True,
        )
    )

  model = load_model(config["arch"], get_num_classes(config["dataset"])).to(
      config["device"]
  )
  return model, attack_loaders


def evaluate_table1_lira(
    df,
    metrics_dict,
    fpr_level=0.001,
    k_fracs=[0.01, 0.03, 0.05, 0.10, 0.20, 0.50],
):
  """Computes Precision@k and Recall@k against LiRA ground truth at a fixed FPR threshold."""
  members = df[df["target_trained_on"] == True]
  fprs, _, thresholds = roc_curve(
      df["target_trained_on"], df["lira_score"], drop_intermediate=False
  )
  target_idx = np.where(fprs <= fpr_level)[0][-1]
  tau = thresholds[target_idx]
  vulnerable_set = set(members[members["lira_score"] >= tau].index)
  total_vuln = len(vulnerable_set)

  results = {m: {} for m in metrics_dict}
  for m_name, (col, ascending) in metrics_dict.items():
    sorted_members = members.sort_values(by=col, ascending=ascending)
    for k in k_fracs:
      n = int(np.round(k * len(members)))
      top_k_idx = set(sorted_members.head(n).index)
      tp = len(top_k_idx & vulnerable_set)

      prec = float(tp) / n if n > 0 else 0.0
      rec = float(tp) / total_vuln if total_vuln > 0 else 0.0

      results[m_name][f"k={int(k*100)}% Precision"] = prec
      results[m_name][f"k={int(k*100)}% Recall"] = rec

  res_df = pd.DataFrame.from_dict(results, orient="index")
  ordered_cols = []
  for k in k_fracs:
    ordered_cols.extend([f"k={int(k*100)}% Precision", f"k={int(k*100)}% Recall"])
  return res_df[ordered_cols]


def evaluate_table5_union(df, metrics_dict, fpr_level=0.001, k_frac=0.01):
  """Computes Precision@k and Recall@k against the union of LiRA, Attack R, and RMIA."""
  members = df[df["target_trained_on"] == True]
  union_vuln = set()

  for atk in ["lira_score", "attackr_score", "rmia_score"]:
    if atk in df.columns:
      fprs, _, thresholds = roc_curve(
          df["target_trained_on"], df[atk], drop_intermediate=False
      )
      target_idx = np.where(fprs <= fpr_level)[0][-1]
      tau = thresholds[target_idx]
      union_vuln.update(members[members[atk] >= tau].index)

  total_union = len(union_vuln)
  n = int(np.round(k_frac * len(members)))

  records = []
  for m_name, (col, ascending) in metrics_dict.items():
    sorted_members = members.sort_values(by=col, ascending=ascending)
    top_k_idx = set(sorted_members.head(n).index)
    tp = len(top_k_idx & union_vuln)

    prec = float(tp) / n if n > 0 else 0.0
    rec = float(tp) / total_union if total_union > 0 else 0.0

    records.append(
        {"Metric": m_name, "Precision on union": prec, "Recall on union": rec}
    )

  return pd.DataFrame(records)


def compute_precision_curve(df, metric_col, ascending=False, k_frac=0.01):
  """Calculates Precision@k at variable FPR thresholds for plotting."""
  members = df[df["target_trained_on"] == True]
  fprs, _, thresholds = roc_curve(
      df["target_trained_on"], df["lira_score"], drop_intermediate=False
  )

  target_fprs = np.logspace(-5, 0, 100)
  sorted_members = members.sort_values(by=metric_col, ascending=ascending)
  n = int(np.round(k_frac * len(members)))
  top_k_idx = set(sorted_members.head(n).index)

  precisions = []
  actual_fprs = []

  for f in target_fprs:
    valid_idxs = np.where(fprs <= f)[0]
    if len(valid_idxs) == 0:
      continue
    idx = valid_idxs[-1]
    tau = thresholds[idx]
    vulnerable_set = set(members[members["lira_score"] >= tau].index)
    tp = len(top_k_idx & vulnerable_set)
    precisions.append(float(tp) / n if n > 0 else 0.0)
    actual_fprs.append(fprs[idx] if fprs[idx] > 0 else 1e-5)

  return np.array(actual_fprs), np.array(precisions)


def main():
  args = parse_args()
  os.makedirs(args.output_dir, exist_ok=True)
  plots_dir = os.path.join(args.output_dir, "plots")
  os.makedirs(plots_dir, exist_ok=True)

  print(
      f"[1/5] Loading target model and baseline MIA scores for {args.exp_id}"
      f" (n_shadows={args.n_shadows})..."
  )

  # Load shadow attack score frames with explicit n_shadows resolution
  df = safe_call(
      get_lira_scores,
      args.exp_id,
      target_id="target",
      n_shadows=args.n_shadows,
  )
  df["attackr_score"] = safe_call(
      get_attackr_scores,
      args.exp_id,
      target_id="target",
      n_shadows=args.n_shadows,
  )
  df["rmia_score"] = safe_call(
      get_rmia_scores,
      args.exp_id,
      "target",
      n_shadows=args.n_shadows,
      return_full_df=False,
  )

  # Compute/cache final model instance metrics
  model_dir = os.path.join(MODEL_DIR, args.exp_id)
  target_model_file = os.path.join(model_dir, "target")
  if not os.path.exists(target_model_file):
    target_model_file = os.path.join(model_dir, "model")

  saves = torch.load(target_model_file, map_location=args.device)
  model, data_loader = initialize_model_and_loaders(vars(args))
  model.load_state_dict(saves["model_state_dict"])

  fin_losses_csv = os.path.join(args.output_dir, f"fin_losses_{args.exp_id}.csv")
  grads_csv = os.path.join(args.output_dir, f"grads_{args.exp_id}.csv")
  shap_csv = os.path.join(args.output_dir, f"shap_{args.exp_id}.csv")

  if not os.path.exists(fin_losses_csv):
    get_final_model_metrics(
        model, data_loader[0], metrics=["loss"]
    ).to_csv(fin_losses_csv)
  if not os.path.exists(grads_csv):
    get_final_model_metrics(
        model, data_loader[0], metrics=["grads"]
    ).to_csv(grads_csv)
  if not os.path.exists(shap_csv):
    get_final_model_metrics(
        model, data_loader[0], metrics=["shap"]
    ).to_csv(shap_csv)

  df["param_grads"] = pd.read_csv(grads_csv)["param_grad_norm"]
  df["input_grads"] = pd.read_csv(grads_csv)["input_grad_norm"]
  df["shap"] = pd.read_csv(shap_csv)["shap_norm"]
  df["confidence"] = pd.read_csv(fin_losses_csv)["confidence"]
  df["loss"] = pd.read_csv(fin_losses_csv)["loss"]
  df["lt_iqr"] = safe_call(
      get_trace_reduction, args.exp_id, reduction="iqr"
  )

  print("[2/5] Loading training trajectory artifacts (losses and csg_norms)...")
  losses_df, csg_df = load_trajectories(args.exp_id)
  num_epochs = losses_df.shape[1]
  gl_weights = compute_gl_weights(
      num_epochs, initial_lr=args.initial_lr, rho=args.rho
  )

  # Compute CSL and GL-CSL
  df["csl"] = losses_df.sum(axis=1).values
  df["gl_csl"] = (losses_df.values * gl_weights).sum(axis=1)

  # Compute CSG, GL-CSG, and Late-CSG tracking windows
  if csg_df is not None:
    csg_mat = csg_df.values
    df["csg"] = csg_mat.sum(axis=1)
    df["gl_csg"] = (csg_mat * gl_weights).sum(axis=1)

    idx_10 = int(np.floor(0.90 * num_epochs))
    idx_30 = int(np.floor(0.70 * num_epochs))
    idx_40 = int(np.floor(0.60 * num_epochs))

    df["late_csg_10"] = csg_mat[:, idx_10:].sum(axis=1)
    df["late_csg_30"] = csg_mat[:, idx_30:].sum(axis=1)
    df["late_csg_40"] = csg_mat[:, idx_40:].sum(axis=1)
  else:
    for col in [
        "csg",
        "gl_csg",
        "late_csg_10",
        "late_csg_30",
        "late_csg_40",
    ]:
      df[col] = 0.0

  scores_path = os.path.join(args.output_dir, "all_sample_scores.csv")
  df.to_csv(scores_path, index=False)
  print(f"Full per-sample scores exported to: {scores_path}")

  # Registry: metric_key -> (dataframe_column, sort_ascending)
  eval_registry = {
      "loss": ("loss", False),
      "confidence": ("confidence", True),  # Lower confidence -> Higher risk
      "param_grads": ("param_grads", False),
      "input_grads": ("input_grads", False),
      "shap": ("shap", False),
      "lt_iqr": ("lt_iqr", False),
      "csl": ("csl", False),
      "gl_csl": ("gl_csl", False),
      "csg": ("csg", False),
      "gl_csg": ("gl_csg", False),
      "late_csg_10": ("late_csg_10", False),
      "late_csg_30": ("late_csg_30", False),
      "late_csg_40": ("late_csg_40", False),
  }

  print("[3/5] Evaluating Table 1 (LiRA ground truth @ FPR=1e-3)...")
  table1_res = evaluate_table1_lira(df, eval_registry, fpr_level=0.001)
  table1_path = os.path.join(args.output_dir, "table1_lira_precision_recall.csv")
  table1_res.to_csv(table1_path)
  print(table1_res.to_string())

  print(
      "\n[4/5] Evaluating Table 5 (Union of LiRA, Attack R, RMIA @ FPR=1e-3,"
      " k=1%)..."
  )
  table5_res = evaluate_table5_union(
      df, eval_registry, fpr_level=0.001, k_frac=0.01
  )
  table5_path = os.path.join(
      args.output_dir, "table5_union_precision_recall.csv"
  )
  table5_res.to_csv(table5_path, index=False)
  print(table5_res.to_string())

  print("\n[5/5] Generating figures...")
  # Figure 1 Style: Precision@k=1% vs. FPR
  plt.figure(figsize=(9, 7))
  plot_metrics = [
      ("lt_iqr", "#0072B2", "-", 2.5, "LT-IQR"),
      ("loss", "#009E73", "-", 1.8, "Final Loss"),
      ("input_grads", "#CC79A7", "--", 1.8, "Input Grad Norm"),
      ("csg", "#D55E00", "-", 2.2, "CSG (Full)"),
      ("csl", "#56B4E9", "--", 1.8, "CSL"),
      ("gl_csg", "#E69F00", "-.", 2.0, "GL-CSG"),
      ("late_csg_30", "#000000", ":", 2.2, "Late-CSG (Last 30%)"),
  ]

  for key, color, ls, lw, lbl in plot_metrics:
    col_name, asc = eval_registry[key]
    fpr_x, prec_y = compute_precision_curve(
        df, col_name, ascending=asc, k_frac=0.01
    )
    plt.plot(fpr_x, prec_y, color=color, linestyle=ls, linewidth=lw, label=lbl)

  # Random guess reference
  members = df[df["target_trained_on"] == True]
  fprs, _, thresholds = roc_curve(
      df["target_trained_on"], df["lira_score"], drop_intermediate=False
  )
  ref_fprs = np.logspace(-5, 0, 100)
  rand_curve = []
  for f in ref_fprs:
    valid = np.where(fprs <= f)[0]
    if len(valid) == 0:
      rand_curve.append(0.0)
      continue
    tau = thresholds[valid[-1]]
    vuln_count = (members["lira_score"] >= tau).sum()
    rand_curve.append(float(vuln_count) / len(members))

  plt.plot(
      ref_fprs,
      rand_curve,
      color="gray",
      linestyle=":",
      linewidth=1.5,
      label="Random Guess",
  )
  plt.xscale("log")
  plt.ylim((0.0, 1.02))
  plt.xlim((1e-5, 1.0))
  plt.xlabel("FPR", fontsize=12)
  plt.ylabel("Precision@k=1%", fontsize=12)
  plt.title(
      f"Precision@k=1% vs. FPR ({args.dataset}, {args.arch},"
      f" {args.n_shadows} shadows)",
      fontsize=13,
  )
  plt.legend(loc="lower right", fontsize=10)
  plt.grid(True, which="both", linestyle="--", alpha=0.5)

  fig1_path = os.path.join(plots_dir, "precision_vs_fpr_comparison.png")
  plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
  plt.close()
  print(f"Saved: {fig1_path}")

  # Late-CSG Window Ablation
  plt.figure(figsize=(8, 6))
  ablation_metrics = [
      ("csg", "#D55E00", "CSG (100% Epochs)"),
      ("late_csg_40", "#0072B2", "Late-CSG (Last 40%)"),
      ("late_csg_30", "#009E73", "Late-CSG (Last 30%)"),
      ("late_csg_10", "#CC79A7", "Late-CSG (Last 10%)"),
  ]

  for key, c, lbl in ablation_metrics:
    col_name, asc = eval_registry[key]
    fpr_x, prec_y = compute_precision_curve(
        df, col_name, ascending=asc, k_frac=0.01
    )
    plt.plot(fpr_x, prec_y, color=c, linewidth=2.0, label=lbl)

  plt.xscale("log")
  plt.ylim((0.0, 1.02))
  plt.xlim((1e-5, 1.0))
  plt.xlabel("FPR", fontsize=12)
  plt.ylabel("Precision@k=1%", fontsize=12)
  plt.title("Late-CSG Ablation: Epoch Tracking Windows", fontsize=13)
  plt.legend(loc="lower right", fontsize=10)
  plt.grid(True, which="both", linestyle="--", alpha=0.5)

  fig2_path = os.path.join(plots_dir, "late_csg_ablation.png")
  plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
  plt.close()
  print(f"Saved: {fig2_path}")


if __name__ == "__main__":
  main()
