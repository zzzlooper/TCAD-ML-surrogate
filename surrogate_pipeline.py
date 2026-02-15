#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

# Avoid matplotlib cache write issues in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MIN_TRAIN_ROWS_FOR_MLP_EARLY_STOPPING = 11


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate TCAD surrogate regressors."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="File or directory of csv/parquet simulation outputs.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        required=True,
        help="Name of target/output metric column.",
    )
    parser.add_argument(
        "--sim-time-col",
        type=str,
        default="simulation_time_s",
        help="Column containing simulation runtime in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for metrics and plots.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split fraction."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic data if real data is unavailable.",
    )
    parser.add_argument(
        "--assumed-sim-time-s",
        type=float,
        default=None,
        help="Fallback simulation time per run when sim-time column is missing.",
    )
    return parser.parse_args()


def generate_demo_data(n_samples=3000):
    rng = np.random.default_rng(RANDOM_STATE)
    voltage = rng.uniform(0.6, 1.2, n_samples)
    temperature = rng.uniform(250, 450, n_samples)
    gate_length_nm = rng.uniform(10, 60, n_samples)
    oxide_thickness_nm = rng.uniform(0.8, 3.0, n_samples)
    doping_cm3 = rng.uniform(5e17, 4e18, n_samples)

    nonlinear = (
        2.0 * voltage
        - 0.003 * (temperature - 300)
        + 0.09 * np.log10(doping_cm3)
        - 0.035 * gate_length_nm
        + 0.04 * np.sin(3 * voltage)
        - 0.06 * np.sqrt(oxide_thickness_nm)
    )
    interaction = 0.00012 * (temperature - 300) * voltage
    noise = rng.normal(0, 0.03, n_samples)
    drain_current = nonlinear + interaction + noise

    simulation_time_s = rng.uniform(20, 120, n_samples)

    return pd.DataFrame(
        {
            "voltage": voltage,
            "temperature": temperature,
            "gate_length_nm": gate_length_nm,
            "oxide_thickness_nm": oxide_thickness_nm,
            "doping_cm3": doping_cm3,
            "simulation_time_s": simulation_time_s,
            "drain_current": drain_current,
        }
    )


def load_runs(data_path):
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    if p.is_file():
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        raise ValueError("Unsupported file format. Use CSV or Parquet.")

    csv_files = sorted(p.rglob("*.csv"))
    parquet_files = sorted(p.rglob("*.parquet"))
    files = csv_files + parquet_files

    if not files:
        raise ValueError(f"No CSV/Parquet files found under directory: {data_path}")

    frames = []
    for file in files:
        if file.suffix.lower() == ".csv":
            frames.append(pd.read_csv(file))
        else:
            frames.append(pd.read_parquet(file))
    return pd.concat(frames, ignore_index=True)


def build_models(numeric_features, mlp_early_stopping=True):
    scaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    unscaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    models = {
        "linear_regression": Pipeline(
            steps=[("preprocess", scaled_preprocessor), ("model", LinearRegression())]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", unscaled_preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            steps=[
                ("preprocess", scaled_preprocessor),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 64),
                        activation="relu",
                        learning_rate_init=1e-3,
                        max_iter=700,
                        early_stopping=mlp_early_stopping,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    return models


def plot_predicted_vs_true(y_true, y_pred, model_name, path):
    plt.figure(figsize=(6.2, 6.2))
    plt.scatter(y_true, y_pred, s=14, alpha=0.65)
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    plt.plot([low, high], [low, high], "r--", linewidth=1.5)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs True ({model_name})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_error_distribution(errors, model_name, path):
    plt.figure(figsize=(6.6, 4.2))
    plt.hist(errors, bins=40, alpha=0.85, edgecolor="black", linewidth=0.4)
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title(f"Error Distribution ({model_name})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compute_breakdown_by_feature(X_test, abs_error, feature, n_bins=8):
    series = X_test[feature]
    bins = pd.qcut(series, q=n_bins, duplicates="drop")
    grouped = (
        pd.DataFrame({"bin": bins, "abs_error": abs_error})
        .dropna(subset=["bin"])
        .groupby("bin", as_index=False, observed=True)
        .agg(mae_in_bin=("abs_error", "mean"))
        .assign(bin=lambda df_: df_["bin"].astype(str), feature=feature)
    )
    return grouped


def plot_breakdown(grouped_df, feature, path):
    g = grouped_df[grouped_df["feature"] == feature].copy()
    plt.figure(figsize=(8.5, 4.0))
    plt.plot(range(len(g)), g["mae_in_bin"], marker="o")
    plt.xticks(range(len(g)), g["bin"], rotation=35, ha="right")
    plt.ylabel("MAE in Bin")
    plt.title(f"Error vs {feature} Range")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        df = generate_demo_data()
        if args.target_col not in df.columns:
            raise ValueError(
                f"In demo mode, target col must be one of: {list(df.columns)}"
            )
    else:
        if not args.data_path:
            raise ValueError("Provide --data-path or use --demo")
        df = load_runs(args.data_path)

    if args.target_col not in df.columns:
        raise ValueError(
            f"Target column '{args.target_col}' not found. Available columns: {list(df.columns)}"
        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {args.target_col, args.sim_time_col}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found after excluding target/runtime columns."
        )

    selected_cols = feature_cols + [args.target_col]
    if args.sim_time_col in df.columns:
        selected_cols.append(args.sim_time_col)

    clean_df = df[selected_cols].copy()
    for col in selected_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    target_values = clean_df.loc[:, str(args.target_col)]
    target_mask = pd.Series(pd.notna(target_values), index=clean_df.index, dtype=bool)
    clean_df = clean_df.loc[target_mask].reset_index(drop=True)

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1 (exclusive).")

    n_samples = len(clean_df)
    n_test = int(np.ceil(n_samples * args.test_size))
    n_train = n_samples - n_test
    if n_train < 1 or n_test < 1:
        raise ValueError(
            "Not enough usable rows after cleaning for the requested --test-size. "
            f"Rows={n_samples}, test_size={args.test_size}."
        )

    X = clean_df[feature_cols]
    y = clean_df[args.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )

    models = build_models(
        feature_cols,
        mlp_early_stopping=len(X_train) >= MIN_TRAIN_ROWS_FOR_MLP_EARLY_STOPPING,
    )

    metrics_rows = []
    runtime_rows = []
    importance_rows = []
    breakdown_rows = []

    for model_name, pipeline in models.items():
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        infer_total_s = time.perf_counter() - t1
        infer_per_sample_s = infer_total_s / max(len(X_test), 1)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan

        metrics_rows.append(
            {
                "model": model_name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "train_time_s": train_time_s,
                "inference_total_s_test": infer_total_s,
                "inference_per_sample_s": infer_per_sample_s,
            }
        )

        sim_time_used = None
        if args.sim_time_col in clean_df.columns:
            sim_time_values = clean_df.loc[:, args.sim_time_col].to_numpy(
                dtype=float, copy=False
            )
            finite_sim_time = sim_time_values[np.isfinite(sim_time_values)]
            if finite_sim_time.size > 0:
                sim_time_used = float(finite_sim_time.mean())
        elif args.assumed_sim_time_s is not None:
            sim_time_used = args.assumed_sim_time_s

        if sim_time_used is not None:
            speedup = sim_time_used / max(infer_per_sample_s, 1e-12)
            runtime_rows.append(
                {
                    "model": model_name,
                    "avg_simulation_time_s_per_sample": sim_time_used,
                    "ml_inference_time_s_per_sample": infer_per_sample_s,
                    "estimated_speedup_x": speedup,
                }
            )

        residuals = y_test - y_pred
        plot_predicted_vs_true(
            y_test,
            y_pred,
            model_name,
            plot_dir / f"pred_vs_true_{model_name}.png",
        )
        plot_error_distribution(
            residuals, model_name, plot_dir / f"error_dist_{model_name}.png"
        )

        perm = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=7,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        perm_importance_mean = np.asarray(perm["importances_mean"], dtype=float)
        importance_frame = pd.DataFrame(
            {"feature": feature_cols, "permutation_importance": perm_importance_mean}
        )
        importance_rows.extend(
            importance_frame.assign(model=model_name).to_dict(orient="records")
        )

        top_features = (
            importance_frame.sort_values("permutation_importance", ascending=False)
            .head(2)["feature"]
            .tolist()
        )
        abs_err = residuals.abs()
        for feat in top_features:
            g = compute_breakdown_by_feature(X_test, abs_err, feat, n_bins=8)
            g["model"] = model_name
            breakdown_rows.append(g)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("mae", ascending=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    if runtime_rows:
        runtime_df = pd.DataFrame(runtime_rows)
        runtime_df.to_csv(output_dir / "runtime_comparison.csv", index=False)
    else:
        runtime_df = None

    importance_df = pd.DataFrame(importance_rows).sort_values(
        ["model", "permutation_importance"], ascending=[True, False]
    )
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    if breakdown_rows:
        breakdown_df = pd.concat(breakdown_rows, ignore_index=True)
        breakdown_df.to_csv(output_dir / "breakdown_mae_by_feature.csv", index=False)
        model_names = [str(m) for m in np.unique(np.asarray(breakdown_df["model"]))]
        for model_name in model_names:
            model_slice = breakdown_df[breakdown_df["model"] == model_name]
            feature_names = [
                str(f) for f in np.unique(np.asarray(model_slice["feature"]))
            ]
            for feat in feature_names:
                plot_breakdown(
                    model_slice, feat, plot_dir / f"breakdown_{model_name}_{feat}.png"
                )

    best_model = metrics_df.iloc[0]["model"]

    summary_lines = []
    summary_lines.append("# Surrogate Modeling Summary")
    summary_lines.append("")
    summary_lines.append(f"Rows used: {len(clean_df)}")
    summary_lines.append(
        f"Features used ({len(feature_cols)}): {', '.join(feature_cols)}"
    )
    summary_lines.append("")
    summary_lines.append("## Model Ranking (by MAE)")
    summary_lines.append(metrics_df.to_csv(index=False))
    summary_lines.append("")
    summary_lines.append(f"Best model: **{best_model}**")

    if runtime_df is not None:
        best_runtime = runtime_df[runtime_df["model"] == best_model]
        if not best_runtime.empty:
            speedup = float(best_runtime.iloc[0]["estimated_speedup_x"])
            summary_lines.append(f"Estimated speedup (best model): **{speedup:.2f}x**")

    summary_lines.append("")
    summary_lines.append("## Where model may break down")
    summary_lines.append(
        "Use `breakdown_mae_by_feature.csv` and corresponding plots to identify parameter ranges with elevated MAE."
    )

    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Run completed.")
    print(f"Artifacts saved to: {output_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
