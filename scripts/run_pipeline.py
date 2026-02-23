from uber_stock.pipeline import run_full_analysis


if __name__ == "__main__":
    df, metrics, model_summary = run_full_analysis()
    print("Pipeline completed successfully ✅")
    print("\nRisk Metrics:")
    for k, v in metrics["risk_metrics"].items():
        print(f"  {k}: {v:.6f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

    print("\nModel Metrics:")
    for k, v in metrics["model_metrics"].items():
        print(f"  {k}: {v:.6f}")