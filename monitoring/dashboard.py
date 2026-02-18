import matplotlib.pyplot as plt
import os

def generate_monitoring_plots(drift_report, recent_scores, output_dir="v4_artifacts/monitoring"):
    os.makedirs(output_dir, exist_ok=True)
    f = list(drift_report["feature_drifts"].keys())
    p = [v["psi"] for v in drift_report["feature_drifts"].values()]
    plt.figure(figsize=(10, 6))
    plt.bar(f, p, color='skyblue')
    plt.axhline(0.25, color='red', linestyle='--')
    plt.title("PSI Drift")
    plt.savefig(os.path.join(output_dir, "psi_dashboard.png"))
    plt.close()
