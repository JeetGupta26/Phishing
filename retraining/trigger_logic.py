import json, os, logging

def check_retraining_trigger(report_path="monitoring/daily_report_v4.json"):
    if not os.path.exists(report_path): return False, "No report"
    try:
        report = json.load(open(report_path, "r"))
        if report.get("drift", {}).get("alert_triggered", False): return True, "Critical drift"
        if report.get("pred", {}).get("anomaly_detected", False): return True, "Anomaly"
        return False, "Stable"
    except Exception: return False, "Error"
