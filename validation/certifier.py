import logging, json, os, asyncio, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation.adversarial_tester import AdversarialTester
from validation.load_tester import LoadTester
from validation.stability_tester import StabilityTester, RealWorldBenchmarker
from monitoring.model_registry import ModelRegistry

async def run_cert():
    logging.info("🏆 Certifying...")
    reg = ModelRegistry()
    m, meta = reg.load_production_model()
    if not m: return
    
    adv = AdversarialTester(m, meta).run_tests()
    load = await LoadTester(m, meta).simulate_load()
    bg = RealWorldBenchmarker(m, meta).run_benchmark()
    
    kpis = {"AUC": {"val": bg["roc_auc"], "pass": bg["roc_auc"] >= 0.90}, "P95": {"val": load["p95_latency_ms"], "pass": load["p95_latency_ms"] < 800}}
    status = "PASS" if all(v["pass"] for v in kpis.values()) else "CONDITIONAL"
    
    rep = {"status": status, "kpis": kpis, "debug": {"adv": adv, "bg": bg, "load": load}}
    json.dump(rep, open("validation/enterprise_certification.json", "w"), indent=4)
    logging.info(f"✅ Status: {status}")

if __name__ == "__main__": asyncio.run(run_cert())
