import json, os, logging
def generate_retraining_report(vid, dec, comp, reas, out="retraining/sim_retraining_report.json"):
    rep = {"vid": vid, "decision": dec, "reason": reas, "comparison": comp}
    json.dump(rep, open(out, "w"), indent=4)
    logging.info(f"Report: {out}")
    return rep
