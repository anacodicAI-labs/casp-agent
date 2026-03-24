"""
One-command pipeline to generate all data needed for the MDPI paper.

Run from the code/ directory:
  python run_all_paper_data.py

This will:
  1. Run evaluate_models.py → model_evaluation_results.json (Module 01, 02, 07 metrics + 5-fold CV)
  2. Run generate_figure_data.py → outputs/figdata/*.csv (figures + tables)

Optional: run_casestudy_output.py → casestudy_output.json (Mumbai–Delhi insulin run)
  python run_casestudy_output.py

Outputs:
  - code/model_evaluation_results.json  (R², MAE, F1, confusion matrix, clusters, CV)
  - outputs/figdata/*.csv    (feature importance, forecast failure, LLM×country, Pareto, clusters, CASP, early warning, confusion matrix, delay by package type)
"""

import os
import sys
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

def main():
    print("=" * 60)
    print("PAPER DATA PIPELINE")
    print("=" * 60)

    # Step 1: Full model evaluation (Module 01, 02, 07) → model_evaluation_results.json
    print("\n[1/2] Running evaluate_models.py ...")
    r1 = subprocess.run([sys.executable, "evaluate_models.py"], cwd=script_dir)
    if r1.returncode != 0:
        print("evaluate_models.py failed; continuing with figure data (confusion_matrix may recompute from EWS).")
    else:
        print("  → model_evaluation_results.json updated.")

    # Step 2: Figure/table CSVs → figdata/
    print("\n[2/2] Running generate_figure_data.py ...")
    r2 = subprocess.run([sys.executable, "generate_figure_data.py"], cwd=script_dir)
    if r2.returncode != 0:
        print("generate_figure_data.py failed.")
        sys.exit(1)
    print("  → figdata/*.csv updated.")

    print("\n" + "=" * 60)
    print("Done. For case study JSON run: python run_casestudy_output.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
