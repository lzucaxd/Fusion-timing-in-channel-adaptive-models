#!/usr/bin/env python3
"""
Compute CHAMMI Performance Score (CPS) from per-task macro-F1 scores.

CPS is defined as a weighted average across the six generalization tasks:
- Allen Task_one (SD)
- Allen Task_two (OOD, known classes)
- HPA Task_one (SD)
- HPA Task_two (OOD, known classes)
- HPA Task_three (OOD, novel classes)
- CP Task_one (SD)

Note: This is an approximation. The exact CHAMMI CPS formula may differ.
We use equal weights for now, but CHAMMI benchmark may use different weights.
"""

def compute_cps(allen_task_one_f1, allen_task_two_f1,
                hpa_task_one_f1, hpa_task_two_f1, hpa_task_three_f1,
                cp_task_one_f1):
    """
    Compute CHAMMI Performance Score (CPS).
    
    CHAMMI CPS is typically a weighted average of macro-F1 across six tasks.
    For now, we use equal weights. Adjust if CHAMMI paper specifies different weights.
    """
    tasks = [
        allen_task_one_f1,
        allen_task_two_f1,
        hpa_task_one_f1,
        hpa_task_two_f1,
        hpa_task_three_f1,
        cp_task_one_f1,
    ]
    
    # Equal weights (CHAMMI may use different weights - verify with paper)
    cps = sum(tasks) / len(tasks)
    return cps

if __name__ == "__main__":
    # HierBoCSetViT-Small results (from COMPREHENSIVE_RESULTS_SUMMARY.md)
    allen_task_one_f1 = 0.6334
    allen_task_two_f1 = 0.5202
    hpa_task_one_f1 = 0.9388
    hpa_task_two_f1 = 0.8398
    hpa_task_three_f1 = 0.2444
    cp_task_one_f1 = 0.9041
    
    cps = compute_cps(
        allen_task_one_f1, allen_task_two_f1,
        hpa_task_one_f1, hpa_task_two_f1, hpa_task_three_f1,
        cp_task_one_f1
    )
    
    print(f"CPS = {cps:.4f}")
    print(f"\nPer-task macro-F1:")
    print(f"  Allen Task_one: {allen_task_one_f1:.4f}")
    print(f"  Allen Task_two: {allen_task_two_f1:.4f}")
    print(f"  HPA Task_one: {hpa_task_one_f1:.4f}")
    print(f"  HPA Task_two: {hpa_task_two_f1:.4f}")
    print(f"  HPA Task_three: {hpa_task_three_f1:.4f}")
    print(f"  CP Task_one: {cp_task_one_f1:.4f}")

