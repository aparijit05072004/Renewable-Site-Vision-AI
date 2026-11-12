"""Run the full TerraGuard-v2 training & validation roadmap (Phases 1-5)."""
from __future__ import annotations

from . import (run_validation_suite, train_layer0_segmentation,
               train_layer1_sda, train_layer2_corrdiff, train_layer2_pinn,
               train_layer3_gnn)

if __name__ == "__main__":
    print("=" * 70, "\nPHASE 1 — Layer 0 Land Image Vision Engine\n", "=" * 70)
    train_layer0_segmentation.main()
    print("=" * 70, "\nPHASE 2 — Layer 1 Score-Based Data Assimilation\n", "=" * 70)
    train_layer1_sda.main()
    print("=" * 70, "\nPHASE 3 — Layer 2 CorrDiff + ABL-PINN\n", "=" * 70)
    train_layer2_corrdiff.main()
    train_layer2_pinn.main()
    print("=" * 70, "\nPHASE 4 — Layer 3 Fuzzy Spatial GNN\n", "=" * 70)
    train_layer3_gnn.main()
    print("=" * 70, "\nPHASE 5 — End-to-End & Persistence Gate\n", "=" * 70)
    run_validation_suite.main()
