"""Phase 1 — Fine-tune the SegFormer-style land segmentation network.

Gate (blueprint Table 4): mIoU >= 0.84 on Barren(5), Scrub(2), Water(4),
Forest(3) held-out tiles, and < 1.5 s inference per 1024x1024 land image.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import CHECKPOINT_DIR, METRICS_DIR, SEED
from ..layer0_vision.segmentation_model import SegFormerNano, mean_iou
from ..layer0_vision.synthetic_tiles import generate_batch

GATE_CLASSES = (5, 2, 4, 3)   # Barren, Scrub, Water, Forest


def class_weights() -> torch.Tensor:
    # roads/urban are thin structures — upweight
    return torch.tensor([2.0, 1.0, 1.0, 1.2, 1.6, 1.2, 3.0])


def main(steps: int = 240, batch: int = 6, tile: int = 192,
         lr: float = 6e-4, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    model = SegFormerNano()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    w = class_weights()

    t_start = time.time()
    model.train()
    for step in range(steps):
        imgs, masks = generate_batch(batch, tile, seed=seed + step)
        x = torch.from_numpy(imgs)
        y = torch.from_numpy(masks)
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=w)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 20 == 0 or step == steps - 1:
            print(f"[L0] step {step:4d}/{steps}  ce={loss.item():.4f}")

    # ---- held-out evaluation ----
    model.eval()
    ious_all, per_class_acc = [], {}
    with torch.no_grad():
        for k in range(8):
            imgs, masks = generate_batch(4, tile, seed=909_000 + k)
            pred = model(torch.from_numpy(imgs)).argmax(1)
            m = mean_iou(pred, torch.from_numpy(masks), classes=GATE_CLASSES)
            ious_all.append(m["miou"])
            for c, v in m["per_class"].items():
                per_class_acc.setdefault(c, []).append(v)
    gate_miou = float(np.mean(ious_all))

    # ---- latency on a 1024x1024 image (inference path resizes to 256) ----
    from ..layer0_vision.land_image_analyzer import LandImagePerceptionEngine
    big = (np.random.default_rng(0).random((1024, 1024, 3)) * 255).astype(np.uint8)
    torch.save({"model": model.state_dict()},
               CHECKPOINT_DIR / "layer0_segformer_nano.pt")
    engine = LandImagePerceptionEngine()
    t0 = time.perf_counter(); engine.segment(big)
    latency = time.perf_counter() - t0

    metrics = {
        "gateMiouBarrenScrubWaterForest": round(gate_miou, 4),
        "perClassIoU": {str(c): round(float(np.mean(v)), 4)
                        for c, v in sorted(per_class_acc.items())},
        "latency1024S": round(latency, 3),
        "gate": {"miou>=0.84": gate_miou >= 0.84,
                 "latency<1.5s": latency < 1.5},
        "trainSteps": steps, "trainMinutes": round((time.time() - t_start) / 60, 2),
        "params": sum(p.numel() for p in model.parameters()),
    }
    (METRICS_DIR / "layer0_segmentation.json").write_text(
        json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    main(steps=ap.parse_args().steps)
