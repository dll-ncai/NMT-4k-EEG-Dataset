from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def predict_recordings(model, loader, records, device, amp: bool = True, desc: str = "Predict"):
    model.eval()
    prob_lists: dict[int, list[float]] = defaultdict(list)
    target_by_rec: dict[int, int] = {}

    use_amp = bool(amp and device.type == "cuda")
    for x, y, rec_idx in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        ys = y.cpu().numpy().astype(int)
        ids = rec_idx.cpu().numpy().astype(int)
        for p, t, rid in zip(probs, ys, ids):
            prob_lists[int(rid)].append(float(p))
            target_by_rec[int(rid)] = int(t)

    out_ids, out_y, out_p, out_n = [], [], [], []
    for rid in sorted(prob_lists):
        out_ids.append(rid)
        out_y.append(target_by_rec[rid])
        out_p.append(float(np.mean(prob_lists[rid])))
        out_n.append(len(prob_lists[rid]))

    result = records.iloc[out_ids].copy().reset_index(drop=True)
    result["target"] = out_y
    result["prob_abnormal"] = out_p
    result["n_windows"] = out_n
    return result
