"""Shared validation/test evaluation with explicit metric and threshold policies."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from utils.data import VesselDataset


def collect(model, items, device, tta=False, normalize=True):
    model.eval()
    result = []
    with torch.inference_mode():
        for image, mask, fov, key in VesselDataset(items):
            x = image[None].to(device)
            views = [()] if not tta else [(), (3,), (2,), (2, 3)]
            prob = 0
            for dims in views:
                view = torch.flip(x, dims) if dims else x
                p = model(view)[-1].float().sigmoid()
                prob = prob + (torch.flip(p, dims) if dims else p)
            h, w = mask.shape
            p = (prob / len(views))[0, 0, :h, :w].cpu().numpy()
            if normalize:
                p = (p-p.min()) / (p.max()-p.min()+1e-8)
            result.append((key, p, mask.astype(bool), fov))
    return result


def metrics(prob, gt, fov, threshold=.5, region='legacy'):
    pred = (prob >= threshold) & fov
    target = gt & fov
    valid = fov if region == 'fov' else np.ones_like(fov, dtype=bool)
    p, g = pred[valid], target[valid]
    tp, tn = np.count_nonzero(p & g), np.count_nonzero(~p & ~g)
    fp, fn = np.count_nonzero(p & ~g), np.count_nonzero(~p & g)
    eps = 1e-8
    # Historical code: AUC inside FOV, other metrics on zero-masked full image.
    auc = float(roc_auc_score(target[fov], prob[fov])) if np.unique(target[fov]).size == 2 else float('nan')
    return {'F1': (2*tp+1e-6)/(2*tp+fp+fn+1e-6), 'IoU': (tp+1e-6)/(tp+fp+fn+1e-6),
            'Sensitivity': tp/(tp+fn+eps), 'Specificity': tn/(tn+fp+eps),
            'Precision': tp/(tp+fp+eps), 'ACC': (tp+tn)/(tp+tn+fp+fn+eps), 'AUC': auc}


def summarize(predictions, threshold=.5, region='legacy'):
    rows = [dict(ID=key, **metrics(p, g, f, threshold, region)) for key,p,g,f in predictions]
    if not rows:
        raise ValueError('No evaluation images')
    return rows, {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'ID'}


def select_threshold(validation, region='legacy'):
    candidates = np.arange(.30, .82, .02)
    return float(max(candidates, key=lambda t: summarize(validation, float(t), region)[1]['F1']))
