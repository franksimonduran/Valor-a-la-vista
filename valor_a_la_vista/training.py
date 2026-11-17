"""Métricas de detección a umbral fijo, sin confundirlas con mAP o confianza clínica."""
from __future__ import annotations

import math


def box_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    a = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
    b = max(0, second[2]-second[0]) * max(0, second[3]-second[1])
    union = a + b - intersection
    return intersection / union if union else 0.0


def matching_counts(predictions, targets, *, confidence=.35, iou=.5):
    """Emparejamiento uno a uno por confianza; pred=[x1,y1,x2,y2,score]. Clase única."""
    if not 0 < confidence <= 1 or not 0 < iou <= 1:
        raise ValueError("Umbral de evaluación inválido.")
    if any(len(row) != 5 or not all(math.isfinite(v) for v in row) for row in predictions):
        raise ValueError("Predicciones inválidas.")
    used = set()
    tp, fp = 0, 0
    for row in sorted((p for p in predictions if p[4] >= confidence), key=lambda p: p[4], reverse=True):
        best, best_iou = None, 0
        for index, target in enumerate(targets):
            if index not in used:
                overlap = box_iou(row[:4], target)
                if overlap > best_iou:
                    best, best_iou = index, overlap
        if best is not None and best_iou >= iou:
            used.add(best)
            tp += 1
        else:
            fp += 1
    return {"tp": tp, "fp": fp, "fn": len(targets) - len(used)}


def count_metrics(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    return {**counts, "precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}
