"""Métricas descriptivas para etiquetas expertas y predicciones visuales."""
from collections import Counter

LABELS = ("bueno", "regular", "malo", "no_visible")


def calculate_metrics(records):
    """Incluye rechazos no visibles; excluye únicamente errores de ejecución."""
    groups = {}
    for record in records:
        if record.get("error"):
            continue
        key = f"{record['asset']}/{record['criterio']}"
        groups.setdefault(key, []).append((record["referencia"], record["prediccion"]))
    results = {}
    for key, pairs in groups.items():
        matrix = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
        for truth, pred in pairs:
            matrix[truth][pred] += 1
        scores = {}
        for label in LABELS:
            tp = matrix[label][label]
            predicted = sum(matrix[t][label] for t in LABELS)
            actual = sum(matrix[label].values())
            precision = tp / predicted if predicted else None
            recall = tp / actual if actual else None
            f1 = (2 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None and precision + recall else
                  0.0 if precision is not None and recall is not None else None)
            scores[label] = {"precision": precision, "recall": recall, "f1": f1, "soporte": actual}
        n = len(pairs)
        observed = sum(matrix[x][x] for x in LABELS) / n
        truths, predictions = Counter(t for t, _ in pairs), Counter(p for _, p in pairs)
        expected = sum(truths[x] * predictions[x] for x in LABELS) / (n * n)
        kappa = (observed - expected) / (1 - expected) if expected != 1 else None
        results[key] = {"n": n, "matriz": matrix, "clases": scores, "acuerdo": observed, "kappa": kappa}
    return results
