import csv
import json

from rag import query_engine

# "Positive" means the bot says YES, the item is on the menu.


def bot_says_yes(item):
    answer = str(query_engine.query(f"¿Tienen {item} en el menú? Responde solo 'sí' o 'no'."))
    return answer.strip().lower().startswith(("sí", "si")), answer.strip()


def label(predicted, actual):
    if predicted and actual:
        return "TP"  # said yes, and it IS on the menu
    if predicted and not actual:
        return "FP"  # said yes, but it is NOT on the menu (made it up)
    if not predicted and actual:
        return "FN"  # said no, but it IS on the menu (lost sale)
    return "TN"  # said no, and it is NOT on the menu


def run_batch(dataset_path):
    """Ask the bot about every item in the dataset and label each answer."""
    results = []
    for row in json.load(open(dataset_path)):
        predicted, answer = bot_says_yes(row["item"])
        results.append({
            "item": row["item"],
            "expected": "sí" if row["on_menu"] else "no",
            "bot_answer": answer,
            "result": label(predicted, row["on_menu"]),
        })
    return results


def score(results):
    """Turn a list of TP/FP/FN/TN labels into metrics. Works for any yes/no dataset."""
    counts = {key: sum(r["result"] == key for r in results) for key in ("TP", "FP", "FN", "TN")}
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {
        **counts,
        "accuracy": (tp + tn) / len(results),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
    }


def save_csv(results, path):
    """One row per item, so you can open it in Excel or Google Sheets."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    results = run_batch("menu_items.json")
    save_csv(results, "results.csv")

    for r in results:
        print(f"{r['result']}  {r['item']:<20} bot: {r['bot_answer']}")

    metrics = score(results)
    print("\nConfusion matrix")
    print("                  real: on menu   real: not on menu")
    print(f"bot said yes      TP = {metrics['TP']:<12} FP = {metrics['FP']}")
    print(f"bot said no       FN = {metrics['FN']:<12} TN = {metrics['TN']}")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"{name.capitalize():<10} {metrics[name]:.2f}")
    print("\nSaved results.csv")
