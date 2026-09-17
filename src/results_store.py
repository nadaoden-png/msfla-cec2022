import json
import os

DB_PATH = "/home/claude/msfla_project/results_db.json"


def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {}


def save_db(db):
    tmp_path = DB_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp_path, DB_PATH)


def count_runs(db, dim, func_num, algo_name):
    d = db.get(str(dim), {}).get(str(func_num), {}).get(algo_name, [])
    return len(d)


def add_run(db, dim, func_num, algo_name, seed, best_f, n_evals, history):
    db.setdefault(str(dim), {})
    db[str(dim)].setdefault(str(func_num), {})
    db[str(dim)][str(func_num)].setdefault(algo_name, [])
    db[str(dim)][str(func_num)][algo_name].append({
        "seed": seed, "best_f": best_f, "n_evals": n_evals, "history": history
    })


def summary(db, target_runs=30):
    lines = []
    total_done, total_needed = 0, 0
    for dim in sorted(db.keys(), key=int):
        for func_num in sorted(db[dim].keys(), key=int):
            for algo in sorted(db[dim][func_num].keys()):
                n = len(db[dim][func_num][algo])
                total_done += n
                total_needed += target_runs
                if n < target_runs:
                    lines.append(f"  D={dim:>3s} F{func_num:>2s} {algo:8s}: {n}/{target_runs}")
    return lines, total_done, total_needed
