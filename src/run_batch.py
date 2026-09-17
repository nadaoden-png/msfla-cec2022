import argparse
import time
import numpy as np

from cec2022_bridge import get_cec2022_function
import algorithms as alg
import results_store as store

TARGET_RUNS = 30

ALGO_RUNNERS = {
    "SFLA":  lambda func, dim, bounds, max_fe, seed: alg.run_sfla(
        "F", func, dim, bounds, F=400, m=20, n=20, N=15, max_fe=max_fe, seed=seed),
    "MSFLA": lambda func, dim, bounds, max_fe, seed: alg.run_msfla(
        "F", func, dim, bounds, F=400, m=20, n=20, N=15, max_fe=max_fe,
        alpha=1.0, theta_deg=30.0, sigma=0.7, mu=0.5, seed=seed),
    "PSO":   lambda func, dim, bounds, max_fe, seed: alg.run_pso(
        "F", func, dim, bounds, max_fe=max_fe, seed=seed),
    "GA":    lambda func, dim, bounds, max_fe, seed: alg.run_ga(
        "F", func, dim, bounds, max_fe=max_fe, seed=seed),
    "DE":    lambda func, dim, bounds, max_fe, seed: alg.run_de(
        "F", func, dim, bounds, max_fe=max_fe, seed=seed),
}

MAX_FE_BY_DIM = {10: 200_000, 20: 1_000_000}


def downsample_history(history, n_points=40):
    if len(history) <= n_points:
        return [[i, float(v)] for i, v in enumerate(history)]
    idx = np.linspace(0, len(history) - 1, n_points).astype(int)
    return [[int(i), float(history[i])] for i in idx]


def run_one(dim, func_num, algo_name, seed):
    func, bounds, f_star = get_cec2022_function(func_num, dim)
    max_fe = MAX_FE_BY_DIM[dim]
    runner = ALGO_RUNNERS[algo_name]
    r = runner(func, dim, bounds, max_fe, seed)
    hist = downsample_history(r.history)
    return r.best_f, r.n_evals, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True, choices=[10, 20])
    ap.add_argument("--funcs", type=str, required=True)
    ap.add_argument("--algos", type=str, required=True)
    ap.add_argument("--time_budget", type=float, default=260.0)
    args = ap.parse_args()

    if "-" in args.funcs:
        lo, hi = map(int, args.funcs.split("-"))
        func_list = list(range(lo, hi + 1))
    else:
        func_list = [int(x) for x in args.funcs.split(",")]
    algo_list = args.algos.split(",")

    db = store.load_db()
    t_start = time.time()
    n_done_this_session = 0

    print(f"เริ่มรัน: D={args.dim}, ฟังก์ชัน={func_list}, อัลกอริทึม={algo_list}, "
          f"เวลาที่ให้={args.time_budget}s\n")

    round_times = []  # เก็บเวลาที่ใช้จริงของแต่ละรอบ เพื่อประมาณเวลารอบถัดไป

    while True:
        elapsed = time.time() - t_start
        if elapsed >= args.time_budget:
            break

        # ประมาณเวลาที่รอบถัดไปน่าจะใช้ จากค่าเฉลี่ยของรอบที่ผ่านมา (ถ้ามี)
        # ถ้าเวลาที่เหลือไม่พอสำหรับอีก 1 รอบ (+ กันชน 15%) ให้หยุดทันที
        # แทนที่จะเริ่มรอบใหม่แล้วโดนตัดกลางคันเสียเวลาฟรี
        if round_times:
            est_next_round = max(round_times[-3:]) * 1.08
            if elapsed + est_next_round > args.time_budget:
                print(f"  [หยุดล่วงหน้า] เวลาที่เหลือ {args.time_budget - elapsed:.0f}s "
                      f"ไม่พอสำหรับอีก 1 รอบ (ประมาณ {est_next_round:.0f}s) "
                      f"-- หยุดเพื่อไม่ให้เสียเวลาฟรี")
                break

        job = None
        for func_num in func_list:
            for algo_name in algo_list:
                n = store.count_runs(db, args.dim, func_num, algo_name)
                if n < TARGET_RUNS:
                    job = (func_num, algo_name, n)
                    break
            if job:
                break

        if job is None:
            print("🎉 ครบทุกงานในขอบเขตที่ระบุแล้ว")
            break

        func_num, algo_name, n_existing = job
        seed = abs(10000 * algo_name.__hash__() % 100000 + n_existing) % (2**31)

        t0 = time.time()
        best_f, n_evals, hist = run_one(args.dim, func_num, algo_name, seed)
        t1 = time.time()
        round_times.append(t1 - t0)

        store.add_run(db, args.dim, func_num, algo_name, seed, best_f, n_evals, hist)
        store.save_db(db)

        n_done_this_session += 1
        print(f"  [{n_done_this_session}] D={args.dim} F{func_num:<2d} {algo_name:6s} "
              f"รอบที่ {n_existing+1}/{TARGET_RUNS}  best={best_f:.4e}  "
              f"evals={n_evals}  ({t1-t0:.1f}s)  "
              f"[เวลารวมที่ใช้ไป {time.time()-t_start:.0f}/{args.time_budget:.0f}s]")

    print(f"\nหยุดแล้ว -- ทำไปทั้งหมด {n_done_this_session} รอบในรอบนี้ "
          f"(เวลาที่ใช้จริง {time.time()-t_start:.0f}s)")

    lines, done, needed = store.summary(db, TARGET_RUNS)
    print(f"\nสถานะรวมล่าสุด (ทุกมิติ/ฟังก์ชัน/อัลกอริทึมที่เคยรัน): {done}/{needed} รอบ")
    if lines:
        print("งานที่ยังไม่ครบ (10 รายการแรก):")
        for line in lines[:10]:
            print(line)


if __name__ == "__main__":
    main()
