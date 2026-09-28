"""Compare angle, naive Cartesian, and exponential-map integration."""
import json
import time
from pathlib import Path

import numpy as np

from zcartpole import ZNCartPole

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "e1_integration.json"

MODELS = {
    2: dict(m=(0.1, 0.1), l=(0.5, 0.5)),
    3: dict(m=(0.1, 0.1, 0.1), l=(0.4, 0.4, 0.4)),
    4: dict(m=(0.1, 0.1, 0.1, 0.1), l=(0.35, 0.35, 0.35, 0.35)),
    5: dict(m=(0.1,) * 5, l=(0.3,) * 5),
}
T_FREE = 2.0
DT_GRID = [1e-2, 3e-3, 1e-3, 3e-4]
DT_REPEAT = 200  



T_ORDER = 1.0
DT_REF = 1e-4
DT_ORDER_GRID = [8e-3, 4e-3, 2e-3]


def naive_step(model, s, u, h):
    """Run naive step."""
    n = model.n

    def f(y):
        x, v = y[0], y[1]
        c = y[2:2 + 3 * n:3]; sn = y[3:2 + 3 * n:3]; w = y[4:2 + 3 * n:3]
        a = model.accel(c + 1j * sn, w, u)
        d = np.empty_like(y)
        d[0], d[1] = v, a[0]
        d[2:2 + 3 * n:3] = -w * sn
        d[3:2 + 3 * n:3] = w * c
        d[4:2 + 3 * n:3] = a[1:]
        return d

    k1 = f(s); k2 = f(s + 0.5 * h * k1); k3 = f(s + 0.5 * h * k2); k4 = f(s + h * k3)
    return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def theta_step(model, y, u, h):
    """Run theta step."""
    n = model.n

    def f(yy):
        x, v = yy[0], yy[1]
        th = yy[2:2 + 2 * n:2]; w = yy[3:2 + 2 * n:2]
        a = model.accel(np.exp(1j * th), w, u)
        d = np.empty_like(yy)
        d[0], d[1] = v, a[0]
        d[2:2 + 2 * n:2] = w
        d[3:2 + 2 * n:2] = a[1:]
        return d

    k1 = f(y); k2 = f(y + 0.5 * h * k1); k3 = f(y + 0.5 * h * k2); k4 = f(y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def pack_naive(model, x, v, th, w):
    n = model.n
    s = np.empty(2 + 3 * n)
    s[0], s[1] = x, v
    s[2:2 + 3 * n:3] = np.cos(th); s[3:2 + 3 * n:3] = np.sin(th); s[4:2 + 3 * n:3] = w
    return s


def pack_theta(x, v, th, w):
    n = len(th)
    y = np.empty(2 + 2 * n)
    y[0], y[1] = x, v
    y[2:2 + 2 * n:2] = th; y[3:2 + 2 * n:2] = w
    return y


def run_free(model, th0, w0, dt, method, t_total=None):
    """Run run free."""
    n = model.n
    steps = int(round((t_total if t_total is not None else T_FREE) / dt))
    E0 = None
    max_zdev = 0.0
    max_dE = 0.0
    max_dP = 0.0

    if method == "expmap":
        s = model.pack(0.0, 0.0, np.exp(1j * th0), w0)
        E0 = model.total_energy(s); P0 = model.momentum(s)
        for _ in range(steps):
            s = model.step(s, 0.0, dt)
            _, _, zs, _ = model.unpack(s)
            max_zdev = max(max_zdev, float(np.abs(np.abs(zs) - 1).max()))
            max_dE = max(max_dE, abs(model.total_energy(s) - E0))
            max_dP = max(max_dP, abs(model.momentum(s) - P0))
        _, _, zs, ws = model.unpack(s)
        th_final = np.angle(zs)

    elif method == "naive":
        s = pack_naive(model, 0.0, 0.0, th0, w0)
        zs0 = np.exp(1j * th0)
        s_std = model.pack(0.0, 0.0, zs0, w0)
        E0 = model.total_energy(s_std); P0 = model.momentum(s_std)
        for _ in range(steps):
            s = naive_step(model, s, 0.0, dt)
            c = s[2:2 + 3 * n:3]; sn = s[3:2 + 3 * n:3]; w = s[4:2 + 3 * n:3]
            zs = c + 1j * sn
            max_zdev = max(max_zdev, float(np.abs(np.abs(zs) - 1).max()))
            s_std = model.pack(s[0], s[1], zs, w)
            max_dE = max(max_dE, abs(model.total_energy(s_std) - E0))
            max_dP = max(max_dP, abs(model.momentum(s_std) - P0))
        th_final = np.arctan2(sn, c)

    elif method == "theta":
        y = pack_theta(0.0, 0.0, th0, w0)
        s_std = model.pack(0.0, 0.0, np.exp(1j * th0), w0)
        E0 = model.total_energy(s_std); P0 = model.momentum(s_std)
        for _ in range(steps):
            y = theta_step(model, y, 0.0, dt)
            th = y[2:2 + 2 * n:2]; w = y[3:2 + 2 * n:2]
            s_std = model.pack(y[0], y[1], np.exp(1j * th), w)
            max_dE = max(max_dE, abs(model.total_energy(s_std) - E0))
            max_dP = max(max_dP, abs(model.momentum(s_std) - P0))
        th_final = y[2:2 + 2 * n:2]
        max_zdev = 0.0  

    else:
        raise ValueError(method)

    return dict(max_zdev=max_zdev, max_dE=max_dE, max_dP=max_dP, theta_final=th_final.tolist())


def agreement_from_cache(cache):
    """Run agreement from cache."""
    ref = np.array(cache["theta"]["theta_final"])
    zref = np.exp(1j * ref)
    out = {}
    for method in ("expmap", "naive"):
        got = np.array(cache[method]["theta_final"])
        z = np.exp(1j * got)
        phi = np.angle(z * np.conj(zref))
        out[method] = float(np.abs(phi).max())
    return out


def order_of_convergence(model, th0, w0):
    """Run order of convergence."""
    ref = run_free(model, th0, w0, DT_REF, "expmap", t_total=T_ORDER)["theta_final"]
    errs = []
    for dt in DT_ORDER_GRID:
        got = run_free(model, th0, w0, dt, "expmap", t_total=T_ORDER)["theta_final"]
        errs.append(float(np.abs(np.array(got) - np.array(ref)).max()))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1) if errs[i + 1] > 0]
    return dict(dts=DT_ORDER_GRID, dt_ref=DT_REF, t_total=T_ORDER, errs=errs, halving_ratios=ratios)


def timing(model, th0, w0, dt):
    n = model.n
    s0 = model.pack(0.0, 0.0, np.exp(1j * th0), w0)
    s0n = pack_naive(model, 0.0, 0.0, th0, w0)
    y0 = pack_theta(0.0, 0.0, th0, w0)

    t0 = time.perf_counter()
    s = s0
    for _ in range(DT_REPEAT):
        s = model.step(s, 0.0, dt)
    t_exp = (time.perf_counter() - t0) / DT_REPEAT

    t0 = time.perf_counter()
    s = s0n
    for _ in range(DT_REPEAT):
        s = naive_step(model, s, 0.0, dt)
    t_naive = (time.perf_counter() - t0) / DT_REPEAT

    t0 = time.perf_counter()
    y = y0
    for _ in range(DT_REPEAT):
        y = theta_step(model, y, 0.0, dt)
    t_theta = (time.perf_counter() - t0) / DT_REPEAT

    return dict(expmap_s=t_exp, naive_s=t_naive, theta_s=t_theta)


def main():
    rng = np.random.default_rng(0)
    results = {"dt_grid": DT_GRID, "t_free_s": T_FREE, "per_n": {}}

    for n, params in MODELS.items():
        model = ZNCartPole(m=params["m"], l=params["l"], u_max=50.0)
        th0 = rng.uniform(-2, 2, n)
        w0 = rng.uniform(-2, 2, n)

        per_dt = {}
        for dt in DT_GRID:
            row = {}
            for method in ("expmap", "naive", "theta"):
                row[method] = run_free(model, th0, w0, dt, method)
            row["agreement_vs_theta"] = agreement_from_cache(row)
            per_dt[f"{dt:g}"] = row

        results["per_n"][str(n)] = dict(
            initial_angles_rad=th0.tolist(),
            initial_speeds_rad_s=w0.tolist(),
            per_dt=per_dt,
            order_check=order_of_convergence(model, th0, w0),
            timing_at_1em3=timing(model, th0, w0, 1e-3),
        )
        print(f"n={n} tamam")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote:", OUT)


if __name__ == "__main__":
    main()
