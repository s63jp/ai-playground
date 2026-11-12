"""Testing networks against the physics engine.

Offline scores (RMSE on a held-out split) say how well a network fits recorded data. They do not
say whether it is useful. This module tests networks the only way that matters here: in the world.

  fresh_engine_test   new throws the network has never seen, run in MuJoCo, scored against the
                      engine's own measurements (catches overfitting to one dataset).
  closed_loop_height  the network is used as an inverse model: for each target peak height it
                      chooses the force (by searching its own predictions), the engine throws with
                      that force, and the measured peak is compared with the target. A linear
                      model and a fixed rule-of-thumb are run through exactly the same loop.
"""
from __future__ import annotations

import math
import random

from . import MLError
from .data import _direction, _Site, generate_throw_dataset
from .nets import Predictor, least_squares, regression_metrics


def fresh_engine_test(cfg: dict, model: dict, n: int = 200, seed: int = 10_007) -> dict:
    """Score a stored model on brand-new engine throws (a seed it was not trained on)."""
    pred = Predictor(model)
    ds = generate_throw_dataset(cfg, n, seed=seed)
    rows = [r for r in ds["rows"] if all(isinstance(r.get(c), (int, float)) for c in pred.spec["inputs"])]
    if not rows:
        raise MLError("not_enough_data", "The fresh engine test produced no usable throws.")
    y = [r[pred.spec["target"]] for r in rows]
    p = pred.predict_rows(rows)
    return {"seed": seed, "throws": len(rows), "metrics": regression_metrics(y, p),
            "dataset_sha256": ds["manifest"]["rows_sha256"]}


def _site_inputs(site: _Site, oid: str, force: float, elev: float, jump: bool) -> dict:
    """The inputs a network would see for a throw from the standard site, taken from a probe of the engine's
    pre-release state (so release height, thrower velocity etc. are real, not assumed)."""
    w = site.world
    w.set_state(site.prepare(oid))
    if jump:
        w.jump(wait="takeoff")
    v = w.agent_vel()
    d = _direction(elev, 0.0)
    return {"force": force, "dir_x": d[0], "dir_y": d[1], "dir_z": d[2], "elevation_deg": elev,
            "agent_vx": float(v[0]), "agent_vy": float(v[1]), "agent_vz": float(v[2]),
            "object_mass": float(w.specs[oid].mass), "feet_height": float(w.feet()),
            "on_ground": 1.0 if w.grounded else 0.0,
            "release_height": float(w.feet() + (2 * w.hh + w.bounding_radius(oid) + 0.05 if d[2] > 0.95 and
                                                math.hypot(d[0], d[1]) <= 0.3 else float(w.ac["hold_height"])))}


def closed_loop_height(cfg: dict, model: dict, targets: list[float] | None = None, elevation: float = 80.0,
                       jump: bool = False, oid: str = "block_01", tolerance: float = 0.25,
                       linear_rows: list[dict] | None = None) -> dict:
    """Inverse control test: the network picks a force for each target peak; MuJoCo throws; we measure."""
    pred = Predictor(model)
    if pred.spec["target"] != "max_height":
        raise MLError("wrong_target", "closed_loop_height needs a model whose target is max_height.")
    if "force" not in pred.spec["inputs"]:
        raise MLError("wrong_inputs", "closed_loop_height needs a model that takes force as an input.")
    site = _Site(cfg)
    forces = [20.0 + 2.0 * i for i in range(91)]       # 20 .. 200 N
    rnd = random.Random(0)
    if targets is None:
        targets = [round(rnd.uniform(1.6, 6.5 if jump else 4.3), 2) for _ in range(12)]

    def engine_peak(force: float) -> float | None:
        row = site.throw(oid, force, _direction(elevation, 0.0), "jump" if jump else "stand")
        return None if row is None else row["max_height"]

    lin = None
    if linear_rows:
        cols = pred.spec["inputs"]
        usable = [r for r in linear_rows if all(isinstance(r.get(c), (int, float)) for c in cols + ["max_height"])]
        lin = least_squares([[r[c] for c in cols] for r in usable], [r["max_height"] for r in usable])

    controllers = {"network": lambda rows: pred.predict_rows(rows)}
    if lin is not None:
        controllers["linear_least_squares"] = lambda rows: lin([[r[c] for c in pred.spec["inputs"]] for r in rows])
    base = {f: _site_inputs(site, oid, f, elevation, jump) for f in forces}

    results = {}
    for cname, f_pred in controllers.items():
        preds = f_pred([base[f] for f in forces])
        trials = []
        for tgt in targets:
            best_f = min(forces, key=lambda f: abs(preds[forces.index(f)] - tgt))
            measured = engine_peak(best_f)
            err = None if measured is None else abs(measured - tgt)
            trials.append({"target": tgt, "chosen_force": best_f,
                           "predicted_peak": round(preds[forces.index(best_f)], 4),
                           "measured_peak": measured, "error": None if err is None else round(err, 4),
                           "hit": err is not None and err <= tolerance})
        errs = [t["error"] for t in trials if t["error"] is not None]
        results[cname] = {"hits": sum(t["hit"] for t in trials), "trials": len(trials),
                          "hit_rate": round(sum(t["hit"] for t in trials) / len(trials), 4),
                          "mean_abs_error": round(sum(errs) / len(errs), 4) if errs else None,
                          "detail": trials}
    # A naive rule of thumb (no learning): force proportional to target height.
    trials = []
    for tgt in targets:
        f = max(20.0, min(200.0, 20.0 + (tgt - 1.2) * 45.0))
        measured = engine_peak(f)
        err = None if measured is None else abs(measured - tgt)
        trials.append({"target": tgt, "chosen_force": round(f, 2), "measured_peak": measured,
                       "error": None if err is None else round(err, 4), "hit": err is not None and err <= tolerance})
    errs = [t["error"] for t in trials if t["error"] is not None]
    results["rule_of_thumb"] = {"hits": sum(t["hit"] for t in trials), "trials": len(trials),
                                "hit_rate": round(sum(t["hit"] for t in trials) / len(trials), 4),
                                "mean_abs_error": round(sum(errs) / len(errs), 4) if errs else None,
                                "detail": trials}
    return {"test": "closed_loop_height", "elevation_deg": elevation, "jump": jump, "object": oid,
            "tolerance": tolerance, "targets": targets, "controllers": results,
            "note": "Every measured_peak is the physics engine's own record of a real throw with the chosen force."}
