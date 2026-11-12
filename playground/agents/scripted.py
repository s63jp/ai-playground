"""The scripted reference agent.

This is NOT an AI. It is a deterministic, hand-written policy used to exercise the
laboratory end to end (tests, the built-in demonstration, pipeline validation of
the benchmark and ablation machinery). What it "decides" was decided by its author.

What is real when it runs:
  * every measurement comes from the physics engine through the same tool interface
    a language model uses; the policy has no access to the simulator;
  * what it knows in a later session comes only from reading its own memory files
    with read_memory / list_programs. Remove an entry and its behaviour changes;
  * its claims go through the same evidence gate and independent verifiers.

Small per-agent variations (which block it reaches for, its first aiming guess, the
order of its probes) are drawn from a random generator seeded with the agent id.
"""
from __future__ import annotations

import heapq
import math
import random
import re
import zlib

GRID = 0.25
INFLATE = 0.33
FORCE_MAX = 200.0


class _TimeUp(Exception):
    pass


class _Give(Exception):
    pass


class Planner:
    """Grid path-finding over what the agent has observed (never the simulator's state)."""

    def __init__(self, half: float, obstacles: list[tuple[float, float, float, float]]):
        self.half = half
        self.n = int(round(2 * half / GRID))
        self.blocked: set[tuple[int, int]] = set()
        for cx, cy, hx, hy in obstacles:
            x0, x1, y0, y1 = cx - hx - INFLATE, cx + hx + INFLATE, cy - hy - INFLATE, cy + hy + INFLATE
            for i in range(self._i(x0), self._i(x1) + 1):
                for j in range(self._i(y0), self._i(y1) + 1):
                    px, py = self._c(i), self._c(j)
                    if x0 <= px <= x1 and y0 <= py <= y1:
                        self.blocked.add((i, j))

    def _i(self, v: float) -> int:
        return max(0, min(self.n - 1, int(math.floor((v + self.half) / GRID))))

    def _c(self, i: int) -> float:
        return -self.half + (i + 0.5) * GRID

    def free(self, x: float, y: float) -> bool:
        lim = self.half - 0.35
        return abs(x) <= lim and abs(y) <= lim and (self._i(x), self._i(y)) not in self.blocked

    def los(self, a, b) -> bool:
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, int(d / (GRID / 2)))
        return all(self.free(a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
                   for k in range(1, steps))

    def path(self, a, b) -> list[tuple[float, float]] | None:
        if self.los(a, b):
            return [tuple(b)]
        start, goal = (self._i(a[0]), self._i(a[1])), (self._i(b[0]), self._i(b[1]))
        open_ = [(0.0, start)]
        came: dict = {start: None}
        cost = {start: 0.0}
        while open_:
            _, cur = heapq.heappop(open_)
            if cur == goal:
                break
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if not (di or dj):
                        continue
                    nx = (cur[0] + di, cur[1] + dj)
                    if not (0 <= nx[0] < self.n and 0 <= nx[1] < self.n):
                        continue
                    if nx in self.blocked and nx != goal:
                        continue
                    if di and dj and ((cur[0] + di, cur[1]) in self.blocked or (cur[0], cur[1] + dj) in self.blocked):
                        continue
                    c = cost[cur] + math.hypot(di, dj)
                    if c < cost.get(nx, 1e18):
                        cost[nx] = c
                        came[nx] = cur
                        heapq.heappush(open_, (c + math.hypot(goal[0] - nx[0], goal[1] - nx[1]), nx))
        if goal not in came:
            return None
        cells, cur = [], goal
        while cur is not None:
            cells.append(cur)
            cur = came[cur]
        pts = [(self._c(i), self._c(j)) for i, j in reversed(cells)][1:-1] + [tuple(b)]
        out, anchor, k = [], tuple(a), 0
        while k < len(pts):
            far = k
            for m in range(len(pts) - 1, k - 1, -1):
                if self.los(anchor, pts[m]):
                    far = m
                    break
            out.append(pts[far])
            anchor, k = pts[far], far + 1
        return out

    def length(self, a, pts) -> float:
        total, cur = 0.0, a
        for p in pts:
            total += math.hypot(p[0] - cur[0], p[1] - cur[1])
            cur = p
        return total


class ReferenceAgent:
    def __init__(self, system: str, opening: str, tools: list[str]):
        self.opening = opening
        m = re.search(r"You are (\S+?)\.", opening)
        self.agent_id = m.group(1) if m else "agent"
        m = re.search(r"This is session (\d+)", opening)
        self.session = int(m.group(1)) if m else None
        m = re.search(r"Task reference: ([a-z_]+)", opening)
        self.task = m.group(1) if m else None
        m = re.search(r"Target height: (\d+(?:\.\d+)?)", opening)
        self.target_height = float(m.group(1)) if m else None
        self.has_memory = "YOUR MEMORY FILES" in opening
        self.rng = random.Random(zlib.crc32(self.agent_id.encode()))
        self.know: dict[str, dict] = {}
        self.retest: dict[str, dict] = {}
        self.goals: list[dict] = []
        self.programs: set[str] = set()
        self.low = False
        self.objs: dict[str, dict] = {}
        self.pos = [0.0, 0.0, 0.8]
        self.holding = None
        self.half = 12.0
        self.used: set[str] = set()
        self.goal_id: str | None = None
        self.lean: float | None = None
        self.facts: list[str] = []
        self.planner: Planner | None = None

    # ------------------------------------------------------------ primitives
    def call(self, tool: str, **args):
        r = yield ("call", tool, args)
        if r is None:
            raise _TimeUp()
        notice = r.get("notice") or ""
        if "remain" in notice or "is over" in notice:
            self.low = True
        if not r["success"] and r["error"]["code"] in ("session_time_over", "session_closed"):
            raise _TimeUp()
        return r

    def say(self, text: str):
        yield ("say", text)

    def check_budget(self):
        if self.low:
            raise _Give("the session is nearly over")

    def observe(self):
        r = yield from self.call("observe_world")
        self.objs = {o["id"]: o for o in r["objects"]}
        self.pos = r["agent"]["position"]
        self.holding = r["agent"]["holding"]
        self._replan()
        return r

    def _replan(self) -> None:
        obstacles = []
        for o in self.objs.values():
            if o["kind"] == "ground" or o.get("held") or o["id"] == self.holding:
                continue
            p = o["position"]
            if not o["movable"]:
                obstacles.append((p[0], p[1], o["size"]["x"] / 2, o["size"]["y"] / 2))
            elif o["kind"] == "ring":
                if p[2] < 1.0:
                    obstacles.append((p[0], p[1], 0.76, 0.76))
            elif p[2] - 0.3 < 1.7:
                r = 0.3 if o["kind"] == "rock" else 0.25
                obstacles.append((p[0], p[1], r, r))
        self.planner = Planner(self.half, obstacles)

    def _set_pos(self, oid: str, position, held: bool = False) -> None:
        if oid in self.objs:
            self.objs[oid]["position"] = list(position)
            self.objs[oid]["held"] = held
        self._replan()

    def goto(self, x: float, y: float, tol: float = 0.12):
        for _ in range(3):
            pts = self.planner.path((self.pos[0], self.pos[1]), (x, y))
            if pts is None:
                yield from self.observe()
                continue
            ok = True
            for px, py in pts:
                r = yield from self.call("walk_to", x=round(px, 3), y=round(py, 3))
                end = r.get("end_position") or (r.get("details") or {}).get("end_position")
                if end:
                    self.pos = end
                if not r["success"]:
                    ok = False
                    break
            if ok and math.hypot(self.pos[0] - x, self.pos[1] - y) <= tol:
                return True
            if not ok:
                yield from self.observe()
        return math.hypot(self.pos[0] - x, self.pos[1] - y) <= 0.35

    def spot_near(self, x: float, y: float, dist: float, prefer: tuple[float, float] | None = None):
        best, best_len = None, 1e18
        for k in range(8):
            a = k * math.pi / 4
            sx, sy = x + dist * math.cos(a), y + dist * math.sin(a)
            if not self.planner.free(sx, sy):
                continue
            pts = self.planner.path((self.pos[0], self.pos[1]), (sx, sy))
            if pts is None:
                continue
            length = self.planner.length((self.pos[0], self.pos[1]), pts)
            if prefer is not None:
                length += 3.0 * math.hypot(sx - prefer[0], sy - prefer[1])
            if length < best_len:
                best, best_len = (sx, sy), length
        return best

    def fetch(self, oid: str):
        o = self.objs.get(oid)
        if o is not None and math.hypot(o["position"][0] - self.pos[0], o["position"][1] - self.pos[1]) < 1.3:
            r = yield from self.call("pick_up", object_id=oid)
            if r["success"]:
                self.holding = oid
                self._set_pos(oid, o["position"], held=True)
                return r
        for _ in range(3):
            o = self.objs.get(oid)
            if o is None:
                raise _Give(f"{oid} is not in the world any more")
            p = o["position"]
            spot = self.spot_near(p[0], p[1], 1.2 if o["kind"] == "ring" else 0.9)
            if spot is None:
                yield from self.observe()
                continue
            yield from self.goto(*spot)
            r = yield from self.call("pick_up", object_id=oid)
            if r["success"]:
                self.holding = oid
                self._set_pos(oid, p, held=True)
                return r
            yield from self.observe()
        raise _Give(f"could not pick up {oid}")

    def free_blocks(self, near: tuple[float, float] | None = None) -> list[str]:
        ref = near or (self.pos[0], self.pos[1])
        out = [o for o in self.objs.values() if o["kind"] == "block" and not o.get("held")
               and o["id"] not in self.used and o["position"][2] < 0.4]
        out.sort(key=lambda o: (round(math.hypot(o["position"][0] - ref[0], o["position"][1] - ref[1]), 2), o["id"]))
        return [o["id"] for o in out]

    def block_count(self) -> int:
        return sum(1 for o in self.objs.values() if o["kind"] == "block")

    def open_site(self, clearance: float = 1.6, exclude: list | None = None):
        """A patch of open ground, chosen from what has been observed."""
        candidates = [(0.0, 3.0), (-1.5, 4.5), (1.0, -7.0), (-2.0, -7.5), (4.0, -7.0), (-7.5, 0.0),
                      (0.0, 5.0), (-8.0, -8.0), (3.0, -8.5), (-5.0, 0.5), (0.5, 1.5)]
        self.rng.shuffle(candidates)
        for cx, cy in candidates:
            ok = True
            for o in self.objs.values():
                if o["kind"] == "ground":
                    continue
                p = o["position"]
                reach = clearance + (max(o["size"]["x"], o["size"]["y"]) / 2 if not o["movable"] else 0.3)
                if math.hypot(p[0] - cx, p[1] - cy) < reach:
                    ok = False
                    break
            for ex in exclude or []:
                if math.hypot(ex[0] - cx, ex[1] - cy) < 2 * clearance:
                    ok = False
            if ok:
                return (cx, cy)
        return None

    # ------------------------------------------------------------ memory
    def load_memory(self):
        r = yield from self.call("read_memory", file="discoveries")
        for d in r.get("discoveries", []):
            cat = d.get("category")
            if not cat:
                continue
            if d.get("flags", {}).get("challenged") or d["status"] == "hypothesis":
                self.retest[cat] = d
            elif d["status"] in ("confirmed", "probable"):
                self.know[cat] = d
        r = yield from self.call("read_memory", file="goals")
        self.goals = r.get("goals", [])
        r = yield from self.call("list_programs")
        self.programs = {p["name"] for p in r.get("programs", [])}

    def wrap_up(self, reason: str):
        """Leave the memory files in a useful state. Works during the wrap-up window too."""
        try:
            if self.goal_id:
                note = "Unfinished when the session ended."
                if "jump_throw" in self.know:
                    note += " Use the take-off throw."
                if self.lean is not None:
                    note += f" Next aim to try: lean={self.lean:.2f}."
                yield from self.call("update_goal", goal_id=self.goal_id, status="paused", note=note)
            text = f"Session ended ({reason}). " + " ".join(self.facts[-6:])
            yield from self.call("write_diary", text=text.strip())
        except _TimeUp:
            return

    # ------------------------------------------------------------ throwing at the spire
    def spire_spot(self):
        s = self.objs["spire"]
        return (s["position"][0], s["position"][1] - s["size"]["y"] / 2 - 0.9), s

    def throw_at_spire(self, block: str, lean: float, force: float, jumping: bool, vertical: bool = False):
        (sx, sy), _ = self.spire_spot()
        if self.holding != block:
            yield from self.fetch(block)
        yield from self.goto(sx, sy)
        # vertical=True: a height test, thrown nearly straight up and slightly away from the spire.
        direction = [0.0, -0.1, 1.0] if vertical else [0.0, round(lean, 3), 1.0]
        refs = []
        if jumping and "jump_throw" in self.programs:
            p = yield from self.call("execute_program", name="jump_throw",
                                     params={"block": block, "direction": direction, "force": force})
            r = p["results"][-1] if p.get("results") else p
            refs = [x["action_index"] for x in p.get("results", [])]
        else:
            if jumping:
                j = yield from self.call("jump", wait="takeoff")
                refs.append(j["action_index"])
            r = yield from self.call("throw", object_id=block, direction=direction, force=force)
            refs.append(r["action_index"])
        self.holding = None
        if not r.get("success"):
            yield from self.observe()
            raise _Give(f"the throw failed: {r.get('error', {}).get('message')}")
        if r.get("outcome") == "lost":
            self.objs.pop(block, None)
            self._replan()
        elif r.get("final_position"):
            self._set_pos(block, r["final_position"])
        r["_refs"] = refs
        r["_landed"] = "spire" in (r.get("resting_on") or [])
        return r

    def adjust_lean(self, lean: float, r: dict) -> float:
        """Feedback from where the block came down, relative to the side thrown from."""
        _, s = self.spire_spot()
        fp = r.get("final_position") or r.get("first_impact_position") or [0, -99, 0]
        return round(lean + 0.04, 3) if fp[1] < s["position"][1] else round(lean - 0.02, 3)

    def discover_jump_throw(self, block: str, standing: dict | None, lean: float):
        """Height test: a near-vertical standing throw against near-vertical take-off throws, written up."""
        yield from self.say("Before aiming again I want to know how high I can get a block at all. I will "
                            "throw nearly straight up, standing, and then while jumping.")
        block = block if block in self.objs else (self.free_blocks() or [None])[0]
        standing = yield from self.throw_at_spire(block, lean, FORCE_MAX, jumping=False, vertical=True)
        yield from self.say("What if I am already moving upward when I let go? I will jump and throw at the "
                            "moment I leave the ground.")
        exps, last = [], None
        for trial in range(2):
            self.check_budget()
            block = block if block in self.objs else (self.free_blocks() or [None])[0]
            if block is None:
                raise _Give("no block left to throw")
            t = yield from self.throw_at_spire(block, lean, FORCE_MAX, jumping=True, vertical=True)
            last = t
            higher = t["max_height"] > standing["max_height"] + 0.5
            if trial == 0:
                yield from self.say(f"That went to {t['max_height']} m. The standing throw with the same force "
                                    f"reached {standing['max_height']} m. I should repeat it before believing it.")
            e = yield from self.call(
                "record_experiment",
                question="Does a block go higher if I throw it at the moment of take-off from a jump?",
                hypothesis="A throw released at take-off peaks higher than a standing throw with the same force.",
                action_description="Stood beside the spire and threw a block nearly straight up with full force, "
                                   "once standing and once jumping and releasing at take-off.",
                expected_result="A higher peak than the standing throw.",
                actual_result=f"Peak {t['max_height']} m against {standing['max_height']} m standing.",
                measurements={"max_height": t["max_height"], "release_speed": t["release_speed"]},
                action_refs=t["_refs"] + standing["_refs"], supported=bool(higher), needs_followup=trial == 0,
                conclusion="My upward speed at release appears to add to the block's speed.")
            exps.append(e["experiment_id"])
        d = yield from self.call(
            "record_discovery",
            statement=(f"Throwing a block at the moment of take-off from a jump sends it higher than a standing "
                       f"throw: about {last['max_height']} m peak instead of {standing['max_height']} m with the "
                       "same force. Sequence: jump with wait=takeoff, then throw."),
            status="confirmed", experiment_ids=exps, category="jump_throw")
        if d["granted_status"] in ("confirmed", "probable"):
            self.know["jump_throw"] = d
        self.facts.append(f"Take-off throws peak near {last['max_height']} m; standing throws near "
                          f"{standing['max_height']} m (handbook status: {d['granted_status']}).")
        if "jump_throw" not in self.programs:
            p = yield from self.call("save_program", name="jump_throw",
                                     description="Jump and throw the held block at the moment of take-off.",
                                     steps=[{"tool": "jump", "args": {"wait": "takeoff"}},
                                            {"tool": "throw", "args": {"object_id": "${block}",
                                                                       "direction": "${direction}",
                                                                       "force": "${force}"}}])
            if p["success"]:
                self.programs.add("jump_throw")
        return last, block

    def land_on_spire(self, block: str | None, lean: float, max_tries: int = 6):
        """Aim take-off throws at the spire top, correcting from where each one comes down."""
        self.lean = lean
        for _ in range(max_tries):
            self.check_budget()
            if block is None or block not in self.objs:
                (sx, sy), _ = self.spire_spot()
                free = self.free_blocks(near=(sx, sy))
                if not free:
                    raise _Give("no block left to throw")
                block = free[0]
            r = yield from self.throw_at_spire(block, self.lean, FORCE_MAX, jumping=True)
            if r["_landed"]:
                self.used.add(block)
                self.facts.append(f"{block} came to rest on top of the spire with lean={self.lean:.2f}.")
                return block, r
            new = self.adjust_lean(self.lean, r)
            yield from self.say(f"Missed with lean {self.lean:.2f}: the block ended at {r.get('final_position')}. "
                                f"Trying lean {new:.2f}.")
            self.facts.append(f"lean={self.lean:.2f} missed the spire top.")
            self.lean = new
        return None, None

    # ------------------------------------------------------------ the ring
    def ring_probe(self, cycles: int = 2, site=None, record: bool = True):
        """Lay the ring around a block on open ground and see what happens."""
        yield from self.observe()
        if "ring" not in self.objs:
            raise _Give("there is no ring")
        site = site or self.open_site()
        free = self.free_blocks(near=site)
        if site is None or not free:
            raise _Give("nowhere suitable to try the ring")
        base = free[0]
        yield from self.fetch(base)
        yield from self.goto(site[0], site[1] + 0.9)
        r = yield from self.call("place", object_id=base, position=[site[0], site[1]])
        self.holding = None
        if not r["success"]:
            raise _Give("could not set the block down")
        self._set_pos(base, r["final_position"])
        self.used.add(base)
        yield from self.fetch("ring")
        yield from self.goto(site[0], site[1] + 1.25)
        exps, made = [], 0
        for c in range(cycles):
            self.check_budget()
            if c > 0:
                pk = yield from self.call("pick_up", object_id="ring")
                if not pk["success"]:
                    break
            before = yield from self.call("get_object_positions")
            pl = yield from self.call("place", object_id="ring", position=[site[0], site[1]])
            if not pl["success"]:
                break
            w = yield from self.call("wait", seconds=2.0)
            after = yield from self.call("get_object_positions")
            new = sorted(set(after["objects"]) - set(before["objects"]))
            if new:
                made += len(new)
                if c == 0:
                    yield from self.say(f"A new object is here that was not before: {new}. I did not expect that. "
                                        "I will lift the ring and lay it down again to see if it repeats.")
            if record:
                e = yield from self.call(
                    "record_experiment",
                    question="What happens when the ring is laid flat around a block?",
                    hypothesis="Laying the ring flat around a block makes a new block appear.",
                    action_description="Put a block on open ground, laid the ring flat around it, waited two "
                                       "seconds and counted the objects before and after.",
                    expected_result="One more object than before." if c else "Nothing in particular.",
                    actual_result=(f"Objects before: {before['count']}, after: {after['count']}. "
                                   f"New: {new or 'none'}."),
                    measurements={"count_before": before["count"], "count_after": after["count"]},
                    action_refs=[before["action_index"], pl["action_index"], w["action_index"],
                                 after["action_index"]],
                    supported=bool(new), needs_followup=c == 0)
                exps.append(e["experiment_id"])
        self.holding = None
        yield from self.observe()
        if record and exps:
            if "ring_generation" in self.retest:
                d = yield from self.call("update_discovery", discovery_id=self.retest["ring_generation"]["id"],
                                         status="confirmed" if made else "disproven", add_experiment_ids=exps,
                                         note="Re-tested.")
            else:
                d = yield from self.call(
                    "record_discovery",
                    statement="Laying the ring flat on the ground around a single block makes a new block appear "
                              "on top of it after about a second. Lifting the ring and laying it down again "
                              "repeats it.",
                    status="confirmed" if made else "disproven", experiment_ids=exps, category="ring_generation")
            if d.get("granted_status") in ("confirmed", "probable"):
                self.know["ring_generation"] = d
            self.facts.append(f"Ring around a block: {made} new block(s) in {len(exps)} trial(s) "
                              f"(handbook status: {d.get('granted_status')}).")
        return made, base, site

    def ring_generate(self, n: int, site=None):
        """Use the ring technique to make n blocks. Returns how many appeared."""
        made, base, site = yield from self.ring_probe(cycles=n, site=site, record=False)
        return made, base, site

    # ------------------------------------------------------------ towers
    def build_tower(self, site, target: int, existing: list[str] | None = None):
        """Stack blocks at a site. Uses a spare block as a step when the top is out of reach."""
        tx, ty = site
        column = list(existing or [])
        step = None
        while len(column) < target:
            self.check_budget()
            free = [b for b in self.free_blocks(near=site) if b != step]
            if not free:
                break
            b = free[0]
            if step is None:
                yield from self.fetch(b)
                yield from self.goto(tx, ty + 0.9)
                r = yield from self.call("place", object_id=b, position=[tx, ty])
                if not r["success"] and r["error"]["code"] == "out_of_reach":
                    yield from self.say("The top of the column is out of reach from the ground. I will put a "
                                        "block beside it to stand on.")
                    yield from self.goto(tx, ty + 1.65)
                    s = yield from self.call("place", object_id=b, position=[tx, ty + 0.75])
                    self.holding = None
                    if not s["success"]:
                        raise _Give("could not set a step down")
                    self._set_pos(b, s["final_position"])
                    step = b
                    self.used.add(b)
                    continue
            else:
                yield from self.fetch(b)
                yield from self.goto(tx, ty + 1.65)
                c = yield from self.call("climb", direction="south")
                if not c["success"]:
                    yield from self.call("put_down")
                    self.holding = None
                    yield from self.observe()
                    raise _Give("could not climb onto the step while carrying a block")
                self.pos = c["end_position"]
                r = yield from self.call("place", object_id=b, position=[tx, ty])
                m = yield from self.call("move", direction="north", distance=1.3)
                self.pos = m.get("end_position") or (m.get("details") or {}).get("end_position") or self.pos
            self.holding = None
            if not r["success"]:
                yield from self.call("put_down")
                yield from self.observe()
                break
            self._set_pos(b, r["final_position"])
            self.used.add(b)
            if not r.get("stayed_where_placed", True):
                yield from self.observe()
                break
            column.append(b)
        if step is not None:
            self.used.discard(step)
        return column

    # ------------------------------------------------------------ sessions
    def run(self):
        try:
            if self.has_memory:
                yield from self.load_memory()
            if self.task:
                yield from self.run_task()
            else:
                yield from self.run_exploration()
            yield from self.call("end_session", reason="Finished what I set out to do for now.")
        except _Give as g:
            yield from self.wrap_up(str(g))
            try:
                yield from self.call("end_session", reason=str(g))
            except _TimeUp:
                return
        except _TimeUp:
            yield from self.wrap_up("out of time")

    def run_exploration(self):
        yield from self.observe()
        b = yield from self.call("get_world_boundaries")
        self.half = b["walkable_area"]["x"][1]
        self._replan()
        spire_goal = next((g for g in self.goals if "SPIRE" in g["title"].upper()
                           and g["status"] in ("active", "paused")), None)
        done_before = any("SPIRE" in g["title"].upper() and g["status"] == "completed" for g in self.goals)
        if spire_goal or (not done_before and not self.goals):
            yield from self.spire_project(spire_goal)
        if not self.low:
            yield from self.ring_curiosity()
        if not self.low:
            yield from self.call("write_diary", text=" ".join(self.facts[-6:]) or "A quiet session.")

    def spire_project(self, goal: dict | None):
        _, spire = self.spire_spot()
        top = spire["top_height"]
        if goal is None:
            yield from self.call("look_around", radius=8)
            j = yield from self.call("jump")
            yield from self.say(f"The spire is the tallest thing here: its top is {top} m up and my own jump "
                                f"only lifts my feet {j.get('apex_above_takeoff')} m. I would like to get one of "
                                "the blocks to rest on top of it.")
            g = yield from self.call("set_goal", title="Land a block on top of THE SPIRE",
                                     description="Get a block to rest on the spire's top without climbing it.")
            self.goal_id = g["goal_id"]
            lean = round(self.rng.choice([0.10, 0.12, 0.14]), 2)
        else:
            self.goal_id = goal["id"]
            notes = " ".join(h.get("note") or "" for h in goal.get("history", []))
            m = re.findall(r"lean=(\d+(?:\.\d+)?)", notes)
            lean = float(m[-1]) if m else 0.12
            yield from self.call("update_goal", goal_id=self.goal_id, status="active",
                                 note="Picking this up again.")
            yield from self.say("I left this unfinished last time." + (
                " My handbook says a throw at take-off goes much higher than a standing throw, so I will "
                "start with that." if "jump_throw" in self.know else
                " I have no record of a way to throw high enough, so I will have to work it out."))
        (sx, sy), _ = self.spire_spot()
        block = self.free_blocks(near=(sx, sy))[self.rng.randrange(2)]
        if "jump_throw" not in self.know:
            first = yield from self.throw_at_spire(block, lean, FORCE_MAX, jumping=False)
            yield from self.say(f"A full-force standing throw peaked at {first['max_height']} m. The top is at "
                                f"{top} m, so it cannot get there.")
            h = yield from self.call("measure_height", target="spire")
            yield from self.call(
                "record_experiment", question="Can I throw a block onto the spire from the ground?",
                hypothesis="A full-force standing throw lifts a block above the top of the spire.",
                action_description="Stood beside the spire and threw a block upward with the maximum force.",
                expected_result=f"A peak above {top} m.", actual_result=f"Peak {first['max_height']} m.",
                measurements={"max_height": first["max_height"], "top_height": h["top_height"]},
                action_refs=first["_refs"] + [h["action_index"]], supported=False, needs_followup=True,
                conclusion="Full force is not enough from a standing start.")
            self.facts.append(f"Standing full-force throws peak near {first['max_height']} m, below the spire "
                              f"top at {top} m.")
            # A different approach: get higher first.
            terr = self.objs.get("platform_terrace")
            if terr is not None:
                yield from self.say("Maybe I can throw from higher up. The terrace beside the spire is the "
                                    "nearest raised ground.")
                wx = terr["position"][0] - terr["size"]["x"] / 2 - 0.3
                yield from self.goto(wx, terr["position"][1])
                c = yield from self.call("climb", direction="east")
                if not c["success"]:
                    yield from self.call(
                        "record_mistake", what="Tried to climb straight onto the Spire Terrace from the ground.",
                        why=c["error"]["message"],
                        lesson="That ledge is too high to climb in one go. I would need something to stand on.",
                        action_refs=[c["action_index"]])
                    self.facts.append("The Spire Terrace cannot be climbed from the ground in one go.")
                else:
                    self.pos = c["end_position"]
                    m = yield from self.call("move", direction="west", distance=1.5)
                    self.pos = m.get("end_position") or self.pos
                    yield from self.observe()
            self.check_budget()
            yield from self.discover_jump_throw(block, first, lean)
            self.lean = lean
            yield from self.say("So a take-off throw can get high enough. Now I have to aim it at the top.")
        else:
            self.lean = lean
        landed, r = yield from self.land_on_spire(None, self.lean)
        if landed:
            yield from self.finish_spire(landed)
        else:
            raise _Give("could not land a block on the spire this time")

    def finish_spire(self, block: str):
        u = yield from self.call("update_goal", goal_id=self.goal_id, status="completed",
                                 note=f"{block} is resting on the spire.",
                                 claim={"type": "object_on", "params": {"object": block, "surface": "spire"}})
        self.goal_id = None
        ok = (u.get("claim_check") or {}).get("verified")
        self.facts.append(f"Goal reached: {block} on the spire (independent check: {ok}).")
        yield from self.call("assign_meaning", target=[block, "spire"],
                             meaning="A lighthouse: the spire is the tower and the block on top is its lamp.")

    def ring_curiosity(self):
        if "ring_generation" in self.know:
            yield from self.say("My handbook says the ring makes new blocks. I will use it to grow a column "
                                "and see how tall it gets.")
            g = yield from self.call("set_goal", title="Grow a column with the ring, taller than last time",
                                     description="Only by laying the ring down again and again.")
            self.goal_id = g["goal_id"]
            made, base, site = yield from self.ring_generate(4)
            u = yield from self.call("update_goal", goal_id=self.goal_id, status="completed",
                                     note=f"{made} blocks appeared on {base}.",
                                     claim={"type": "tower", "params": {"min_blocks": made + 1}})
            self.goal_id = None
            self.facts.append(f"Ring column: {made} blocks added on {base} (independent check: "
                              f"{(u.get('claim_check') or {}).get('verified')}).")
            return
        yield from self.say("The ring is the only one of its kind here. I wonder what it does around a block.")
        yield from self.ring_probe(cycles=2)

    # ------------------------------------------------------------ benchmark tasks
    def run_task(self):
        yield from self.observe()
        handler = getattr(self, f"task_{self.task}", None)
        if handler is None:
            raise _Give(f"I have no approach for task '{self.task}'")
        yield from handler()

    def task_spire_landing(self):
        (sx, sy), spire = self.spire_spot()
        free = self.free_blocks(near=(sx, sy))
        block = free[0]
        lean = 0.12
        if "jump_throw" not in self.know:
            first = yield from self.throw_at_spire(block, lean, FORCE_MAX, jumping=False)
            if first["_landed"]:
                return
            yield from self.discover_jump_throw(block, first, lean)
        yield from self.land_on_spire(None, lean, max_tries=8)

    def task_throw_height(self):
        target = self.target_height or 6.5
        block = self.free_blocks()[0]
        site = self.open_site() or (0.0, 3.0)
        yield from self.fetch(block)
        yield from self.goto(*site)

        def throw(force, jumping):
            refs = []
            if jumping:
                j = yield from self.call("jump", wait="takeoff")
                refs.append(j["action_index"])
            r = yield from self.call("throw", object_id=block, direction=[0.12, 0.0, 1.0], force=force, wait="impact")
            w = yield from self.call("wait", seconds=1.5)
            o = yield from self.call("inspect_object", object_id=block)
            self.holding = None
            self._set_pos(block, o["object"]["position"])
            if r.get("max_height") is None:
                raise _Give(f"the throw did not report a height: {r.get('error')}")
            r["_refs"] = refs + [r["action_index"]]
            return r

        jumping = "jump_throw" in self.know
        r = yield from throw(FORCE_MAX, jumping)
        if not jumping and r["max_height"] < target - 0.25:
            yield from self.say(f"Full force from standing only reaches {r['max_height']} m; the target is "
                                f"{target} m. I will try throwing at take-off.")
            yield from self.fetch(block)
            standing = r
            r = yield from throw(FORCE_MAX, True)
            if r["max_height"] > standing["max_height"] + 0.5:
                jumping = True
        lo, hi, force = 20.0, FORCE_MAX, FORCE_MAX
        for _ in range(7):
            self.check_budget()
            err = r["max_height"] - target
            if abs(err) <= 0.1:
                break
            if err > 0:
                hi = force
            else:
                lo = force
            if hi - lo < 1.0:
                break
            force = round((lo + hi) / 2, 1)
            yield from self.fetch(block)
            r = yield from throw(force, jumping)

    def task_generate_blocks(self):
        if "ring_generation" in self.know:
            yield from self.ring_generate(1)
            return
        probes = ["stack", "rock", "ring"]
        self.rng.shuffle(probes)
        for probe in probes:
            self.check_budget()
            before = self.block_count()
            if probe == "stack":
                free = self.free_blocks()
                a, b = free[0], free[1]
                yield from self.fetch(a)
                p = self.objs[b]["position"]
                spot = self.spot_near(p[0], p[1], 0.9)
                yield from self.goto(*spot)
                r = yield from self.call("place", object_id=a, position=[p[0], p[1]])
                self.holding = None
                if r["success"]:
                    self._set_pos(a, r["final_position"])
                yield from self.call("wait", seconds=2.0)
            elif probe == "rock" and "rock" in self.objs:
                free = self.free_blocks()
                yield from self.fetch("rock")
                p = self.objs[free[0]]["position"]
                spot = self.spot_near(p[0], p[1], 0.9)
                yield from self.goto(*spot)
                r = yield from self.call("place", object_id="rock", position=[p[0], p[1]])
                self.holding = None
                yield from self.call("wait", seconds=2.0)
            elif probe == "ring":
                yield from self.ring_probe(cycles=1, record=True)
            yield from self.observe()
            if self.block_count() > before:
                return

    def task_five_towers(self):
        sites: list = []
        towers = 0
        if "ring_generation" in self.know:
            yield from self.say("My handbook says the ring makes new blocks appear on the block it surrounds, "
                                "so I can let it grow a column for me.")
            site = self.open_site(exclude=sites)
            made, base, site = yield from self.ring_generate(4, site=site)
            sites.append(site)
            yield from self.fetch("ring")
            ring_home = self.open_site(exclude=sites) or (9.0, -9.0)
            yield from self.goto(ring_home[0], ring_home[1] + 1.25)
            yield from self.call("place", object_id="ring", position=[ring_home[0], ring_home[1]])
            self.holding = None
            yield from self.observe()
            for o in self.objs.values():
                if o["kind"] == "block" and math.hypot(o["position"][0] - site[0], o["position"][1] - site[1]) < 0.4:
                    self.used.add(o["id"])
            towers += 1 if made >= 4 else 0
        while len(self.free_blocks()) >= 6 or (len(self.free_blocks()) >= 5 and towers == 0):
            self.check_budget()
            site = self.open_site(exclude=sites)
            if site is None:
                break
            sites.append(site)
            col = yield from self.build_tower(site, 5)
            yield from self.observe()
            if len(col) >= 5:
                towers += 1
            else:
                break

    def task_limited_structure(self):
        site = self.open_site() or (0.0, 3.0)
        if "ring_generation" in self.know and "ring" in self.objs:
            yield from self.ring_generate(8, site=site)
            return
        yield from self.build_tower(site, len(self.free_blocks()))

    def task_complete_construction(self):
        r = yield from self.observe()
        towers = r.get("towers") or []
        if not towers:
            raise _Give("I cannot find the unfinished column")
        t = towers[0]
        for b in t["blocks"]:
            self.used.add(b)
        site = (t["base_position"][0], t["base_position"][1])
        yield from self.build_tower(site, 5, existing=list(t["blocks"]))


def reference_policy(system: str, opening: str, tools: list[str]):
    agent = ReferenceAgent(system, opening, tools)
    yield from agent.run()
