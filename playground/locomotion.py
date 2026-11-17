"""Gait controller for humanoid agents.

Physics honesty contract
------------------------
- The controller writes ONLY joint-space PD targets via ``world.set_joint_target``
  (applied once per step by PhysicsWorld._step's PD loop). The ``world.target_vel``
  request it also sets feeds the ROOT-TRACTION pathway — which is the BOX agent's
  locomotion model only: for the humanoid the traction is disabled in
  ``PhysicsWorld._step`` and every bit of forward motion comes from leg torque
  against foot-ground contact friction.
- It NEVER writes ``qpos`` or ``qvel``. Forward motion emerges from the capped
  traction force plus limb actuation; balance and limb dynamics are pure physics.
- Recovery is a scheduled joint-target sequence; ``recovery_complete`` is emitted
  only after the sustained fallen-condition (agent_fallen) clears — success is
  measured, never assumed.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULTS = {
    "mode": "stand",
    "speed": 0.0,            # forward speed target (m/s)
    "heading_deg": 0.0,      # yaw in degrees, 0 = world +x
    "step_frequency": 1.0,   # Hz at speed 1.0 (scales with speed, capped 2.4)
    "hip_amplitude": 0.30,   # rad at speed 1.0 (scales with speed, capped 0.95)
    "knee_amplitude": 0.15,  # swing-knee flexion rad at speed 1.0; stance knee stays ~0
    "hip_gain": 45.0,
    "hip_damp": 5.0,
    "knee_gain": 50.0,
    "knee_damp": 5.0,
    "ankle_gain": 25.0,
    "ankle_damp": 1.0,       # ankle foot inertia ~0.003 kg m^2: kd<=I/dt=1.5 or
                             # the damping torque overshoots the dt-step impulse
                             # limit and the free swing foot spins to NaN
    "spine_gain": 35.0,
    "spine_damp": 4.0,
    "shoulder_gain": 25.0,
    "shoulder_damp": 3.0,
    "posture_gain": 150.0,   # stand-mode joint stiffness (contact-driven model:
    "posture_damp": 10.0,    # the legs hold the body through contacts, needs real Nm)
    "traction_ramp_s": 1.5,  # ramp window; 0 = full amplitude immediately
    "crouch_depth": 0.0,     # reserved
    "recovery_timeout_s": 6.0,
}


class GaitController:
    """Phase-based joint-target generator + recovery schedule. PD-targets only."""

    def __init__(self, world, cfg: dict):
        self.world = world
        self.cfg = dict(cfg)
        self.time = 0.0          # controller clock (sim seconds since reset)
        self.phase = 0.0         # gait phase in [0, 1)
        self._recover_t0 = None  # sim time the recovery schedule started
        self._recovery_done = False

    # ---------------------------------------------------------------- configure
    def configure(self, mode: str, speed: float = 1.0, heading: float | None = None) -> None:
        cfg = self.cfg
        cfg["mode"] = mode
        cfg["speed"] = float(speed if mode in ("walk", "run") else 0.0)
        if heading is not None:
            # Public convention: degrees, wrapped to (-180, 180]. (The old
            # "small values are radians" heuristic turned a 3 deg step from
            # move() into a 172 deg turn.)
            h = float(heading)
            wd = ((h + 180.0) % 360.0) - 180.0
            if wd <= -180.0:
                wd = 180.0
            cfg["heading_deg"] = wd
        freq = float(cfg.get("step_frequency", DEFAULTS["step_frequency"]))
        amp = float(cfg.get("hip_amplitude", DEFAULTS["hip_amplitude"]))
        kar = float(cfg.get("knee_amplitude", DEFAULTS["knee_amplitude"]))
        s = max(0.0, float(cfg["speed"]))
        cfg["step_frequency"] = min(2.4, freq * (0.7 + 0.3 * s) if s > 0 else freq * 0.7)
        cfg["hip_amplitude"] = min(0.95, amp * (0.6 + 0.4 * s) if s > 0 else amp * 0.6)
        cfg["knee_amplitude"] = min(0.5, kar * (0.6 + 0.4 * s) if s > 0 else kar * 0.6)
        self.phase = 0.0
        self.time = 0.0
        self._recovery_done = False
        self._apply_drive()

    def _apply_drive(self) -> None:
        """Set the world's traction-velocity request (force-capped root drive).

        This is the same pathway PhysicsWorld.move() drives box agents with:
        _step() converts target_vel into a tau-limited, walk_force-capped ground
        force on the root DOFs each step. No qpos/qvel writes happen here.
        The request ramps in over the first ~1.5 s — a 0→target_vel step dumps
        the full walk_force cap into the rig in one step and kicks the limb
        PDs into their joint limits.
        """
        w = self.world
        if self.cfg.get("mode") in ("walk", "run"):
            rad = math.radians(float(self.cfg.get("heading_deg", 0.0)))
            cap = float(getattr(w.ac, "get", lambda *_: 4.0)("max_speed", 4.0))
            spd = min(float(self.cfg.get("speed", 0.0)), cap)
            ramp = max(0.0, float(self.cfg.get("traction_ramp_s",
                                               DEFAULTS["traction_ramp_s"])))
            if ramp > 0.0:
                spd *= min(1.0, self.time / ramp)  # traction ramp
            w.target_vel[0] = spd * math.cos(rad)
            w.target_vel[1] = spd * math.sin(rad)
        else:
            w.target_vel[:] = 0.0

    def stop(self) -> None:
        self.configure("stop", speed=0.0)

    # ---------------------------------------------------------------- recovery
    def begin_recovery(self) -> None:
        self.cfg["mode"] = "recover"
        self._recover_t0 = self.world.time
        self._recovery_done = False
        self._recovered_since = None
        # Seed the sit-to-stand ramp from the MEASURED fallen pose (joint qpos
        # is a sensor read here, never a write): the legs are driven back to
        # neutral along the same path the fall took.
        w = self.world
        def _q(jname):
            j = w.m.joint(jname)
            return float(w.d.qpos[int(j.qposadr[0])])
        self._rec_hip0 = min(1.45, max(0.0, 0.5 * (_q("left_hip") + _q("right_hip"))))
        self._rec_knee0 = max(-1.45, min(0.0, 0.5 * (_q("left_knee") + _q("right_knee"))))
        self.phase = 0.0
        w.target_vel[:] = 0.0

    def _recovery_targets(self) -> None:
        """Scheduled joint-space PD targets for a collapsed-to-standing attempt.

        Rig convention (Nov 17 re-axising): sagittal joints are X-hinges --
        POSITIVE hip = leg forward; knee NEGATIVE = heel-back flexion (the fold
        direction). The rigid column's fall mode is a leg-fold sit (z ~0.42).
        The recovery is ONE continuous inverse ramp from the measured folded
        pose to neutral: hip and knee run to 0 together so the shin stays
        nearly vertical and the feet stay planted under the knee. Separate
        phase targets overshot through zero, hyperextended the knee and dumped
        the pelvis back onto the ground (Nov 17 probes).
        """
        w = self.world
        t = w.time - (self._recover_t0 if self._recover_t0 is not None else w.time)
        hip0 = getattr(self, "_rec_hip0", 1.0)

        def _q(jname: str) -> float:
            j = w.m.joint(jname)
            return float(w.d.qpos[int(j.qposadr[0])])
        # Stage 1 ROLL (0-1.0 s): rotate the thighs up-forward so the shins
        # pendulum clear of the ground (a soft knee lets them swing).
        if t < 1.0:
            knee_t, hip_t = 0.3, 1.55
        elif t < 1.8:
            # Stage 2 CATCH (1.0-1.8 s): hold the thighs up and pull the shin
            # vertical under the knee (knee = -hip): the foot plants flat.
            knee_t, hip_t = -1.55, 1.55
        if t < 1.8:
            for side in ("left", "right"):
                w.set_joint_target(f"{side}_knee", knee_t, kp=250.0, kd=10.0)
        else:
            # Stage 3 EXIT (1.8-4.8 s): the crouch-exit. Only the hips are
            # driven; the knee target is the 2-LINK IK for the MEASURED
            # hip->ankle distance, so the knee servo follows the planted foot
            # through whatever shin angle the geometry dictates. Pinning the
            # shin angle (vertical or otherwise) forces the foot to slide out
            # from under the knee and dumps the pelvis (Nov 17 traces).
            prog = min(1.0, (t - 1.8) / 3.0)
            hip_t = 1.55 * (1.0 - prog)
            for side in ("left", "right"):
                bid = int(w.m.body(f"{side}_hip").id)
                fbid = int(w.m.body(f"{side}_foot").id)
                hp = np.asarray(w.d.xpos[bid], dtype=float)
                fp = np.asarray(w.d.xpos[fbid], dtype=float)
                d = float(np.linalg.norm(fp - hp))
                # law of cosines for thigh 0.38 / shin 0.44: d^2 = a^2+b^2
                # + 2ab cos(kappa) with kappa the heel-back flexion angle.
                c = (d * d - (0.38 ** 2 + 0.44 ** 2)) / (2 * 0.38 * 0.44)
                kappa = math.acos(max(-1.0, min(1.0, c)))
                w.set_joint_target(f"{side}_knee", -kappa, kp=250.0, kd=10.0)
        # Ankle neutral: with the shin vertical (catch/exit) a 0 ankle keeps
        # the sole flat; during the roll the foot hangs free.
        w.set_joint_target("left_hip", hip_t, kp=200.0, kd=12.0)
        w.set_joint_target("right_hip", hip_t, kp=200.0, kd=12.0)
        w.set_joint_target("left_ankle", 0.0, kp=150.0, kd=6.0)
        w.set_joint_target("right_ankle", 0.0, kp=150.0, kd=6.0)
        w.set_joint_target("spine_L2", 0.0, kp=60.0, kd=5.0)
        if t >= 4.8:
            # Posture hold at neutral standing targets -- the recovered column
            # must actually hold its height, not sag back once the ramp ends.
            for j in ("left_hip", "right_hip", "left_knee", "right_knee",
                      "waist_yaw", "spine_L2"):
                w.set_joint_target(j, 0.0, kp=150.0, kd=10.0)
        w.set_joint_target("left_foot_yaw", 0.0, kp=8.0, kd=1.0)
        w.set_joint_target("right_foot_yaw", 0.0, kp=8.0, kd=1.0)

    # ---------------------------------------------------------------- gait cycle
    LEG_SPAN = 0.82          # hip -> ankle distance at a straight leg (metres)
    SWING_KNEE_AMP = 0.60    # swing-knee flexion bump (rad): the foot must CLEAR
                             # the floor by ~9 cm mid-swing -- a shallow bump
                             # toe-drags the swing leg (~36 N of mu*N) which
                             # cancels the stance push and crawls the walk
    MAX_SWEEP = 0.24         # rad: keeps the CoM ground line inside the sole
                             # span (0.82*sin(a) <= heel-toe reach) so the CoP
                             # moment the ankle must hold stays small
    STANCE_KNEE_KP = 400.0   # holds the straight strut under landing impacts
    STANCE_ANKLE_KP = 600.0  # flat-sole CoP moment; planted-foot inertia is
                             # large. Airborne the foot is ~massless, so gains
                             # switch on ACTUAL CONTACT (with hysteresis), not
                             # phase: a stiff gain on a free foot exceeds the
                             # I/dt step-impulse limit and blows up the solver.
    STANCE_ANKLE_KD = 6.0
    GAIN_DWELL_S = 0.06      # minimum hold time per stiffness state
    SWING_KNEE_KP = 50.0
    SWING_KNEE_KD = 5.0

    def _foot_planted(self, world, side: str) -> bool:
        """True while the foot body (foot_yaw child of the ankle) touches anything."""
        bid = int(world.m.body(f"{side}_foot").id)
        for ci in range(world.d.ncon):
            c = world.d.contact[ci]
            if int(world.m.geom_bodyid[int(c.geom1)]) == bid or \
               int(world.m.geom_bodyid[int(c.geom2)]) == bid:
                return True
        return False

    def _gait_targets(self) -> None:
        w = self.world
        cfg = self.cfg
        freq = float(cfg.get("step_frequency", DEFAULTS["step_frequency"]))
        amp0 = float(cfg.get("hip_amplitude", DEFAULTS["hip_amplitude"]))
        ramp = max(0.0, float(cfg.get("traction_ramp_s", DEFAULTS["traction_ramp_s"])))
        # The ramp eases the STRIDE length (hip sweep) and the waist twist-in —
        # nothing else. Swing clearance stays at full strength from the first
        # step: a ramped swing knee drops the reaching foot onto the floor,
        # the toe catches and the ankle spins off (Nov 17 lab explosion).
        ramp_factor = min(1.0, self.time / ramp) if ramp > 0.0 else 1.0
        a = min(amp0 * ramp_factor, self.MAX_SWEEP)
        ka = self.SWING_KNEE_AMP * min(1.0, float(cfg.get("knee_amplitude", DEFAULTS["knee_amplitude"])) / 0.15)
        self.phase = (self.phase + freq * w.dt) % 1.0
        phi = self.phase
        th = 2.0 * math.pi * phi
        c = math.cos(th)
        s = math.sin(th)
        # Phase layout (Nov 17 redesign, rig faces local +Y):
        #   hip  = +a*cos(2*pi*phi) left, -a*cos right: at phi=0 the left foot
        #     lands AHEAD (hip +a), the body advances over it through phi=0.5
        #     (hip -a, toe-off) while the right leg swings back-to-front; roles
        #     swap at phi=0.5. The stance sweep IS the drive: hip torque
        #     against the planted flat sole pushes the body through real
        #     foot-ground contact.
        #   knee = swing-clearance bump only (heel-back flexion = negative on
        #     an X-hinge); the planted leg stays a straight strut — with a flat
        #     sole the hip->ankle length is constant, so no crouch is needed.
        #   ankle = -(hip+knee): foot pitch about the walk axis sums to zero,
        #     keeping the sole flat. A flat sole is the friction anchor.
        hip_l = a * c
        hip_r = -a * c
        # Rig geometry (Nov 17 re-axising): the figure faces its local +Y and
        # every sagittal limb hinge is an X-hinge, so POSITIVE hip = leg swings
        # forward (+Y) and heel-back knee flexion is NEGATIVE. At phi=0 the
        # left foot lands AHEAD of the body, the body advances over it through
        # phi=0.5 (toe-off) while the right leg swings back-to-front; roles
        # swap at phi=0.5. The stance sweep IS the drive: hip torque against
        # the planted flat sole pushes the body through real foot contact.
        sw_l = max(0.0, -s)   # left-leg swing bump, peaks mid-swing (phi 0.75)
        sw_r = max(0.0, s)    # right-leg swing bump (phi 0.25)
        knee_l = -ka * sw_l
        knee_r = -ka * sw_r
        ankle_l = -(hip_l + knee_l)
        ankle_r = -(hip_r + knee_r)
        # Heading: the legs sweep along the figure's local +Y, so the world
        # walking direction (cos h, sin h) comes from yawing the figure at the
        # waist: local +Y rotated by theta gives (-sin th, cos th), hence
        # th = h - 90 deg. The ramp window eases the twist instead of dumping
        # an instant turn on the planted feet.
        hd = math.radians(float(cfg.get("heading_deg", 0.0)) - 90.0) * ramp_factor
        pg = float(cfg.get("posture_gain", DEFAULTS["posture_gain"]))
        pdd = float(cfg.get("posture_damp", DEFAULTS["posture_damp"]))
        w.set_joint_target("waist_yaw", hd, kp=pg, kd=pdd)
        # Hip gains follow the phase (the hips are heavy and never violate the
        # step-impulse limit); knee/ankle stiffness follows ACTUAL CONTACT so a
        # momentarily airborne foot never carries a stiff gain.
        hg = float(cfg.get("hip_gain", DEFAULTS["hip_gain"]))
        hdd = float(cfg.get("hip_damp", DEFAULTS["hip_damp"]))
        ag = float(cfg.get("ankle_gain", DEFAULTS["ankle_gain"]))
        adg = float(cfg.get("ankle_damp", DEFAULTS["ankle_damp"]))
        kdg = float(cfg.get("knee_damp", DEFAULTS["knee_damp"]))
        shg = 1.2 * pg  # stance-hip drive: the sweep torque is the whole
                        # propulsion of the contact-driven walk
        now = w.time
        if not hasattr(self, "_foot_stiff"):
            self._foot_stiff = {"left": (True, now), "right": (True, now)}
        planted = {"left": self._foot_planted(w, "left"),
                   "right": self._foot_planted(w, "right")}
        # Foot yaw: soft self-alignment. Straight walking keeps the toes on the
        # shin axis; during waist turns the planted feet twist in place against
        # the ground instead of scrubbing sideways on a hard friction anchor
        # (the X-hinge chain has no other yaw compliance — without this dof the
        # turn torque blew the solver apart, feet into the torso, Nov 17).
        fyg, fyd = 8.0, 1.0
        w.set_joint_target("left_foot_yaw", 0.0, kp=fyg, kd=fyd)
        w.set_joint_target("right_foot_yaw", 0.0, kp=fyg, kd=fyd)
        for k in ("left", "right"):
            st, since = self._foot_stiff[k]
            if planted[k] != st and now - since >= self.GAIN_DWELL_S:
                self._foot_stiff[k] = (planted[k], now)
        stiff_l = self._foot_stiff["left"][0]
        stiff_r = self._foot_stiff["right"][0]
        w_st_l = max(0.0, 1.0 - 2.0 * abs(phi - 0.25)) if phi < 0.5 else 0.0
        w_st_r = 1.0 - 2.0 * abs((phi % 1.0) - 0.75) if phi >= 0.5 else 0.0
        w_st_r = max(0.0, w_st_r)
        kp_lh = hg + (shg - hg) * w_st_l
        kp_rh = hg + (shg - hg) * w_st_r
        kp_lk = self.STANCE_KNEE_KP if stiff_l else self.SWING_KNEE_KP
        kp_rk = self.STANCE_KNEE_KP if stiff_r else self.SWING_KNEE_KP
        kp_la = self.STANCE_ANKLE_KP if stiff_l else ag
        kp_ra = self.STANCE_ANKLE_KP if stiff_r else ag
        kd_lk = pdd if stiff_l else self.SWING_KNEE_KD
        kd_rk = pdd if stiff_r else self.SWING_KNEE_KD
        kd_la = self.STANCE_ANKLE_KD if stiff_l else adg
        kd_ra = self.STANCE_ANKLE_KD if stiff_r else adg
        w.set_joint_target("left_hip", hip_l, kp=kp_lh, kd=hdd)
        w.set_joint_target("right_hip", hip_r, kp=kp_rh, kd=hdd)
        w.set_joint_target("left_knee", knee_l, kp=kp_lk, kd=kd_lk)
        w.set_joint_target("right_knee", knee_r, kp=kp_rk, kd=kd_rk)
        w.set_joint_target("left_ankle", ankle_l, kp=kp_la, kd=kd_la)
        w.set_joint_target("right_ankle", ankle_r, kp=kp_ra, kd=kd_ra)
        # Counter-sway spine + arm swing for balance.
        sg = float(cfg.get("spine_gain", DEFAULTS["spine_gain"]))
        sd = float(cfg.get("spine_damp", DEFAULTS["spine_damp"]))
        w.set_joint_target("spine_L2", 0.15 * a * math.sin(2.0 * th), kp=sg, kd=sd)
        w.set_joint_target("left_shoulder", -0.5 * a * math.sin(th),
                           kp=float(cfg.get("shoulder_gain", DEFAULTS["shoulder_gain"])),
                           kd=float(cfg.get("shoulder_damp", DEFAULTS["shoulder_damp"])))
        w.set_joint_target("right_shoulder", 0.5 * a * math.sin(th),
                           kp=float(cfg.get("shoulder_gain", DEFAULTS["shoulder_gain"])),
                           kd=float(cfg.get("shoulder_damp", DEFAULTS["shoulder_damp"])))

    # ---------------------------------------------------------------- per step
    def update(self, world) -> None:
        """Called each physics step before the single PD application point."""
        self.time += world.dt
        mode = self.cfg.get("mode", "stand")
        if mode == "recover":
            world.target_vel[:] = 0.0
            self._recovery_targets()
            fallen = world.agent_fallen() if hasattr(world, "agent_fallen") else False
            t = world.time - (self._recover_t0 if self._recover_t0 is not None else world.time)
            if not self._recovery_done:
                if not fallen:
                    # recovery_complete is emitted only after the measured
                    # upright state (agent_fallen False) SUSTAINS for half a
                    # second -- success is measured, never assumed.
                    if self._recovered_since is None:
                        self._recovered_since = world.time
                    elif world.time - self._recovered_since >= 0.5:
                        self._recovery_done = True
                        world.emit("recovery_complete", after_s=round(t, 3))
                        # Stay in recover mode: the schedule's posture hold
                        # (t >= 4.8) keeps the same targets running. Switching
                        # straight to stand steps the targets to (0, 0) while
                        # the actual joints still lag and can dump the figure
                        # (Nov 17 test run).
                else:
                    self._recovered_since = None
                    if t > float(self.cfg.get("recovery_timeout_s", DEFAULTS["recovery_timeout_s"])):
                        self._recovery_done = True
                        world.emit("recovery_failed", after_s=round(t, 3))
            return
        self._apply_drive()
        if mode in ("walk", "run"):
            self._gait_targets()
        else:  # stand / stop / crouch hold neutral posture
            cfg = self.cfg
            pg = float(cfg.get("posture_gain", DEFAULTS["posture_gain"]))
            pdd = float(cfg.get("posture_damp", DEFAULTS["posture_damp"]))
            for j in ("left_hip", "right_hip", "left_knee", "right_knee",
                      "left_ankle", "right_ankle", "waist_yaw",
                      "left_foot_yaw", "right_foot_yaw"):
                world.set_joint_target(j, 0.0, kp=pg, kd=pdd)
            world.set_joint_target("spine_L2", 0.0, kp=pg * 1.3, kd=pdd)

    # ---------------------------------------------------------------- state
    def get_state(self) -> dict:
        return {"cfg": dict(self.cfg), "time": self.time, "phase": self.phase,
                "recover_t0": self._recover_t0, "recovery_done": self._recovery_done}

    def set_state(self, state: dict) -> None:
        self.cfg.update(dict(state.get("cfg", {})))
        self.time = float(state.get("time", 0.0))
        self.phase = float(state.get("phase", 0.0))
        self._recover_t0 = state.get("recover_t0")
        self._recovery_done = bool(state.get("recovery_done", False))
