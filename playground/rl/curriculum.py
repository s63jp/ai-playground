
class Curriculum:
    def __init__(self, phases):
        self.phases = phases
        self.current = 0
        self.step_in_phase = 0
    def set_phase(self, idx):
        self.current = idx
        self.step_in_phase = 0
    def progress(self, env):
        phase = self.phases[self.current]
        # apply gravity scale
        if 'gravity_scale' in phase:
            env.world.opt.gravity[:] = [g * phase['gravity_scale'] for g in env.world.opt.gravity]
        # apply upright assist force (simple placeholder)
        if 'upright_assist' in phase:
            env.world.upright_assist = phase['upright_assist']
        # max steps per phase
        max_steps = phase.get('max_steps', env.max_steps)
        if self.step_in_phase >= max_steps:
            if self.current + 1 < len(self.phases):
                self.set_phase(self.current + 1)
        self.step_in_phase += 1
