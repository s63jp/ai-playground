// player.js – client side controller for human input
// No external imports; relies on global THREE object from three.module.min.js

class PlayerClient {
  constructor(baseUrl = '') {
    this.base = baseUrl;
    this.state = {move: [0,0], yaw: 0, pitch: 0, jump: false, run: false};
    this.keyMap = {
      ArrowUp: [0, 1], ArrowDown: [0, -1], ArrowLeft: [-1, 0], ArrowRight: [1, 0],
      KeyW: [0, 1], KeyS: [0, -1], KeyA: [-1, 0], KeyD: [1, 0],
      Space: 'jump', ShiftLeft: 'run', ShiftRight: 'run'
    };
    this.activeKeys = new Set();
    this.lastSent = 0;
    this.cameraMode = 'first_person';
    this.initListeners();
    this.attachCamera();
    this.pollState();
  }

  initListeners() {
    window.addEventListener('keydown', (e) => this.onKey(e, true));
    window.addEventListener('keyup', (e) => this.onKey(e, false));
    const canvas = document.querySelector('canvas');
    canvas.addEventListener('click', () => canvas.requestPointerLock());
    document.addEventListener('mousemove', (e) => this.onMouseMove(e));
    document.addEventListener('keydown', (e) => {
      if (e.code === 'KeyF') {
        this.cameraMode = this.cameraMode === 'first_person' ? 'third_person' : 'first_person';
        if (this.camera && this.camera.applyMode) this.camera.applyMode(this.cameraMode);
      }
    });
  }

  onKey(e, down) {
    const map = this.keyMap[e.code];
    if (!map) return;
    if (down) this.activeKeys.add(e.code);
    else this.activeKeys.delete(e.code);
    this.updateStateFromKeys();
  }

  updateStateFromKeys() {
    let moveX = 0, moveY = 0;
    this.state.jump = false;
    this.state.run = false;
    for (const k of this.activeKeys) {
      const v = this.keyMap[k];
      if (Array.isArray(v)) { moveX += v[0]; moveY += v[1]; }
      else if (v === 'jump') this.state.jump = true;
      else if (v === 'run') this.state.run = true;
    }
    this.state.move = [moveX, moveY];
    this.sendInput();
  }

  onMouseMove(e) {
    if (document.pointerLockElement) {
      const sens = 0.1;
      this.state.yaw = (this.state.yaw - e.movementX * sens) % 360;
      this.state.pitch = Math.max(-89, Math.min(89, this.state.pitch - e.movementY * sens));
      this.sendInput();
    }
  }

  async sendInput() {
    const now = performance.now();
    if (now - this.lastSent < 50) return; // ~20Hz
    this.lastSent = now;
    try {
      await fetch(this.base + '/api/player/input', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(this.state)
      });
    } catch (_) {}
  }

  async pollState() {
    try {
      const r = await fetch(this.base + '/api/player/state');
      const data = await r.json();
      if (data.state && this.camera && this.camera.applyPose) this.camera.applyPose(data.state);
    } catch (_) {}
    setTimeout(() => this.pollState(), 200);
  }

  attachCamera(camera) {
    this.camera = camera;
    if (camera && camera.applyMode) camera.applyMode(this.cameraMode);
  }
}

window.PlayerClient = PlayerClient;

