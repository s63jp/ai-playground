// ImpactFX: explosion particle burst visualiser.
// Uses THREE global. Creates an InstancedMesh for the particles and animates them with
// simple gravity, linear drag and fade. Configuration (particle count, directions) is taken
// from the 'explosion' event payload.

export class ImpactFX {
  /**
   * @param {THREE.Scene} scene – three.js scene to attach meshes to
   * @param {THREE.Camera} camera – camera (unused but kept for API compatibility)
   * @param {object} [options]
   * @param {number} [options.maxParticles=2000] – global particle pool size
   */
  constructor(scene, camera, options = {}) {
    this.scene = scene;
    this.camera = camera;
    this.maxParticles = options.maxParticles || 2000;
    this.particleSize = 0.05;
    this.gravity = new THREE.Vector3(0, 0, -9.8);
    this.drag = 0.5; // simple linear drag coefficient
    // Geometry for a particle – small sphere.
    this.geometry = new THREE.SphereGeometry(this.particleSize, 6, 6);
    const material = new THREE.MeshBasicMaterial({ color: 0xffaa33, transparent: true, opacity: 1, depthWrite: false });
    this.mesh = new THREE.InstancedMesh(this.geometry, material, this.maxParticles);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.instanceMatrix.needsUpdate = true;
    this.scene.add(this.mesh);
    // Per‑instance state arrays.
    this.velocities = new Array(this.maxParticles).fill(null);
    this.lifetimes = new Float32Array(this.maxParticles);
    this.age = 0;
    this.nextIdx = 0;
    // Start the animation loop.
    this._animate = this._animate.bind(this);
    requestAnimationFrame(this._animate);
  }

  // Create particles from an explosion event payload.
  // Expected payload fields: pos (xyz), particle_dirs (list of unit vectors),
  // particle_count (int), impulse_magnitude (optional, scales initial speed).
  handleEvent(ev) {
    if (!ev || ev.type !== "explosion") return;
    const p = ev.data;
    const origin = new THREE.Vector3(p.pos[0], p.pos[1], p.pos[2]);
    const dirs = p.particle_dirs || [];
    const count = Math.min(p.particle_count || dirs.length, this.maxParticles);
    const speedScale = p.impulse_magnitude ? p.impulse_magnitude / 10 : 1; // heuristic scaling
    for (let i = 0; i < count; i++) {
      const dirArr = dirs[i] || [0, 0, 1];
      const dir = new THREE.Vector3(dirArr[0], dirArr[1], dirArr[2]).normalize();
      const velocity = dir.multiplyScalar(2 * speedScale); // 2 m/s base speed
      const idx = this.nextIdx;
      this._setParticleMatrix(idx, origin);
      this.velocities[idx] = velocity;
      this.lifetimes[idx] = 2.0; // seconds
      this.nextIdx = (this.nextIdx + 1) % this.maxParticles;
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  _setParticleMatrix(idx, position) {
    const dummy = new THREE.Matrix4();
    dummy.makeTranslation(position.x, position.y, position.z);
    this.mesh.setMatrixAt(idx, dummy);
  }

  // Animation step – advances particles, applies gravity, drag and fade.
  _animate(timestamp) {
    const dt = 0.016; // approx 60 fps fixed step
    for (let i = 0; i < this.maxParticles; i++) {
      const vel = this.velocities[i];
      if (!vel) continue;
      // Update velocity
      vel.addScaledVector(this.gravity, dt);
      vel.multiplyScalar(1 - this.drag * dt);
      // Update position matrix
      const matrix = new THREE.Matrix4();
      this.mesh.getMatrixAt(i, matrix);
      const pos = new THREE.Vector3();
      pos.setFromMatrixPosition(matrix);
      pos.addScaledVector(vel, dt);
      this._setParticleMatrix(i, pos);
      // Fade over lifetime
      this.lifetimes[i] -= dt;
      if (this.lifetimes[i] <= 0) {
        this.velocities[i] = null;
        // Move far away to keep it invisible.
        this._setParticleMatrix(i, new THREE.Vector3(0, 0, -1000));
      } else {
        const meshMat = this.mesh.material;
        meshMat.opacity = Math.max(0, this.lifetimes[i] / 2.0);
        meshMat.needsUpdate = true;
      }
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    requestAnimationFrame(this._animate);
  }
}

