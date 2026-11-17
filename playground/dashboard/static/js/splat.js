// SplatRenderer: surface‑conforming decal splats
// Uses THREE global. One InstancedMesh per colour (single draw call per colour).
// Each splat is a small plane (size 0.5m) oriented to the surface normal and rotated
// around that normal by the impact angle. Oldest splats are recycled when the
// per‑colour capacity is reached.

export class SplatRenderer {
  /**
   * @param {THREE.Scene} scene – three.js scene to attach meshes to
   * @param {THREE.Camera} camera – camera (unused but kept for API compatibility)
   * @param {object} [options]
   * @param {number} [options.capacity=2000] – max splats per colour
   */
  constructor(scene, camera, options = {}) {
    this.scene = scene;
    this.camera = camera;
    this.capacity = options.capacity || 2000;
    // Map colour string -> {mesh, count, nextIdx}
    this.pools = new Map();
    this.defaultColour = "#ff4444";
    // Plane geometry for a decal – +Z faces forward.
    this.plane = new THREE.PlaneGeometry(0.5, 0.5);
  }

  // Retrieve or create a pool (InstancedMesh) for a given colour.
  _getPool(colour) {
    if (!this.pools.has(colour)) {
      const material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(colour),
        transparent: true,
        opacity: 0.7,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.InstancedMesh(this.plane, material, this.capacity);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceMatrix.needsUpdate = true;
      this.scene.add(mesh);
      this.pools.set(colour, { mesh, count: 0, nextIdx: 0 });
    }
    return this.pools.get(colour);
  }

  // Add a splat at world position `pos` (THREE.Vector3), with surface `normal`
  // (THREE.Vector3 normalized), rotation `angleDeg` around the normal, and CSS colour.
  addSplatter(pos, normal, angleDeg, colour = null) {
    const col = colour || this.defaultColour;
    const pool = this._getPool(col);
    const { mesh, count, nextIdx } = pool;
    // Quaternion: +Z → normal, then rotate around normal by angleDeg.
    const up = new THREE.Vector3(0, 0, 1);
    const quat = new THREE.Quaternion().setFromUnitVectors(up, normal);
    const rot = new THREE.Quaternion().setFromAxisAngle(normal, THREE.MathUtils.degToRad(angleDeg));
    quat.multiply(rot);
    const matrix = new THREE.Matrix4();
    matrix.compose(pos, quat, new THREE.Vector3(1, 1, 1));
    const idx = count < this.capacity ? count : nextIdx;
    mesh.setMatrixAt(idx, matrix);
    if (count < this.capacity) {
      pool.count = count + 1;
    } else {
      pool.nextIdx = (nextIdx + 1) % this.capacity;
    }
    mesh.instanceMatrix.needsUpdate = true;
  }

  // Handle a projectile impact event – convert to a splat.
  // Payload: {proj_id, pos, normal, angle_deg, surface, speed, t}
  handleEvent(ev) {
    if (!ev || ev.type !== "projectile_impact") return;
    const p = ev.data;
    const pos = new THREE.Vector3(p.pos[0], p.pos[1], p.pos[2]);
    const normal = new THREE.Vector3(p.normal[0], p.normal[1], p.normal[2]).normalize();
    const angle = p.angle_deg || 0;
    // Deterministic colour based on proj_id hash – hue across spectrum.
    const hue = (this._hashString(p.proj_id) % 360) / 360;
    const colour = new THREE.Color().setHSL(hue, 0.6, 0.5).getStyle();
    this.addSplatter(pos, normal, angle, colour);
  }

  // Simple djb2 hash for deterministic colour selection.
  _hashString(str) {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) + hash) + str.charCodeAt(i);
    }
    return hash >>> 0;
  }
}

