import * as THREE from 'three';

/**
 * HumanoidView renders an articulated skeleton from API state.
 * Expected state format: { skeleton: [{ name, pos: [x,y,z], quat: [w,x,y,z] }, ...] }
 * If absent, no meshes are created (fallback to existing rendering).
 */
export class HumanoidView {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.bodies = {};
    // simple colour map for teams
    this.teamColors = { red: 0xff4444, blue: 0x4444ff, default: 0x888888 };
  }

  _meshFor(name) {
    // map body part name to geometry – simplistic mapping
    const torso = ['torso', 'spine', 'pelvis'];
    const limb = ['upperarm', 'lowerarm', 'hand',
                  'upperleg', 'lowerleg', 'foot', 'head'];
    let geometry;
    if (torso.includes(name)) {
      geometry = new THREE.BoxGeometry(0.3, 0.5, 0.2);
    } else if (limb.includes(name)) {
      geometry = new THREE.CapsuleGeometry(0.07, 0.3, 4, 8);
    } else {
      geometry = new THREE.SphereGeometry(0.1, 8, 8);
    }
    const material = new THREE.MeshStandardMaterial({ color: 0x888888 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = name;
    return mesh;
  }

  setSkeleton(skel) {
    if (!Array.isArray(skel)) return;
    const present = new Set(skel.map(b => b.name));
    // remove stale bodies
    for (const name of Object.keys(this.bodies)) {
      if (!present.has(name)) {
        this.group.remove(this.bodies[name]);
        delete this.bodies[name];
      }
    }
    // create / update bodies
    for (const part of skel) {
      let mesh = this.bodies[part.name];
      if (!mesh) {
        mesh = this._meshFor(part.name);
        this.bodies[part.name] = mesh;
        this.group.add(mesh);
      }
      const [x, y, z] = part.pos || [0, 0, 0];
      mesh.position.set(x, y, z);
      const [w, ix, iy, iz] = part.quat || [1, 0, 0, 0];
      mesh.quaternion.set(ix, iy, iz, w);
    }
  }
}
