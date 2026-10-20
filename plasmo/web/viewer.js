// Visor 3D: carga la vista previa GLB (en mm, Y arriba) sobre una cama de impresión.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

export class Viewer {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.5, 5000);
    this.camera.position.set(160, 140, 220);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x1a1410, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(120, 260, 160);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xffb38a, 1.0);
    rim.position.set(-180, 80, -140);
    this.scene.add(rim);

    this.bed = new THREE.Group();
    this.scene.add(this.bed);
    this.setBed(256, 256);

    this.model = null;
    this.wire = false;
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      requestAnimationFrame(loop);
    };
    loop();
  }

  resize() {
    const w = this.el.clientWidth || 1;
    const h = this.el.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setBed(x, y) {
    this.bed.clear();
    const plate = new THREE.Mesh(
      new THREE.PlaneGeometry(x, y),
      new THREE.MeshStandardMaterial({ color: 0x1c222d, roughness: 0.9, transparent: true, opacity: 0.85 })
    );
    plate.rotation.x = -Math.PI / 2;
    plate.position.y = -0.05;
    this.bed.add(plate);
    const grid = new THREE.GridHelper(Math.max(x, y), Math.round(Math.max(x, y) / 10), 0x3a4352, 0x262d39);
    this.bed.add(grid);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(x, 0.01, y)),
      new THREE.LineBasicMaterial({ color: 0xff7a45, transparent: true, opacity: 0.5 })
    );
    this.bed.add(edges);
  }

  toggleBed() {
    this.bed.visible = !this.bed.visible;
    return this.bed.visible;
  }

  toggleWire() {
    this.wire = !this.wire;
    this.model?.traverse((o) => { if (o.isMesh) o.material.wireframe = this.wire; });
    return this.wire;
  }

  clear() {
    if (this.model) {
      this.scene.remove(this.model);
      this.model.traverse((o) => { if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); } });
      this.model = null;
    }
  }

  async load(url) {
    const gltf = await new GLTFLoader().loadAsync(url);
    this.clear();
    const root = gltf.scene;
    root.traverse((o) => {
      if (!o.isMesh) return;
      const hasColor = !!o.geometry.attributes.color;
      o.material = new THREE.MeshStandardMaterial({
        color: hasColor ? 0xffffff : 0xff8a55,
        vertexColors: hasColor,
        roughness: 0.55,
        metalness: 0.05,
        flatShading: false,
        wireframe: this.wire,
      });
      o.geometry.computeVertexNormals();
    });
    this.model = root;
    this.scene.add(root);
    this.frame();
    const box = new THREE.Box3().setFromObject(root);
    const s = box.getSize(new THREE.Vector3());
    return { x: s.x, y: s.z, z: s.y };
  }

  frame() {
    if (!this.model) return;
    const box = new THREE.Box3().setFromObject(this.model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.75 + 10;
    const dist = radius / Math.sin((this.camera.fov * Math.PI) / 360);
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(0.55, 0.45, 0.85).normalize().multiplyScalar(dist));
    this.camera.near = dist / 100;
    this.camera.far = dist * 20;
    this.camera.updateProjectionMatrix();
  }
}
