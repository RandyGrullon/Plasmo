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
    this.renderer.localClippingEnabled = true;
    el.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.5, 5000);
    this.camera.position.set(160, 140, 220);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotateSpeed = 1.1;
    // gira como tocadiscos hasta que el usuario toma el control
    this.controls.addEventListener("start", () => { this.controls.autoRotate = false; });
    this.anim = null;

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
      if (this.anim) this.anim(performance.now());
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
    const tmp = new THREE.Color();
    root.traverse((o) => {
      if (!o.isMesh) return;
      const colors = o.geometry.attributes.color;
      const hasColor = !!colors;
      if (colors) {
        // el GLB trae los colores de vértice en sRGB; three los espera lineales
        for (let i = 0; i < colors.count; i++) {
          tmp.fromBufferAttribute(colors, i).convertSRGBToLinear();
          colors.setXYZ(i, tmp.r, tmp.g, tmp.b);
        }
        colors.needsUpdate = true;
      }
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
    this.controls.autoRotate = true;
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) this.reveal(root, box);
    return { x: s.x, y: s.z, z: s.y };
  }

  // Revela la pieza de abajo hacia arriba con un plano de corte brillante, como si se imprimiera
  reveal(root, box) {
    this.anim?.(Infinity);
    const plane = new THREE.Plane(new THREE.Vector3(0, -1, 0), box.min.y);
    const meshes = [];
    root.traverse((o) => { if (o.isMesh) { o.material.clippingPlanes = [plane]; meshes.push(o); } });
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const slice = new THREE.Group();
    const sheet = new THREE.Mesh(
      new THREE.PlaneGeometry(size.x * 1.3 + 8, size.z * 1.3 + 8),
      new THREE.MeshBasicMaterial({ color: 0xff8a50, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending })
    );
    sheet.rotation.x = -Math.PI / 2;
    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x * 1.3 + 8, 0.01, size.z * 1.3 + 8)),
      new THREE.LineBasicMaterial({ color: 0xffc27a, transparent: true, opacity: 0.9 })
    );
    slice.add(sheet, edge);
    slice.position.set(center.x, box.min.y, center.z);
    this.scene.add(slice);
    const start = performance.now();
    const duration = 1400 + Math.min(900, size.y * 4);
    this.anim = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const e = 1 - Math.pow(1 - t, 3);
      const y = box.min.y + (size.y + 0.5) * e;
      plane.constant = y;
      slice.position.y = y;
      slice.scale.setScalar(1 + 0.08 * Math.sin(t * Math.PI * 6));
      sheet.material.opacity = 0.22 * (1 - t * 0.7);
      edge.material.opacity = 0.9 * (1 - t * t);
      if (t >= 1) {
        meshes.forEach((m) => { m.material.clippingPlanes = []; });
        this.scene.remove(slice);
        sheet.geometry.dispose(); sheet.material.dispose(); edge.geometry.dispose(); edge.material.dispose();
        this.anim = null;
      }
    };
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
