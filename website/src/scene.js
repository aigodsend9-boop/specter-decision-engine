import * as THREE from 'three/webgpu';

const GREEN = 0x7df3aa;
const PALE = 0xb9ffcf;
const DARK = 0x06110b;

function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(180,255,207,.9)');
  g.addColorStop(.16, 'rgba(117,243,170,.55)');
  g.addColorStop(.45, 'rgba(85,219,142,.13)');
  g.addColorStop(1, 'rgba(85,219,142,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

function curveTube(points, material) {
  const curve = new THREE.CatmullRomCurve3(points);
  const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, .018, 5, false), material);
  return { curve, tube };
}

export async function initSpecterScene(canvas) {
  if (!canvas) return;
  try {
    const renderer = new THREE.WebGPURenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    await renderer.init();

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, .1, 100);
    camera.position.set(0, 0, 11);
    const rig = new THREE.Group();
    scene.add(rig);
    scene.add(new THREE.AmbientLight(0x8fc8a8, .55));
    const key = new THREE.PointLight(0x89ffb2, 10, 16, 2);
    key.position.set(4.5, 3, 5);
    scene.add(key);

    const coreMaterial = new THREE.MeshStandardMaterial({
      color: DARK, emissive: 0x185f38, emissiveIntensity: 1.25,
      roughness: .28, metalness: .55, transparent: true, opacity: .9,
    });
    const core = new THREE.Mesh(new THREE.IcosahedronGeometry(1.15, 2), coreMaterial);
    rig.add(core);
    const wire = new THREE.LineSegments(
      new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(1.18, 1)),
      new THREE.LineBasicMaterial({ color: PALE, transparent: true, opacity: .24 }),
    );
    rig.add(wire);

    const ringMaterial = new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: .12 });
    const ringA = new THREE.Mesh(new THREE.TorusGeometry(1.62, .012, 5, 96), ringMaterial);
    ringA.rotation.x = 1.08;
    const ringB = new THREE.Mesh(new THREE.TorusGeometry(2.12, .008, 5, 96), ringMaterial.clone());
    ringB.material.opacity = .07;
    ringB.rotation.set(.45, .3, 1.15);
    rig.add(ringA, ringB);

    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: PALE, transparent: true, opacity: .52, blending: THREE.AdditiveBlending, depthWrite: false }));
    sprite.scale.set(4.8, 4.8, 1);
    rig.add(sprite);
    const pathMaterial = new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: .16, blending: THREE.AdditiveBlending, depthWrite: false });
    const paths = [
      curveTube([new THREE.Vector3(.9,.35,0), new THREE.Vector3(1.7,.75,.1), new THREE.Vector3(2.6,1.2,-.1), new THREE.Vector3(3.8,1.35,0)], pathMaterial),
      curveTube([new THREE.Vector3(1.0,0,0), new THREE.Vector3(1.9,.05,-.15), new THREE.Vector3(2.8,.05,.12), new THREE.Vector3(4.15,.05,0)], pathMaterial.clone()),
      curveTube([new THREE.Vector3(.9,-.35,0), new THREE.Vector3(1.7,-.7,.15), new THREE.Vector3(2.65,-1.15,-.1), new THREE.Vector3(3.9,-1.32,0)], pathMaterial.clone()),
    ];
    paths.forEach(({ tube }) => rig.add(tube));

    const nodeGeometry = new THREE.IcosahedronGeometry(.075, 1);
    const nodeMaterial = new THREE.MeshBasicMaterial({ color: PALE, transparent: true, opacity: .8 });
    const nodes = new THREE.InstancedMesh(nodeGeometry, nodeMaterial, 18);
    const matrix = new THREE.Matrix4();
    let ni = 0;
    paths.forEach(({ curve }, branch) => {
      for (let i = 1; i <= 6; i += 1) {
        const p = curve.getPoint(i / 6);
        const scale = i === 6 ? 1.5 : .65 + ((i + branch) % 3) * .12;
        matrix.compose(p, new THREE.Quaternion(), new THREE.Vector3(scale, scale, scale));
        nodes.setMatrixAt(ni++, matrix);
      }
    });
    rig.add(nodes);

    const pulseGeometry = new THREE.SphereGeometry(.055, 10, 8);
    const pulses = paths.map(({ curve }) => {
      const mesh = new THREE.Mesh(pulseGeometry, new THREE.MeshBasicMaterial({ color: PALE }));
      mesh.userData.curve = curve;
      rig.add(mesh);
      return mesh;
    });
    const count = 150;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      const r = 2.2 + Math.random() * 3.8;
      const a = Math.random() * Math.PI * 2;
      positions[i * 3] = Math.cos(a) * r;
      positions[i * 3 + 1] = (Math.random() - .5) * 5.4;
      positions[i * 3 + 2] = Math.sin(a) * r - 1.2;
    }
    const dustGeometry = new THREE.BufferGeometry();
    dustGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const dust = new THREE.Points(dustGeometry, new THREE.PointsMaterial({ color: 0x9ff6bd, size: .025, transparent: true, opacity: .33, depthWrite: false }));
    rig.add(dust);

    const pointer = new THREE.Vector2();
    window.addEventListener('pointermove', (event) => {
      pointer.x = (event.clientX / window.innerWidth - .5) * 2;
      pointer.y = (event.clientY / window.innerHeight - .5) * 2;
    }, { passive: true });

    function placeRig() {
      const mobile = window.innerWidth < 760;
      rig.position.set(mobile ? .5 : 2.55, mobile ? -1.55 : .5, mobile ? -1.8 : -.1);
      rig.scale.setScalar(mobile ? .76 : 1);
    }
    placeRig();

    const clock = new THREE.Clock();
    renderer.setAnimationLoop(() => {
      if (document.hidden) return;
      const t = clock.getElapsedTime();
      core.rotation.set(t * .08, t * .12, Math.sin(t * .2) * .16);
      wire.rotation.copy(core.rotation);
      ringA.rotation.z = t * .08;
      ringB.rotation.y = .3 - t * .045;
      dust.rotation.y = t * .012;
      pulses.forEach((mesh, i) => mesh.position.copy(mesh.userData.curve.getPoint((t * .13 + i * .31) % 1)));
      camera.position.x += (pointer.x * .22 - camera.position.x) * .025;
      camera.position.y += (-pointer.y * .15 - camera.position.y) * .025;
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
    });
    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
      renderer.setSize(window.innerWidth, window.innerHeight, false);
      placeRig();
    }, { passive: true });
  } catch (error) {
    console.warn('Specter scene unavailable; continuing with static UI.', error);
    canvas.hidden = true;
  }
}
