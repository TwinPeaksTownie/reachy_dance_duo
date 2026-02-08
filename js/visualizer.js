import * as THREE from 'three';
import { SVGLoader } from 'three/addons/loaders/SVGLoader.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';

// --- VISUAL EFFECTS ---
function createNoteTexture(type = 0) {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.font = 'bold 48px Arial';
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(type === 0 ? '♪' : '♫', 32, 32);
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
}

// Helper: VIBGYOR Rainbow Mapping
const rainbowStops = [
    { t: 0.00, c: new THREE.Color(0x8B00FF) }, // Violet
    { t: 0.20, c: new THREE.Color(0x4B0082) }, // Indigo
    { t: 0.40, c: new THREE.Color(0x0000FF) }, // Blue
    { t: 0.50, c: new THREE.Color(0x00FF00) }, // Green (Center)
    { t: 0.60, c: new THREE.Color(0xFFFF00) }, // Yellow
    { t: 0.80, c: new THREE.Color(0xFF7F00) }, // Orange
    { t: 1.00, c: new THREE.Color(0xFF0000) }  // Red
];
const _tempColor = new THREE.Color();

export function getRainbowColor(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 0; i < rainbowStops.length - 1; i++) {
        const start = rainbowStops[i];
        const end = rainbowStops[i + 1];
        if (t >= start.t && t <= end.t) {
            const localT = (t - start.t) / (end.t - start.t);
            return _tempColor.copy(start.c).lerp(end.c, localT);
        }
    }
    return _tempColor.copy(rainbowStops[rainbowStops.length - 1].c);
}

export class MusicNoteSystem {
    constructor(scene) {
        this.scene = scene;
        this.particles = [];
        this.textures = [createNoteTexture(0), createNoteTexture(1)];
        this.maxParticles = 50;

        // Pre-create pool
        for (let i = 0; i < this.maxParticles; i++) {
            const material = new THREE.SpriteMaterial({
                map: this.textures[i % 2],
                color: 0xffffff,
                transparent: true,
                opacity: 0,
                depthWrite: false,
                blending: THREE.AdditiveBlending
            });
            const sprite = new THREE.Sprite(material);
            sprite.scale.set(0.5, 0.5, 0.5);
            sprite.visible = false;
            this.scene.add(sprite);
            this.particles.push({
                mesh: sprite,
                active: false,
                life: 0,
                maxLife: 0,
                velocity: new THREE.Vector3(),
                swayOffset: Math.random() * 100, // Phase shift for sine wave
                swaySpeed: 1 + Math.random()
            });
        }
    }

    spawn(position, color) {
        // Find inactive particle
        const p = this.particles.find(p => !p.active);
        if (!p) return;

        p.active = true;
        p.life = 0;
        p.maxLife = 1.0 + Math.random() * 1.0; // 1-2 seconds

        p.mesh.visible = true;
        p.mesh.position.copy(position);
        // Add random offset around spawn point
        p.mesh.position.x += (Math.random() - 0.5) * 0.5;
        p.mesh.position.y += (Math.random() - 0.5) * 0.5;
        p.mesh.position.z += (Math.random() - 0.5) * 0.5;

        p.mesh.material.color.copy(color);
        p.mesh.material.rotation = (Math.random() - 0.5) * 0.5; // Slight tilt
        p.mesh.scale.setScalar(0.3 + Math.random() * 0.3);

        // Upward velocity with drift
        p.velocity.set(
            (Math.random() - 0.5) * 0.1, // Slight X drift
            0.5 + Math.random() * 0.5,   // Upward speed
            (Math.random() - 0.5) * 0.1  // Slight Z drift
        );
    }

    update(dt) {
        this.particles.forEach(p => {
            if (!p.active) return;

            p.life += dt;
            if (p.life >= p.maxLife) {
                p.active = false;
                p.mesh.visible = false;
                return;
            }

            // Normalized life (0 to 1)
            const t = p.life / p.maxLife;

            // Physics
            p.mesh.position.addScaledVector(p.velocity, dt);

            // Sway: Add sin wave to X position
            const sway = Math.sin(p.life * 5 + p.swayOffset) * 0.02;
            p.mesh.position.x += sway;

            // Fade In / Fade Out
            // Fade in quickly (10%), Fade out slowly (last 50%)
            let opacity = 1;
            if (t < 0.1) opacity = t / 0.1;
            else if (t > 0.5) opacity = 1 - ((t - 0.5) / 0.5);

            p.mesh.material.opacity = opacity;
        });
    }
}

// --- SVG 3D SYSTEM ---
export class SVG3DSystem {
    constructor(scene) {
        this.scene = scene;
        this.group = new THREE.Group();
        this.scene.add(this.group);
        this.floatTime = 0;
        // Default Config
        this.config = { z: -4.5, y: 1.4, drift: 0.4, color: '#f7cf02', redBeam: 0.35, beamFeather: 0.8 };
        this.loadedMesh = null;
    }

    load(url) {
        const loader = new SVGLoader();
        loader.load(url, (data) => {
            // Prevent Duplication: Clear existing meshes
            this.group.clear();

            const paths = data.paths;
            const group = new THREE.Group();

            // Track Global Bounds for Planar UV Mapping
            let globalMin = new THREE.Vector3(Infinity, Infinity, Infinity);
            let globalMax = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
            const coreMeshes = [];

            for (let i = 0; i < paths.length; i++) {
                const path = paths[i];
                // Material is placeholder; updateTexture() will behave correct one
                const material = new THREE.MeshStandardMaterial({
                    color: new THREE.Color(this.config.color),
                    emissive: new THREE.Color(this.config.color),
                    emissiveIntensity: 2.0,
                    roughness: 0.2,
                    metalness: 0.1
                });

                const shapes = SVGLoader.createShapes(path);

                for (let j = 0; j < shapes.length; j++) {
                    const shape = shapes[j];
                    // 1. Core Geometry (Clean, Single Layer)
                    const geometry = new THREE.ExtrudeGeometry(shape, {
                        depth: 80, // Much thicker (Was 20)
                        bevelEnabled: true,
                        bevelThickness: 2,
                        bevelSize: 2,
                        bevelSegments: 3
                    });
                    geometry.scale(1, -1, 1);

                    // Compute Bounds locally to update Global
                    geometry.computeBoundingBox();
                    globalMin.min(geometry.boundingBox.min);
                    globalMax.max(geometry.boundingBox.max);

                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.name = "CoreLetter";
                    group.add(mesh);
                    coreMeshes.push(mesh);
                }
            }

            // PASS 2: Apply Planar UVs based on Global Bounds
            coreMeshes.forEach(mesh => {
                this.applyPlanarUVs(mesh.geometry, globalMin, globalMax);
            });

            // Center the Group
            const box = new THREE.Box3().setFromObject(group);
            const center = box.getCenter(new THREE.Vector3());

            // Offset each child to center it locally
            group.children.forEach(child => {
                child.position.x -= center.x;
                child.position.y -= center.y;
                child.position.z -= center.z;
            });

            // Final Placement
            group.scale.set(0.002, 0.002, 0.002); // Adjust scale to fit scene
            group.position.set(0, 2.0, -2.0); // Move closer (was -3.0)

            // Add PointLight
            const light = new THREE.PointLight(0xffff00, 3, 10);
            light.position.set(0, 0, 20);
            group.add(light);

            // --- ADD 3D TITLE CARD TEXT ---
            const fontLoader = new FontLoader();
            fontLoader.load('https://unpkg.com/three@0.160.0/examples/fonts/helvetiker_bold.typeface.json', (font) => {

                const textMat = new THREE.MeshStandardMaterial({
                    color: 0xffffff,
                    emissive: 0xffffff,
                    emissiveIntensity: 1.5,
                    roughness: 0.1,
                    metalness: 0.2
                });

                // 1. "REACHY DANCE DUO"
                const textGeo1 = new TextGeometry('REACHY DANCE DUO', {
                    font: font,
                    size: 35,
                    height: 5,
                    curveSegments: 4,
                    bevelEnabled: true,
                    bevelThickness: 1,
                    bevelSize: 1,
                    bevelSegments: 2
                });
                textGeo1.computeBoundingBox();
                const center1 = textGeo1.boundingBox.getCenter(new THREE.Vector3());
                const mesh1 = new THREE.Mesh(textGeo1, textMat);
                mesh1.name = "TitleText";
                mesh1.position.x = -center1.x;
                mesh1.position.y = 60;
                mesh1.position.z = 50;
                group.add(mesh1);

                // 2. "TWIN BEATS"
                const textGeo2 = new TextGeometry('TWIN BEATS', {
                    font: font,
                    size: 80,
                    height: 10,
                    curveSegments: 4,
                    bevelEnabled: true,
                    bevelThickness: 3,
                    bevelSize: 2,
                    bevelSegments: 3
                });
                textGeo2.computeBoundingBox();
                const center2 = textGeo2.boundingBox.getCenter(new THREE.Vector3());
                const mesh2 = new THREE.Mesh(textGeo2, textMat);
                mesh2.name = "TitleText";
                mesh2.position.x = -center2.x;
                mesh2.position.y = -60;
                mesh2.position.z = 80;
                mesh2.scale.x = 1.2;
                group.add(mesh2);
            });

            this.group.add(group);
            this.loadedMesh = group;

            // Force Initial Texture
            this.updateTexture(this.config.color, this.config.redBeam, this.config.beamFeather);

            console.log("✅ SVG Loaded");
            window.dispatchEvent(new CustomEvent('svg-loaded'));
        });
    }

    update(dt) {
        if (this.loadedMesh) {
            this.floatTime += dt;
            const yawOffset = this.config.drift > 0 ? Math.sin(this.floatTime * this.config.drift) * 0.15 : 0;

            this.loadedMesh.position.z = this.config.z;
            this.loadedMesh.position.y = this.config.y + Math.sin(this.floatTime * 0.5) * 0.1;
            this.loadedMesh.rotation.y = yawOffset;
        }
    }

    createGradientTexture(baseColorHex, beamY = 0.5, feather = 0.5) {
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 256;
        const ctx = canvas.getContext('2d');

        const grad = ctx.createLinearGradient(0, 0, 0, 256);
        const base = new THREE.Color(baseColorHex).getStyle();
        const pos = 1.0 - beamY;
        const coreWidth = 0.02; // Consistent solid red core
        const maxSpread = 0.8;  // Allow much wider transition zone
        const currentSpread = feather * maxSpread;

        grad.addColorStop(0.0, base);
        const s1 = Math.max(0, pos - coreWidth - currentSpread);
        const s2 = Math.max(0, pos - coreWidth);
        const s3 = Math.min(1, pos + coreWidth);
        const s4 = Math.min(1, pos + coreWidth + currentSpread);

        grad.addColorStop(s1, base);
        grad.addColorStop(s2, '#ff0000');
        grad.addColorStop(s3, '#ff0000');
        grad.addColorStop(s4, base);
        grad.addColorStop(1.0, base);

        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 32, 256);

        return new THREE.CanvasTexture(canvas);
    }

    applyPlanarUVs(geometry, globalMin, globalMax) {
        if (!globalMin || !globalMax) return;
        const rangeY = globalMax.y - globalMin.y;
        const rangeX = globalMax.x - globalMin.x;
        const uvAttribute = geometry.attributes.uv;
        const posAttribute = geometry.attributes.position;

        for (let i = 0; i < posAttribute.count; i++) {
            const x = posAttribute.getX(i);
            const y = posAttribute.getY(i);
            const u = (x - globalMin.x) / rangeX;
            const v = (y - globalMin.y) / rangeY;
            uvAttribute.setXY(i, u, v);
        }
        uvAttribute.needsUpdate = true;
    }

    updateTexture(baseColorHex, beamY = 0.5, feather = 0.5) {
        const newTex = this.createGradientTexture(baseColorHex, beamY, feather);
        if (this.loadedMesh) {
            this.loadedMesh.children.forEach(group => {
                group.traverse((child) => {
                    if (child.isMesh && child.name === "CoreLetter") {
                        child.material.emissiveMap = newTex;
                        child.material.emissiveIntensity = 1.0;
                        child.material.emissive.setHex(0xffffff);
                        child.material.color.set(baseColorHex);
                        child.material.needsUpdate = true;
                    }
                });
            });
        }
    }
}
