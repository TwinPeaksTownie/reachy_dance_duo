import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

/**
 * MJCF (MuJoCo XML) Parser & Loader
 * Parses reachy_mini.xml to build the robot structure using Three.js Groups.
 * Loads matching STL meshes from assets/ folder.
 * 
 * extracted from index_backup.html
 */
export class MJCFLoader {
    constructor(manager) {
        this.manager = manager || THREE.DefaultLoadingManager;
        this.stlLoader = new STLLoader(this.manager);
        this.meshPath = 'assets/meshes_optimized/'; // Use optimized GLB meshes
    }

    load(url, onLoad, onProgress, onError) {
        const loader = new THREE.FileLoader(this.manager);
        loader.setPath(this.path);
        loader.load(url, async (text) => {
            try {
                // 1. Parse XML and build structure
                const { group, promises } = this.parse(text);

                // 2. Wait for all meshes to load
                if (promises.length > 0) {
                    await Promise.all(promises);
                }

                onLoad(group);

            } catch (e) {
                if (onError) onError(e);
            }
        }, onProgress, onError);
    }

    parse(text) {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(text, 'text/xml');
        const mujoco = xmlDoc.getElementsByTagName('mujoco')[0];

        const robotGroup = new THREE.Group();
        robotGroup.name = 'reachy_mini';
        const promises = [];

        // 1. Parse Assets (Mesh Name -> File Path Map)
        this.meshes = {};
        const assets = mujoco.getElementsByTagName('asset')[0];
        if (assets) {
            const meshes = assets.getElementsByTagName('mesh');
            for (const m of meshes) {
                const file = m.getAttribute('file');
                let name = m.getAttribute('name');

                if (!name && file) {
                    const parts = file.split('/');
                    const filename = parts[parts.length - 1];
                    name = filename.replace(/\.[^/.]+$/, "");
                }

                if (name && file) {
                    this.meshes[name] = file;
                }
            }
        }

        // 2. Parse Worldbody
        const worldbody = mujoco.getElementsByTagName('worldbody')[0];
        if (worldbody) {
            this.parseBodyChildren(worldbody, robotGroup, promises);
        }

        return { group: robotGroup, promises: promises };
    }

    parseBodyChildren(xmlNode, parent3D, promises) {
        const children = Array.from(xmlNode.children);

        for (const child of children) {
            const tagName = child.tagName.toLowerCase();

            if (tagName === 'body') {
                const bodyGroup = new THREE.Group();
                bodyGroup.name = child.getAttribute('name') || '';

                // Position
                if (child.hasAttribute('pos')) {
                    const pos = child.getAttribute('pos').trim().split(/\s+/).map(Number);
                    bodyGroup.position.set(pos[0], pos[1], pos[2]);
                }

                // Quaternion (MJCF: w x y z -> Three.js: x y z w)
                if (child.hasAttribute('quat')) {
                    const quat = child.getAttribute('quat').trim().split(/\s+/).map(Number);
                    bodyGroup.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
                }

                parent3D.add(bodyGroup);
                this.parseBodyChildren(child, bodyGroup, promises);
            }
            else if (tagName === 'geom') {
                const type = child.getAttribute('type');
                const classAttr = child.getAttribute('class');

                // SKIP COLLISION MESHES
                if (classAttr === 'collision' || child.getAttribute('group') === '3') {
                    continue;
                }

                if (type === 'mesh') {
                    const meshName = child.getAttribute('mesh');
                    if (meshName) {
                        // TRIGGER LOAD
                        const p = this.loadGeomMesh(meshName, parent3D, child);
                        promises.push(p);
                    }
                }
            }
            else if (tagName === 'site') {
                // SITE PARSING for Connection Points
                const siteGroup = new THREE.Group();
                siteGroup.name = child.getAttribute('name') || '';
                siteGroup.userData.type = 'site';

                // Position
                if (child.hasAttribute('pos')) {
                    const pos = child.getAttribute('pos').trim().split(/\s+/).map(Number);
                    siteGroup.position.set(pos[0], pos[1], pos[2]);
                }

                // Quaternion
                if (child.hasAttribute('quat')) {
                    const quat = child.getAttribute('quat').trim().split(/\s+/).map(Number);
                    siteGroup.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
                }

                parent3D.add(siteGroup);
            }
            else if (tagName === 'joint') {
                const name = child.getAttribute('name');
                if (name) {
                    parent3D.userData.jointName = name;
                }
            }
        }
    }

    loadGeomMesh(meshName, parent3D, geomNode) {
        return new Promise((resolve) => {
            // Resolve filename from assets map
            let filename = this.meshes[meshName];
            if (!filename) filename = meshName + ".stl";

            const url = `${this.meshPath}${filename}`;

            this.stlLoader.load(url,
                (geometry) => {
                    // HEURISTIC MATERIAL MAPPING
                    // Default to Glossy White for Body
                    let color = 0xffffff;
                    let emissive = 0x333333;
                    let roughness = 0.1; // GLOSSY
                    let metalness = 0.0; // PLASTIC

                    const matName = geomNode.getAttribute('material') || '';
                    const meshNameLower = meshName.toLowerCase();
                    const matNameLower = matName.toLowerCase();

                    // IDENTIFY DARK PARTS (Base, Tech, Camera, Caps)
                    const isDark =
                        matNameLower.includes('black') ||
                        matNameLower.includes('dark') ||
                        matNameLower.includes('antenna_material') ||
                        matNameLower.includes('cap') ||
                        matNameLower.includes('speaker') ||
                        meshNameLower.includes('arducam') ||  // Camera
                        meshNameLower.includes('foot') ||     // Base/Foot
                        meshNameLower.includes('bearing');    // Mechanical bits

                    // IDENTIFY GREY/SILVER PARTS (Actuators, rods)
                    const isSilver =
                        meshNameLower.includes('stewart') ||
                        meshNameLower.includes('rod') ||
                        meshNameLower.includes('link') ||
                        meshNameLower.includes('arm');

                    // IDENTIFY GLOWING PARTS (Antennae/Horns)
                    // Strict match for 'antenna' mesh to avoid coloring 'antenna_body' or 'antenna_holder'
                    const isAntenna = meshNameLower.includes('horn') || meshNameLower === 'antenna';

                    // IDENTIFY REFLECTIVE PARTS (Eyes/Lens)
                    // "Lens" matches 'lens_cap' too, so we must separate distinct glass from plastic caps.
                    const isGlass = meshNameLower.includes('lens') && !meshNameLower.includes('cap');

                    // IDENTIFY MATTE HOLDERS (Caps, ArduCam, Carter, Glasses Holder)
                    const isHolder =
                        meshNameLower.includes('cap') ||
                        meshNameLower.includes('carter') ||
                        meshNameLower.includes('arducam') ||
                        meshNameLower.includes('glasses');

                    if (isDark) {
                        color = 0x111111;
                        emissive = 0x000000;
                        roughness = 0.8; // Matte
                        metalness = 0.1;
                    }
                    else if (isSilver) {
                        color = 0xaaaaaa; // Light Grey for Silver
                        emissive = 0x111111;
                        roughness = 0.2; // Shiny
                        metalness = 1.0; // Metallic
                    }
                    else if (matNameLower.includes('blue')) {
                        color = 0x0088ff;
                        emissive = 0x002266;
                    }

                    // OVERRIDES
                    if (isAntenna) {
                        // Explicit Override: Force Black Base + Faint White Glow
                        // This matches original hardware (Black) but adds glow for visibility against darkness.
                        color = 0x111111; // Black Base
                        emissive = 0xffffff;
                        roughness = 0.6; // Matte
                        metalness = 0.1;
                    }

                    if (isGlass) {
                        // Really Reflective Lenses (Big/Small/Fisheye)
                        // DIELECTRIC BLACK creates sharp white highlights on black body
                        color = 0x000000;
                        emissive = 0x000000;
                        roughness = 0.0; // Perfect Mirror Sharpness
                        metalness = 0.0; // Plastic/Glass (Not Metal)
                    }

                    if (isHolder) {
                        // Matte Textured Plastic (Caps, Camera Holder)
                        color = 0x111111; // Dark Grey/Black
                        emissive = 0x000000;
                        roughness = 0.9; // Very Matte (Textured look)
                        metalness = 0.1; // Plastic
                    }

                    const material = new THREE.MeshStandardMaterial({
                        color: color,
                        roughness: roughness,
                        metalness: metalness,
                        emissive: emissive,
                        emissiveIntensity: (isAntenna) ? 0.01 : ((isDark) ? 0.0 : ((isSilver) ? 0.2 : 0.3))
                    });


                    const mesh = new THREE.Mesh(geometry, material);

                    if (geomNode.hasAttribute('pos')) {
                        const pos = geomNode.getAttribute('pos').trim().split(/\s+/).map(Number);
                        mesh.position.set(pos[0], pos[1], pos[2]);
                    }
                    if (geomNode.hasAttribute('quat')) {
                        const quat = geomNode.getAttribute('quat').trim().split(/\s+/).map(Number);
                        mesh.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
                    }

                    parent3D.add(mesh);
                    resolve();
                },
                undefined,
                (err) => {
                    console.warn(`Could not load mesh: ${url}`, err);
                    resolve();
                }
            );
        });
    }
}
