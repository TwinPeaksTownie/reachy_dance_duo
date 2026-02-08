import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

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
        this.gltfLoader = new GLTFLoader(this.manager);

        // Configure Draco loader for compressed GLB files
        this.dracoLoader = new DRACOLoader();
        this.dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');
        this.gltfLoader.setDRACOLoader(this.dracoLoader);

        this.meshPath = 'assets/';
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

    createMaterialForMesh(meshName, geomNode) {
        let color = 0xffffff;
        let emissive = 0x333333;
        let roughness = 0.1;
        let metalness = 0.0;

        const matName = geomNode.getAttribute('material') || '';
        const meshNameLower = meshName.toLowerCase();
        const matNameLower = matName.toLowerCase();

        const isDark =
            matNameLower.includes('black') ||
            matNameLower.includes('dark') ||
            matNameLower.includes('antenna_material') ||
            matNameLower.includes('cap') ||
            matNameLower.includes('speaker') ||
            meshNameLower.includes('arducam') ||
            meshNameLower.includes('foot') ||
            meshNameLower.includes('bearing');

        const isSilver =
            meshNameLower.includes('stewart_link_ball') ||
            meshNameLower.includes('rod') ||
            meshNameLower.includes('link') ||
            meshNameLower.includes('arm');

        const isAntenna = meshNameLower.includes('horn') || meshNameLower === 'antenna';
        const isGlass = meshNameLower.includes('lens') && !meshNameLower.includes('cap');
        const isHolder =
            meshNameLower.includes('cap') ||
            meshNameLower.includes('carter') ||
            meshNameLower.includes('arducam') ||
            meshNameLower.includes('glasses');

        // Explicitly check for white head components
        const isWhiteHead =
            meshNameLower.includes('head_front') ||
            meshNameLower.includes('head_back') ||
            meshNameLower.includes('head_mic') ||
            meshNameLower.includes('antenna_body') ||
            meshNameLower.includes('body_top') ||
            meshNameLower.includes('body_down');

        if (isWhiteHead) {
            color = 0xffffff;
            emissive = 0x555555; // Boosted for bloom (was 0x333333)
            roughness = 0.3;
            metalness = 0.0;
        }
        else if (isDark) {
            color = 0x111111;
            emissive = 0x000000;
            roughness = 0.8;
            metalness = 0.1;
        }
        else if (isSilver) {
            color = 0xaaaaaa;
            emissive = 0x111111;
            roughness = 0.2;
            metalness = 1.0;
        }
        else if (matNameLower.includes('blue')) {
            color = 0x0088ff;
            emissive = 0x002266;
        }

        if (isAntenna) {
            color = 0x111111;
            emissive = 0xffffff;
            roughness = 0.6;
            metalness = 0.1;
        }

        if (isGlass) {
            color = 0x000000;
            emissive = 0x000000;
            roughness = 0.0;
            metalness = 0.0;
        }

        if (isHolder) {
            color = 0x111111;
            emissive = 0x000000;
            roughness = 0.9;
            metalness = 0.1;
        }

        const emissiveIntensity = isAntenna ? 0.01 : (isWhiteHead ? 0.2 : (isDark ? 0.0 : (isSilver ? 0.2 : 0.3)));

        return new THREE.MeshStandardMaterial({
            color: color,
            roughness: roughness,
            metalness: metalness,
            emissive: emissive,
            emissiveIntensity: emissiveIntensity
        });
    }

    applyTransform(mesh, geomNode) {
        if (geomNode.hasAttribute('pos')) {
            const pos = geomNode.getAttribute('pos').trim().split(/\s+/).map(Number);
            mesh.position.set(pos[0], pos[1], pos[2]);
        }
        if (geomNode.hasAttribute('quat')) {
            const quat = geomNode.getAttribute('quat').trim().split(/\s+/).map(Number);
            mesh.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
        }
    }

    loadGLB(url, material, parent3D, geomNode, resolve) {
        this.gltfLoader.load(
            url,
            (gltf) => {
                let geometry = null;
                gltf.scene.traverse((child) => {
                    if (child.isMesh && !geometry) {
                        geometry = child.geometry;
                    }
                });

                if (geometry) {
                    const mesh = new THREE.Mesh(geometry, material);
                    this.applyTransform(mesh, geomNode);
                    parent3D.add(mesh);
                    resolve();
                } else {
                    console.warn(`No geometry found in GLB: ${url}`);
                    resolve();
                }
            },
            undefined,
            (err) => {
                console.error(`GLTFLoader failed for ${url}:`, err);
                resolve();
            }
        );
    }

    loadSTL(url, material, parent3D, geomNode, resolve) {
        this.stlLoader.load(
            url,
            (geometry) => {
                const mesh = new THREE.Mesh(geometry, material);
                this.applyTransform(mesh, geomNode);
                parent3D.add(mesh);
                resolve();
            },
            undefined,
            (err) => {
                console.warn(`STLLoader failed for ${url}:`, err);
                resolve();
            }
        );
    }

    loadGeomMesh(meshName, parent3D, geomNode) {
        return new Promise((resolve) => {
            let filename = this.meshes[meshName];
            if (!filename) filename = meshName + ".glb";

            const url = `${this.meshPath}${filename}`;
            const ext = filename.split('.').pop().toLowerCase();

            const material = this.createMaterialForMesh(meshName, geomNode);

            if (ext === 'glb' || ext === 'gltf') {
                this.loadGLB(url, material, parent3D, geomNode, resolve);
            } else {
                this.loadSTL(url, material, parent3D, geomNode, resolve);
            }
        });
    }
}
