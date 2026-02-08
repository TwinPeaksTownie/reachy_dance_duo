import * as THREE from 'three';
import { MJCFLoader } from './mjcf_loader.js';
import { calculateActiveMotorAngles, calculatePassiveJoints, buildHeadPoseMatrix } from './reachy_kinematics.js';

export class ReachyRobot {
    constructor(name, xOffset, mirrored = false) {
        this.name = name;
        this.xOffset = xOffset;
        this.mirrored = mirrored;
        this.group = new THREE.Group();
        this.joints = {};
        this.isLoaded = false;
        this.loader = new MJCFLoader();
    }

    async load(xmlPath) {
        return new Promise((resolve) => {
            this.loader.load(xmlPath, (group) => {
                // Attach to class instance group
                this.group.add(group);

                // Setup Kinematic Structures
                this._setupKinematics(group);

                // Setup Base Transforms - TEST2 DEFAULT
                this.group.rotation.x = -Math.PI / 2; // Z-up fix

                const angle35 = (35 * Math.PI) / 180;
                if (this.mirrored) {
                    this.group.rotation.y = Math.PI + angle35;
                } else {
                    this.group.rotation.y = angle35;
                }

                this.group.position.set(this.xOffset, 0, 0);
                this.group.scale.set(5, 5, 5); // Test2 scale (consistent with 1.0 positions)

                this.isLoaded = true;
                resolve();
            });
        });
    }

    _setupKinematics(robotGroup) {
        const headSites = new Map();
        const rodEndSites = new Map();
        const rodBodies = new Map();

        // 1. Traverse and Map Components
        robotGroup.traverse(c => {
            // Map BODY Joints
            if (c.userData && c.userData.jointName) {
                this.joints[c.userData.jointName] = c;
                if (!c.userData.origQuat) c.userData.origQuat = c.quaternion.clone();
            }

            // Map Sites
            if (c.userData.type === 'site') {
                if (c.name.startsWith('closing_')) {
                    const parts = c.name.split('_');
                    const rodId = parseInt(parts[1]);
                    const siteId = parseInt(parts[2]);

                    if (!isNaN(rodId)) {
                        if (siteId === 2) headSites.set(rodId, c);
                        if (siteId === 1) rodEndSites.set(rodId, c);
                    }
                }
            }

            // Map Rod Bodies
            const lowerName = c.name.toLowerCase();
            if (lowerName.includes('rod') || (lowerName.includes('stewart') && !lowerName.includes('ball') && !lowerName.includes('horn'))) {
                const parts = c.name.split('_');
                const lastPart = parts[parts.length - 1];
                let id = parseInt(lastPart);
                if (isNaN(id) && lowerName.includes('rod')) id = 1;

                if (!isNaN(id)) {
                    rodBodies.set(id, c);
                }
            }
        });

        // 2. Setup Rod Connections (Visual Parenting)
        robotGroup.userData.rods = [];

        const head = robotGroup.getObjectByName('xl_330');
        if (head) {
            // Head base rotation fix for MuJoCo XML alignment
            head.rotation.set(0, 0, Math.PI);
            this.headMesh = head;
            this.headPivot = head;
        }

        // 3. Link Rods to Head Sites
        for (let i = 1; i <= 6; i++) {
            const rodBody = rodBodies.get(i);
            const headSocket = headSites.get(i);
            const endSite = rodEndSites.get(i);

            if (rodBody && (headSocket || i === 6)) {
                let target = headSocket;
                if (i === 6 && head) target = head;

                if (target) {
                    robotGroup.userData.rods.push({
                        id: i,
                        rodBody: rodBody,
                        headSocket: target,
                        lengthVector: endSite ? endSite.position.clone() : new THREE.Vector3(0.085, 0, 0)
                    });
                }
            }
        }
    }

    applyPose(poseFrame) {
        if (!this.isLoaded || !poseFrame) return;

        // Pre-allocate math objects for efficiency
        const _q1 = new THREE.Quaternion();
        const _e1 = new THREE.Euler();
        const _zAxis = new THREE.Vector3(0, 0, 1);

        const pose = {
            x: poseFrame.pos[0],
            y: this.mirrored ? -poseFrame.pos[1] : poseFrame.pos[1],
            z: poseFrame.pos[2],
            roll: this.mirrored ? -poseFrame.rot[0] : poseFrame.rot[0],
            pitch: poseFrame.rot[1],
            yaw: this.mirrored ? -poseFrame.rot[2] : poseFrame.rot[2]
        };

        const bodyYaw = this.mirrored ? -poseFrame.hip : poseFrame.hip;
        const headMat = buildHeadPoseMatrix(pose);
        const activeAngles = calculateActiveMotorAngles(headMat, bodyYaw);

        // Apply Active Joints
        const activeNames = ['yaw_body', 'stewart_1', 'stewart_2', 'stewart_3', 'stewart_4', 'stewart_5', 'stewart_6'];
        activeNames.forEach((name, i) => {
            const joint = this.joints[name];
            if (joint) {
                _q1.setFromAxisAngle(_zAxis, activeAngles[i]);
                joint.quaternion.copy(joint.userData.origQuat).multiply(_q1);
            }
        });

        // Apply Passive Joints
        const passiveAngles = calculatePassiveJoints(activeAngles, headMat);
        for (let i = 1; i <= 7; i++) {
            const name = `passive_${i}`;
            const joint = this.joints[name];
            if (joint) {
                const idx = (i - 1) * 3;
                _e1.set(passiveAngles[idx], passiveAngles[idx + 1], passiveAngles[idx + 2], 'XYZ');
                _q1.setFromEuler(_e1);
                joint.quaternion.copy(joint.userData.origQuat).multiply(_q1);
            }
        }

        // Apply Antennas
        if (poseFrame.ant) {
            const lAnt = this.joints['left_antenna'];
            const rAnt = this.joints['right_antenna'];
            const rVal = this.mirrored ? poseFrame.ant[1] : poseFrame.ant[0];
            const lVal = this.mirrored ? poseFrame.ant[0] : poseFrame.ant[1];

            if (rAnt) {
                _q1.setFromAxisAngle(_zAxis, -rVal);
                rAnt.quaternion.copy(rAnt.userData.origQuat).multiply(_q1);
            }
            if (lAnt) {
                _q1.setFromAxisAngle(_zAxis, -lVal);
                lAnt.quaternion.copy(lAnt.userData.origQuat).multiply(_q1);
            }
        }
    }
}
