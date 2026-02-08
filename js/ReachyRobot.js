import * as THREE from 'three';
import { MJCFLoader } from './MJCFLoader.js';
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

                // Setup Kinematic Structures (from index.html logic)
                this._setupKinematics(group);

                // Setup Base Transforms
                this.group.rotation.x = -Math.PI / 2; // Z-up to Y-up

                // Rotation Y: Angle 35 degrees towards center
                const angle35 = (35 * Math.PI) / 180;
                if (this.mirrored) {
                    this.group.rotation.y = Math.PI + angle35;
                } else {
                    this.group.rotation.y = angle35;
                }

                this.group.position.set(this.xOffset, 0, 0);
                this.group.scale.set(5, 5, 5); // Visible scale

                // Add to Scene (assuming scene is global or passed... wait, scene is not global here)
                // We should let the caller add 'this.group' to the scene, OR pass scene in constructor.
                // Let's assume global 'scene' exists in index.html, but here we can't access it unless passed.
                // I'll make index.html add it, but for compatibility with dance_duo.html logic which called '.load', 
                // dance_duo.html added it inside .load().
                // I'll dispatch an event or rely on main code to add it? 
                // dance_duo.html line 1341: scene.add(this.group). 
                // I will add a method 'addToScene(scene)' or just let the user access .group.
                // Better: index.html logic can do scene.add(robot.group).

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
            // Map BODY Joints using XML data
            if (c.userData && c.userData.jointName) {
                this.joints[c.userData.jointName] = c;
                if (!c.userData.origQuat) c.userData.origQuat = c.quaternion.clone();
            }

            // Map Sites (Closing Sites & Head Sites)
            if (c.userData.type === 'site') {
                if (c.name.startsWith('closing_')) {
                    const parts = c.name.split('_');
                    // Expected format: closing_ROD_END (e.g. closing_1_1, closing_1_2) 
                    // closing_X_1 = Rod End? closing_X_2 = Head Socket?
                    // XML spec: closing_X_2 are on the head (xl_330).
                    const rodId = parseInt(parts[1]);
                    const siteId = parseInt(parts[2]);

                    if (!isNaN(rodId)) {
                        if (siteId === 2) headSites.set(rodId, c);
                        if (siteId === 1) rodEndSites.set(rodId, c);
                    }
                }
            }

            // Map Rod Bodies (Heuristic name match)
            // Rods are usually named 'rod_1', 'rod_2' etc, or 'stewart_arm_X'
            // In MJCF parsing, names come from Body names.
            // Let's look for "rod" or numbers in the name
            const lowerName = c.name.toLowerCase();
            if (lowerName.includes('rod') || (lowerName.includes('stewart') && !lowerName.includes('ball') && !lowerName.includes('horn'))) {
                // Try to extract ID
                // Format might be 'rod_1' or just '1' if name is simple
                const parts = c.name.split('_');
                const lastPart = parts[parts.length - 1];
                let id = parseInt(lastPart);
                if (isNaN(id) && lowerName.includes('rod')) id = 1; // Default

                if (!isNaN(id)) {
                    rodBodies.set(id, c);
                    // Ensure silver material
                    c.traverse(child => {
                        if (child.isMesh) {
                            child.material.color.setHex(0xaaaaaa);
                            child.material.roughness = 0.2;
                            child.material.metalness = 1.0;
                        }
                    });
                }
            }
        });

        // 2. Setup Rod Connections (Visual Parenting)
        robotGroup.userData.rods = [];

        // Find Head (xl_330)
        const head = robotGroup.getObjectByName('xl_330');
        if (head) {
            // Create Pivot Group wrapper for Head
            const pivotGroup = new THREE.Group();
            pivotGroup.name = "HeadPivotGroup";

            // Parent the pivot to the head's current parent (rod_6 usually)
            if (head.parent) {
                const parent = head.parent;
                // Re-parent head into pivot
                // Save transformations
                const headPos = head.position.clone();
                const headQuat = head.quaternion.clone();

                // We want pivot at head location? Or pivot at rotation center?
                // Rod 6 is the neck. Head attaches to it.
                // We insert PivotGroup between Rod6 and Head for easier animation control.

                pivotGroup.position.copy(headPos);
                pivotGroup.quaternion.copy(headQuat);

                parent.add(pivotGroup);
                pivotGroup.add(head);

                // Zero out head local transform as it is now relative to pivot
                head.position.set(0, 0, 0);
                head.rotation.set(0, 0, 0);

                this.headPivot = pivotGroup;
                this.headMesh = head;

                // Save initial pose
                pivotGroup.userData.initialPosition = pivotGroup.position.clone();
                pivotGroup.userData.initialQuaternion = pivotGroup.quaternion.clone();
            }
        }

        // 3. Link Rods to Head Sites
        for (let i = 1; i <= 6; i++) {
            const rodBody = rodBodies.get(i);
            const headSocket = headSites.get(i);
            const endSite = rodEndSites.get(i);

            // Rod 6 is special (Spine), handled by hierarchy mostly
            // Rods 1-5 are LookAt

            if (rodBody && (headSocket || i === 6)) {
                let target = headSocket;
                if (i === 6 && head) target = head; // Rod 6 looks at / holds head

                if (target) {
                    // Store for Update Loop
                    // We calculate vector from Rod Origin to Target
                    robotGroup.userData.rods.push({
                        id: i,
                        rodBody: rodBody,
                        headSocket: target,
                        // If we found 'endSite' (tip of rod), we use it for offset calculation
                        lengthVector: endSite ? endSite.position.clone() : new THREE.Vector3(0.085, 0, 0)
                    });
                }
            }
        }
    }

    applyPose(frame) {
        if (!this.isLoaded || !frame || !this.headPivot) return;

        // Extract Frame Data
        // Frame pos/rot are usually in "Head Frame" or "Body Frame" depending on recorder.
        // Assuming Standard Reachy SDK format:
        // pos: [x, y, z], rot: [roll, pitch, yaw]

        // Handle Mirroring (Left vs Right)
        const mirror = this.mirrored;

        let tx = frame.pos[0];
        let ty = frame.pos[1];
        let tz = frame.pos[2];

        let roll = frame.rot[0];
        let pitch = frame.rot[1];
        let yaw = frame.rot[2];

        if (mirror) {
            // Mirror logic: 
            // Position: Y is Lateral usually. If X is Fwd.
            // Reachy coord: X=Fwd, Y=Left, Z=Up.
            // Mirror about XZ plane -> Invert Y.
            ty = -ty;

            // Rotation:
            // Scroll (Y-axis rotation) inverted?
            // Pitch (Y-axis) -> same
            // Roll (X-axis) -> inverted
            // Yaw (Z-axis) -> inverted
            roll = -roll;
            yaw = -yaw;
        }

        // 1. Apply to Head Pivot (Visual)
        // We add delta to initial position? Or absolute?
        // Assuming Absolute from SDK.

        // Map to Three.js Frame (Z-up vs Y-up conversion happens at Root)
        // Since Robot is -90 X, local axes align with World Y-up = Local Z-up.

        // Let's perform IK to correct rods
        // Create 4x4 Matrix for Head Pose
        const poseObj = { x: tx, y: ty, z: tz, roll, pitch, yaw };
        const headMat = buildHeadPoseMatrix(poseObj);

        // Use provided Hip Yaw or 0
        const hipYaw = frame.hip || 0;
        const actualHip = mirror ? -hipYaw : hipYaw;

        // Calculate Motors (IK)
        // Ensure inputs are valid
        // Note: SDK usually gives pose relative to body base.
        // calculateActiveMotorAngles expects 4x4 matrix row-major
        const activeAngles = calculateActiveMotorAngles(headMat, actualHip);

        // If IK fails (out of reach), activeAngles has 0s.

        // Apply Active Angles to Joints
        // Joints: yaw_body, stewart_1...6
        const jointNames = ['yaw_body', 'stewart_1', 'stewart_2', 'stewart_3', 'stewart_4', 'stewart_5', 'stewart_6'];

        activeAngles.forEach((angle, idx) => {
            const name = jointNames[idx];
            const jointBody = this.joints[name];
            if (jointBody) {
                // Rotate around Z axis (local joint axis)
                const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), angle);
                if (jointBody.userData.origQuat) {
                    jointBody.quaternion.copy(jointBody.userData.origQuat).multiply(q);
                }
            }
        });

        // Apply Passive Joints
        // Returns 21 numbers (7 sets of Euler XYZ)
        const passiveAngles = calculatePassiveJoints(activeAngles, headMat);

        // Map to passive_1...7
        // passive_1 to 6 are on the platform/rods
        // passive_7 is the head connection (Rod 6 -> Head)

        for (let i = 1; i <= 7; i++) {
            const name = `passive_${i}`; // Need to map this string to a Body?
            // MJCF usually names them 'passive_1' etc if defined.
            // If not found in this.joints, we skip.
            // NOTE: MJCFLoader logic needs to ensure 'passive_X' joints are found.
            // In Reachy Mini XML, passive joints usually exist.

            // If Kinematics.js calculates them, we should apply them.
            // If specific joint bodies aren't named 'passive_X', we need a map.
            // For now assuming 1:1 naming.
        }

        // --- HEAD MOVEMENT (Visual fallback if Passive Joints fail) ---
        // If we don't have full passive chain working, we manually move the head pivot
        // to match the frame position, ensuring the "Soul" is in the right place.

        // Note: Passive joint 7 drives the head orientation relative to Rod 6.
        // If that works, Head updates automatically.
        // If not, we force it:
        /*
        if (this.headPivot) {
            // Apply Transform locally
            // This is "cheating" the kinematics but ensures visual sync with music
            // We can blend IK with this.
        }
        */

        // Handle Antennas
        if (frame.ant) {
            const lAnt = this.joints['left_antenna'];
            const rAnt = this.joints['right_antenna'];
            let lVal = frame.ant[0]; // 0-1 or angle? Usually angle.
            let rVal = frame.ant[1];

            if (mirror) [lVal, rVal] = [rVal, lVal];

            if (lAnt) {
                const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -lVal); // Invert?
                lAnt.quaternion.copy(lAnt.userData.origQuat).multiply(q);
            }
            if (rAnt) {
                const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -rVal);
                rAnt.quaternion.copy(rAnt.userData.origQuat).multiply(q);
            }
        }
    }
}
