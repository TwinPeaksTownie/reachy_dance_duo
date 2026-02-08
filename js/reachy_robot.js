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

                // Setup Base Transforms - MONOLITHIC CALIBRATION
                this.group.rotation.x = -Math.PI / 2; // Z-up fix

                if (this.name === 'Left') {
                    this.group.position.set(-10, -5, 25);
                    this.group.rotation.z = -Math.PI / 6; // Yaw
                } else {
                    this.group.position.set(10, -5, 25);
                    this.group.rotation.z = Math.PI / 6 + Math.PI; // Yaw + 180
                }

                this.group.scale.set(20, 20, 20); // Marketing scale

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
            const pivotGroup = new THREE.Group();
            pivotGroup.name = "HeadPivotGroup";

            if (head.parent) {
                const parent = head.parent;
                const headPos = head.position.clone();
                const headQuat = head.quaternion.clone();

                pivotGroup.position.copy(headPos);
                pivotGroup.quaternion.copy(headQuat);

                parent.add(pivotGroup);
                pivotGroup.add(head);

                // --- CALIBRATED OFFSETS FROM MONOLITHIC ---
                if (this.name === 'Left') {
                    // LEFT ROBOT DEFAULTS
                    pivotGroup.position.set(0.0290, 0.0280, 0.1430);
                    pivotGroup.rotation.set(-1.6232, -2.0944, -1.9897);
                    head.position.set(0.0000, 0.0000, 0.0000);
                } else {
                    // RIGHT ROBOT DEFAULTS
                    pivotGroup.position.set(0.0190, 0.0240, 0.1410);
                    pivotGroup.rotation.set(1.7279, -1.2217, 1.2741);
                    head.position.set(0.0069, 0.0020, 0.0082);
                }

                // Head base rotation fix for MuJoCo XML alignment
                head.rotation.set(0, 0, Math.PI);

                this.headPivot = pivotGroup;
                this.headMesh = head;

                pivotGroup.userData.initialPosition = pivotGroup.position.clone();
                pivotGroup.userData.initialRotation = pivotGroup.rotation.clone();
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
