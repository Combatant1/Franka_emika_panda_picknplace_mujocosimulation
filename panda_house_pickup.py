#!/usr/bin/env python3
"""
IMPROVED: Franka Panda Robot - House Interior Cup Pickup Task
Key improvements:
1. Tighter gripper closure for secure grasp
2. Slower, more controlled lifting
3. Better wait times for grasp stabilization
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from simple_ik_solver import PandaSimpleIK

class PandaHousePickup:
    def __init__(self, model_path="panda_house_v2.xml"):
        """Initialize the Panda house pickup controller"""
        print("Loading MuJoCo model...")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Get important body and site IDs
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'gripper')
        self.cup_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cup')
        
        if self.gripper_site_id < 0:
            raise ValueError("Site 'gripper' not found in model")
        if self.cup_body_id < 0:
            raise ValueError("Body 'cup' not found in model")
            
        print(f"  - Body/Site IDs: gripper={self.gripper_site_id}, cup={self.cup_body_id}")
        
        # Set initial state to home keyframe
        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key_id)
            print("  - Initial state set to 'home' keyframe")
        else:
            print("  - Warning: 'home' keyframe not found")
        
        # Control parameters - IMPROVED GRIPPER SETTINGS
        self.dt = self.model.opt.timestep
        self.gripper_closed = 0.02  # Medium Squeeze (Command 2cm width)
        self.gripper_open = 0.08
        
        # Realistic velocity limits for Panda robot (rad/s)
        self.max_joint_velocities = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        self.velocity_scale = 0.3
        
        # Initialize IK solver
        try:
            self.ik_solver = PandaSimpleIK(model_path, ee_site_name="gripper")
            print(f"  - IK Solver: Initialized successfully")
        except Exception as e:
            print(f"  - IK Solver: Error - {e}")
            raise
        
        # Gripper configuration
        self.gripper_idx = [7, 8]
        self.gripper_open = 20.0
        self.gripper_closed = -20.0  # Gentle force to avoid arm deflection
        
        print(f"Model loaded successfully!")
        
    def get_gripper_position(self):
        """Get current gripper (end-effector) position"""
        return self.data.site_xpos[self.gripper_site_id].copy()
    
    def get_cup_position(self):
        """Get current cup CENTER position (xpos is at center of mass)"""
        return self.data.xpos[self.cup_body_id].copy()
    
    def get_cup_velocity(self):
        """Get current cup velocity"""
        cup_qvel_addr = self.model.body_dofadr[self.cup_body_id]
        return np.linalg.norm(self.data.qvel[cup_qvel_addr:cup_qvel_addr+3])
    
    def set_joint_targets(self, joint_positions):
        """Set target positions for arm joints"""
        self.data.ctrl[:7] = joint_positions[:7]
    
    def get_gripper_width(self):
        """Return scalar width of the gripper fingers"""
        # Assuming qpos[7] and qpos[8] are the finger joint positions
        return self.data.qpos[7] + self.data.qpos[8]
    
    def detect_cube_contact(self):
        """Detect if gripper or hand is in contact with the cube"""
        # Check all active contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name
            
            # Check if cube is involved in contact
            if 'cube_geom' in geom1 or 'cube_geom' in geom2:
                # Check if gripper/hand geoms are involved
                gripper_geoms = ['left_finger_pad', 'right_finger_pad', 'hand_capsule']
                for g_geom in gripper_geoms:
                    if g_geom in geom1 or g_geom in geom2:
                        return True, contact.dist  # Return contact depth
        return False, 0.0
    
    def get_contact_force(self):
        """Get total contact force on gripper fingers"""
        total_force = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name
            
            # Check for finger-cube contact
            finger_geoms = ['left_finger_pad', 'right_finger_pad']
            if 'cube_geom' in geom1 or 'cube_geom' in geom2:
                for fg in finger_geoms:
                    if fg in geom1 or fg in geom2:
                        # Get contact force (simplified - using frame data)
                        force_idx = i * 6  # Each contact has 6D wrench
                        if force_idx < len(self.data.efc_force):
                            total_force += abs(self.data.efc_force[force_idx])
        return total_force

    def set_gripper(self, force_command, viewer=None, duration=0.5):
        """
        Apply Force to gripper. 
        force_command: Positive to Open, Negative to Close/Grip.
        """
        self.data.ctrl[self.gripper_idx[0]] = force_command
        self.data.ctrl[self.gripper_idx[1]] = force_command
        
        # Wait for physics to apply force
        t_start = self.data.time
        while self.data.time - t_start < duration:
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(self.dt) # Use self.dt for consistent simulation steps
        
        width = self.get_gripper_width()
        print(f"    [Diagnostic] Gripper Width: {width*1000:.1f}mm (Force: {force_command})")
        return True
    
    def wait_until_stable(self, viewer=None, duration=1.0, max_vel=0.01):
        """Wait until robot and objects are stable"""
        start_time = self.data.time
        stable_count = 0
        required_stable = int(0.5 / self.dt)
        
        while self.data.time - start_time < duration:
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(self.dt)
            
            joint_vel = np.linalg.norm(self.data.qvel[:7])
            cup_vel = self.get_cup_velocity()
            
            if joint_vel < max_vel and cup_vel < max_vel:
                stable_count += 1
                if stable_count >= required_stable:
                    return True
            else:
                stable_count = 0
                
        return stable_count >= required_stable
    
    def move_to_joint_position(self, target_joints, viewer=None, duration=1.5, tolerance=0.01):
        """Move robot to target joint configuration with smooth interpolation"""
        target_joints = np.array(target_joints)
        start_joints = self.data.qpos[:7].copy()
        start_time = self.data.time
        
        while self.data.time - start_time < duration:
            # Cosine interpolation for smoothness
            t = (self.data.time - start_time) / duration
            s = 0.5 * (1.0 - np.cos(np.pi * t))
            
            interp_joints = start_joints + s * (target_joints - start_joints)
            self.set_joint_targets(interp_joints)
            
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(self.dt)
            
            current_joints = self.data.qpos[:7]
            error = np.linalg.norm(current_joints - target_joints)
            if t > 0.95 and error < tolerance:
                break
                
        # Final snap to target to ensure exactness
        self.set_joint_targets(target_joints)
        return True
    
    def inverse_kinematics(self, target_pos, current_joints=None, max_iterations=50, fixed_q=None):
        """Compute IK using the simple IK solver"""
        if current_joints is None:
            current_joints = self.data.qpos[:7].copy()
        
        result = self.ik_solver.solve_ik(
            target_pos, 
            q_init=current_joints,
            max_iter=max_iterations,
            tol=1e-3,
            alpha=0.5,
            regularization=1e-4,
            fixed_q=fixed_q
        )
        
        return result['q']
    
    def get_joint_positions(self):
        return self.data.qpos[:7] # First 7 joints

    def move_to_position(self, target_pos, viewer=None, duration=1.5, tolerance=0.005, fixed_wrist=-0.785, verbose=False):
        """Move gripper to target position with smooth path smoothing"""
        start_time = self.data.time
        target_pos = np.array(target_pos).copy()
        start_pos = self.get_gripper_position()
        
        if target_pos[2] < 0.2:
            target_pos[2] = 0.2
            
        while self.data.time - start_time < duration:
            # Smooth interpolation factor
            t = (self.data.time - start_time) / duration
            s = 0.5 * (1.0 - np.cos(np.pi * t))
            
            # Smoothly move the target position
            current_target = start_pos + s * (target_pos - start_pos)
            
            fixed_q_dict = {6: fixed_wrist} if fixed_wrist is not None else None
            current_joints = self.data.qpos[:7].copy()
            target_joints = self.inverse_kinematics(current_target, current_joints, fixed_q=fixed_q_dict)
            
            self.set_joint_targets(target_joints)
            mujoco.mj_step(self.model, self.data)
            
            if viewer:
                viewer.sync()
                time.sleep(self.dt)
            
            current_pos = self.get_gripper_position()
            error = np.linalg.norm(current_pos - target_pos)
            
            if error < tolerance:
                if verbose:
                    print(f"    [OK] Converged! Error: {error*1000:.2f}mm")
                return True
        
        if verbose:
            current_pos = self.get_gripper_position()
            error = np.linalg.norm(current_pos - target_pos)
            print(f"    [!] Timeout! Final Error: {error*1000:.2f}mm")
        return False
    
    def execute_pickup_sequence(self, viewer=None):
        """Execute the pick, lift, and place back sequence using hardcoded waypoints"""
        print("\n============================================================")
        print("STARTING IMPROVED PICK-AND-PLACE TASK")
        print("============================================================")
        
        # Hardcoded Joint Waypoints
        WAYPOINTS = {
            "home":      [0.0, -0.1973, 0.0, -1.8624, 0.0, 1.2256, 0.785],
            "high":      [-0.0, -0.3983, -0.0, -1.4049, -0.0, 1.4343, 0.785],
            "above_cup": [0.0000, 0.5583, 0.0000, -0.6685, 0.0000, 1.4306, 0.7850], # Vertical Approach
            "grasp_cup": [0.0000, 0.3425, 0.0000, -1.2838, 0.0000, 1.8300, 0.7850], # Z=0.50 (Aligned with cube mid-height)
            "lift_cup":  [0.0000, 0.5583, 0.0000, -0.6685, 0.0000, 1.4306, 0.7850], # Lift Straight Up
            "swing_right": [-1.047, -0.3983, -0.0, -1.4049, -0.0, 1.4343, 0.785]
        }
        
        # Sync physics
        mujoco.mj_forward(self.model, self.data)
        
        # Step 1: Move to home
        print("Step 1: Moving to home position...")
        self.set_gripper(self.gripper_open, viewer)
        self.move_to_joint_position(WAYPOINTS["home"], viewer, duration=1.5)
        print("  [OK] Reached home\n")
        
        # Step 1.5: Move to HIGH (Up) to avoid obstacles
        print("Step 1.5: Moving UP to high waypoint...")
        self.move_to_joint_position(WAYPOINTS["high"], viewer, duration=1.5)
        print("  [OK] Reached high point\n")
        
        # Step 2: Move above cup
        print("Step 2: Moving above cup...")
        self.move_to_joint_position(WAYPOINTS["above_cup"], viewer, duration=1.5)
        print("  [OK] Positioned above cup\n")
        
        # Step 3: VISUAL TRACKING DESCENT - Stop just above cube top
        print("Step 3: Descending to cube (visual tracking)...")
        
        # Get cube dimensions and position
        cube_pos = self.get_cup_position()
        cube_center_z = cube_pos[2]
        cube_half_height = 0.04  # 8cm cube / 2
        cube_top_z = cube_center_z + cube_half_height
        target_ee_z = cube_top_z + 0.02  # Stop 2cm above cube top
        
        print(f"    Cube center Z: {cube_center_z:.3f}m")
        print(f"    Cube top Z: {cube_top_z:.3f}m")
        print(f"    Target EE Z: {target_ee_z:.3f}m")
        
        # Descend slowly while monitoring EE height
        target_joints = np.array(WAYPOINTS["grasp_cup"])
        start_joints = self.data.qpos[:7].copy()
        descent_duration = 4.0
        descent_start = self.data.time
        
        stopped_early = False
        while self.data.time - descent_start < descent_duration:
            t = (self.data.time - descent_start) / descent_duration
            s = 0.5 * (1.0 - np.cos(np.pi * t))
            interp_joints = start_joints + s * (target_joints - start_joints)
            self.set_joint_targets(interp_joints)
            
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(self.dt)
            
            # Check EE height
            current_ee_z = self.get_gripper_position()[2]
            if current_ee_z <= target_ee_z:
                stopped_early = True
                print(f"    [OK] Reached cube at EE Z={current_ee_z:.3f}m")
                break
        
        if not stopped_early:
            final_ee_z = self.get_gripper_position()[2]
            print(f"    [WARNING] Descent timeout at EE Z={final_ee_z:.3f}m")
        
        # Step 4: TIME-BASED GRIP - Close for sufficient duration
        print("  Closing gripper...")
        self.data.ctrl[self.gripper_idx[0]] = self.gripper_closed
        self.data.ctrl[self.gripper_idx[1]] = self.gripper_closed
        
        grip_duration = 1.5
        grip_start = self.data.time
        
        while self.data.time - grip_start < grip_duration:
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(self.dt)
        
        width = self.get_gripper_width()
        print(f"    [OK] Gripper closed, width: {width*1000:.1f}mm\n")
        
        # Step 5: RELATIVE LIFT - Lift UPWARD from current position
        print("Step 5: Lifting cube (relative upward movement)...")
        
        # Get current end-effector position
        current_ee_pos = self.get_gripper_position().copy()
        print(f"    Current EE Z: {current_ee_pos[2]:.3f}m")
        
        # Target: 15cm directly upward
        target_ee_pos = current_ee_pos.copy()
        target_ee_pos[2] += 0.15  # Lift 15cm up
        
        print(f"    Target EE Z: {target_ee_pos[2]:.3f}m (+150mm)")
        
        # Solve IK for the upward position
        try:
            current_joints = self.data.qpos[:7].copy()
            fixed_q_dict = {6: 0.785}  # Keep wrist orientation
            target_joints = self.inverse_kinematics(target_ee_pos, current_joints, fixed_q=fixed_q_dict)
            
            # Execute smooth lift
            self.move_to_joint_position(target_joints, viewer, duration=2.5)
            
            final_ee_z = self.get_gripper_position()[2]
            print(f"    Final EE Z: {final_ee_z:.3f}m")
        except Exception as e:
            print(f"    [ERROR] IK failed for lift: {e}")
            print("    Falling back to absolute waypoint")
            self.move_to_joint_position(WAYPOINTS["lift_cup"], viewer, duration=2.5)
        
        # Wait and check if cube followed
        self.wait_until_stable(viewer, duration=1.0)
        cup_z = self.get_cup_position()[2]
        print(f"  Current cup center Z: {cup_z:.3f}m")
        
        if cup_z > 0.5:
            print(f"  [OK] SUCCESS! Cup lifted\n")
        else:
            print(f"  [X] FAILURE! Cup not lifted\n")
            return False
            
        # Step 4.1: Swing Right 60 degrees
        print("Step 4.1: Swinging 60 degrees RIGHT...")
        self.move_to_joint_position(WAYPOINTS["swing_right"], viewer, duration=2.0)  # Slower swing
        print("  [OK] Swung right\n")
        
        # Step 4.2: Swing back to lift position
        print("Step 4.2: Swinging BACK to center...")
        self.move_to_joint_position(WAYPOINTS["lift_cup"], viewer, duration=2.0)
        print("  [OK] Returned to center\n")
            
        # Step 5: Place it back - SLOWER
        print("Step 5: Placing cup back...")
        self.move_to_joint_position(WAYPOINTS["grasp_cup"], viewer, duration=2.0)
        
        print("  Releasing cup...")
        self.set_gripper(self.gripper_open)
        self.wait_until_stable(viewer, duration=1.5)
        print("  [OK] Cup released\n")
        
        # Step 5.5: Safe Retraction
        print("Step 5.5: Retracting UP to high waypoint...")
        self.move_to_joint_position(WAYPOINTS["high"], viewer, duration=1.5)
        print("  [OK] Reached safe height\n")
        
        # Step 6: Return to home
        print("Step 6: Retracting to home...")
        self.move_to_joint_position(WAYPOINTS["home"], viewer, duration=1.5)
        print("  [OK] Task completed successfully!\n")
        
        return True

def main():
    try:
        pickup = PandaHousePickup("panda_house_v2.xml")
        
        print("\nAttempting to launch MuJoCo viewer...")
        try:
            with mujoco.viewer.launch_passive(pickup.model, pickup.data) as viewer:
                print("[OK] Viewer launched successfully")
                # Set camera to match view_scene.py
                viewer.cam.azimuth = 45
                viewer.cam.elevation = -15
                viewer.cam.distance = 2.5
                viewer.cam.lookat = np.array([0.3, -0.3, 0.4])
                
                # Init viewer
                for _ in range(100):
                    mujoco.mj_step(pickup.model, pickup.data)
                    viewer.sync()
                
                time.sleep(1)
                
                # Execute with viewer
                success = pickup.execute_pickup_sequence(viewer)
                
                print("\nMaintaining state. Press Ctrl+C to exit.")
                while viewer.is_running():
                    mujoco.mj_step(pickup.model, pickup.data)
                    viewer.sync()
                    time.sleep(0.01)
        except Exception as viewer_err:
            print(f"[!] Could not launch viewer: {viewer_err}")
            print("Running in HEADLESS mode (logic only via console logs)...\n")
            
            # Execute without viewer
            success = pickup.execute_pickup_sequence(None)
            print(f"\nHeadless task completed. Success: {success}")
                
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()