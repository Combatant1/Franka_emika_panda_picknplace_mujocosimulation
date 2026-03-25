"""
Simple IK Solver for Franka Emika Panda using MuJoCo's built-in functions
Pure Python implementation - no C++ compilation required
"""

import numpy as np
import mujoco as mj


class PandaSimpleIK:
    """Simple IK solver using Jacobian pseudoinverse method"""
    
    def __init__(self, model_path, ee_site_name="gripper"):
        """
        Initialize IK solver
        
        Args:
            model_path: Path to MuJoCo XML model
            ee_site_name: Name of end-effector site in the model
        """
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # Find end-effector site
        self.ee_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0:
            raise ValueError(f"Site '{ee_site_name}' not found in model")
        
        # Joint limits (first 7 DOF for Panda arm)
        self.q_min = self.model.jnt_range[:7, 0]
        self.q_max = self.model.jnt_range[:7, 1]
        
    def solve_ik(self, target_pos, target_quat=None, q_init=None, 
                 max_iter=100, tol=1e-3, alpha=0.5, regularization=1e-4, fixed_q=None):
        """
        Solve IK using Jacobian pseudoinverse with damped least squares
        
        Args:
            target_pos: Target position (3D array)
            target_quat: Target quaternion [w, x, y, z] (optional)
            q_init: Initial joint configuration (7D array), uses current if None
            max_iter: Maximum iterations
            tol: Convergence tolerance (meters)
            alpha: Step size for gradient descent
            regularization: Damping factor for pseudoinverse
            fixed_q: Dictionary {joint_index: value} of joints to keep fixed
            
        Returns:
            dict with 'q': solution joint angles, 'error': final error, 
                     'success': bool, 'iterations': int
        """
        # Initialize joint angles
        if q_init is not None:
            self.data.qpos[:7] = q_init.copy()
        
        # Apply fixed joints to initial state
        if fixed_q is not None:
            for idx, val in fixed_q.items():
                if 0 <= idx < 7:
                    self.data.qpos[idx] = val
        
        target_pos = np.array(target_pos)
        
        for iteration in range(max_iter):
            # Forward kinematics
            mj.mj_forward(self.model, self.data)
            
            # Get current end-effector pose
            current_pos = self.data.site_xpos[self.ee_site_id].copy()
            
            # Position error
            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)
            
            # Check convergence
            if error_norm < tol:
                return {
                    'q': self.data.qpos[:7].copy(),
                    'error': error_norm,
                    'success': True,
                    'iterations': iteration + 1
                }
            
            # Compute Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mj.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            
            # Use only first 7 DOF (arm joints)
            jac = jacp[:, :7].copy()
            
            # If any joints are fixed, zero out their columns in the Jacobian
            mask = np.ones(7)
            if fixed_q is not None:
                for idx in fixed_q.keys():
                    if 0 <= idx < 7:
                        jac[:, idx] = 0
                        mask[idx] = 0
            
            # Damped least squares
            jac_t = jac.T
            dq = jac_t @ np.linalg.solve(jac @ jac_t + regularization * np.eye(3), pos_error)
            
            # Update with step size and mask
            self.data.qpos[:7] += alpha * dq * mask
            
            # Enforce joint limits
            self.data.qpos[:7] = np.clip(self.data.qpos[:7], self.q_min, self.q_max)
            
            # Ensure fixed joints stay fixed 
            if fixed_q is not None:
                for idx, val in fixed_q.items():
                    if 0 <= idx < 7:
                        self.data.qpos[idx] = val
        
        # Did not converge
        return {
            'q': self.data.qpos[:7].copy(),
            'error': error_norm,
            'success': False,
            'iterations': max_iter
        }
    
    def get_fk(self, q):
        """
        Compute forward kinematics
        
        Args:
            q: Joint angles (7D array)
            
        Returns:
            tuple of (position, quaternion)
        """
        self.data.qpos[:7] = q
        mj.mj_forward(self.model, self.data)
        
        pos = self.data.site_xpos[self.ee_site_id].copy()
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
        
        return pos, quat


if __name__ == "__main__":
    # Simple test
    print("Testing Simple IK Solver for Franka Panda")
    print("=" * 50)
    
    # Initialize solver
    solver = PandaSimpleIK("panda.xml")
    
    # Test case: reach to a point
    target_pos = np.array([0.5, 0.0, 0.3])
    q_init = np.zeros(7)  # Start from zero configuration
    
    print(f"\nTarget position: {target_pos}")
    print(f"Initial joints: {q_init}")
    
    # Solve IK
    result = solver.solve_ik(target_pos, q_init=q_init)
    
    print(f"\nResults:")
    print(f"  Success: {result['success']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final error: {result['error']*1000:.2f} mm")
    print(f"  Joint angles: {np.round(result['q'], 3)}")
    
    # Verify with FK
    final_pos, final_quat = solver.get_fk(result['q'])
    print(f"  Achieved position: {np.round(final_pos, 4)}")
    print(f"  Position error: {np.linalg.norm(final_pos - target_pos)*1000:.2f} mm")
