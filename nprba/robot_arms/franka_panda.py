import pinocchio as pin
import numpy as np
import yaml
from scipy.linalg import block_diag

from nprba.utils.kinematics_numpy import Trans2Rp, VecToso3
from nprba.prba.visualization import visualize_robot


class RobotArmModel():
    def __init__(
        self, 
        path_to_urdf: str = None, 
        ee_frame_name: str = None, 
        calibration_file_path: str = None,
        T_f_b: np.ndarray = None
    ) -> None:
        # Create model using pinocchio
        if path_to_urdf is None:
            path_to_urdf = 'nprba/robot_arms/panda_arm.urdf'

        # Load the urdf model
        self.model, self.cmodel, self.vmodel = pin.buildModelsFromUrdf(path_to_urdf)

        if calibration_file_path and T_f_b:
            raise ValueError("Both calibration_file_path and T_f_b cannot be provided")

        # Adding a dlo-base frame as a child of the last joint
        if calibration_file_path is not None:
            with open(calibration_file_path, 'r') as file:
                calibr_data = yaml.safe_load(file)
            
            frame_name = 'dlo-base'
            frame_placement = pin.SE3(np.array(calibr_data['T_fb']))
            parent_joint_id = self.model.getJointId("panda_joint7")    
            prev_frame_id = self.model.getFrameId("panda_link7")
            frame = pin.Frame(frame_name, parent_joint_id, prev_frame_id, frame_placement, pin.FrameType.OP_FRAME)
            self.model.addFrame(frame)

            self.dlo_base_frame_id = self.model.getFrameId('dlo-base')
        
        if T_f_b is not None:
            frame_name = 'dlo-base'
            frame_placement = pin.SE3(T_f_b)
            parent_joint_id = self.model.getJointId("panda_joint7")    
            prev_frame_id = self.model.getFrameId("panda_link7")
            frame = pin.Frame(frame_name, parent_joint_id, prev_frame_id, frame_placement, pin.FrameType.OP_FRAME)
            self.model.addFrame(frame)

            self.dlo_base_frame_id = self.model.getFrameId('dlo-base')
        
        # Get EE frame id for forward kinematics
        if ee_frame_name is None:
            ee_frame_name = 'panda_link8'
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)

        # Create data required by the algorithms
        self.data = self.model.createData()

        # Useful variables
        self.nx = self.model.nq + self.model.nv
        self.nq = self.model.nq
        self.nu = self.model.nq

    def random_q(self):
        """ Returns a random configuration
        """
        return pin.randomConfiguration(self.model)

    def fk(self, q, frame_id):
        """ Computes forward kinematics for a given frame
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        # pin.updateFramePlacement(self.model, self.data, frame_id)
        T_EE_O = self.data.oMf[frame_id]
        R_EE_O = T_EE_O.rotation
        p_EE_O = T_EE_O.translation
        return R_EE_O, p_EE_O.reshape(-1,1)

    def fk_ee(self, q):
        """ Computes forward kinematics for EE frame in base frame
        """
        return self.fk(q, self.ee_frame_id)

    def fk_dlo_base(self, q):
        return self.fk(q, self.dlo_base_frame_id)

    def ik(self, Td: np.ndarray, q_guess: np.ndarray, verbose: bool = False):
        """
        :param T: homogenous transformation matrix for the end-effector
        """
        max_iter = 500
        eps = 1e-6
        alpha = 1e-1
        Kp = 10*np.eye(3)
        Ko = 10*np.eye(3)
        K = block_diag(Kp, Ko)

        Rd, pd = Trans2Rp(Td)
        q = q_guess
        lyap_fcn_value = 100.
        k = 0
        while lyap_fcn_value > eps and k < max_iter: 
            R, p = self.fk_ee(q)
            error_oreint = orientation_error(R, Rd)
            error_pos = position_error(p, pd)
            error = np.vstack((error_pos, error_oreint))
            J = self.jacobian_ee(q)
            L = omega_to_axisangle_vel_map(R, Rd)
            dq = np.linalg.lstsq(
                J,
                block_diag(Kp, np.linalg.inv(L) @ Ko) @ error,
                rcond=-1
            )[0]
            q += alpha*dq
            lyap_fcn_value = 0.5*error.T @ K @ error
            k += 1
            if verbose:
                print(f" |e_orient| = {np.linalg.norm(error_oreint):.4f}" +
                    f" |e_pos| = {np.linalg.norm(error_pos):.4f}")
        return q

    def velocity_ik(self, q: np.ndarray, v: np.ndarray):
        J = self.jacobian_ee(q)
        return np.linalg.lstsq(J, v, rcond=-1)[0]

    def acceleration_ik(self, q: np.ndarray, dq: np.ndarray, a: np.ndarray):
        J = self.jacobian_ee(q)
        dJ = self.djacobian_ee(q, dq)
        return np.linalg.lstsq(
            J, 
            a - dJ @ dq, 
            rcond=-1
        )[0]

    def frame_velocity(self, q, dq, frame_id):
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        v = pin.getFrameVelocity(self.model, self.data, 
                    frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        return np.hstack((v.linear, v.angular)).reshape(-1,1)

    def ee_velocity(self, q, dq):
        return self.frame_velocity(q, dq, self.ee_frame_id)

    def dlo_base_velocity(self, q, dq):
        return self.frame_velocity(q, dq, self.dlo_base_frame_id)

    def frame_acceleration(self, q, dq, ddq, frame_id):
        pin.forwardKinematics(self.model, self.data, q, dq, ddq)
        pin.updateFramePlacements(self.model, self.data)
        a = pin.getFrameAcceleration(self.model, self.data, 
                    frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        return np.hstack((a.linear, a.angular)).reshape(-1,1)

    def ee_acceleration(self, q, dq, ddq):
        return self.frame_acceleration(q, dq, ddq, self.ee_frame_id)

    def dlo_base_acceleration(self, q, dq, ddq):
        return self.frame_acceleration(q, dq, ddq, self.dlo_base_frame_id)

    def jacobian(self, q, frame_id):
        """ Computes Jacobian for a given frame
        """
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        Jk = pin.getFrameJacobian(self.model, self.data, 
                frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return Jk

    def jacobian_ee(self, q):
        """ Computes jacobian of the EE in base frame
        """
        return self.jacobian(q, self.ee_frame_id)

    def djacobian(self, q, dq, frame_id):
        """ Computes time derivative of the jacobian matrix
        """
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        dJk = pin.getFrameJacobianTimeVariation(self.model, self.data, 
                frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return dJk

    def djacobian_ee(self, q, dq):
        return self.djacobian(q, dq, self.ee_frame_id)

    def visualize_configuration(self, q, t=None):
        visualize_robot(
            np.tile(q.T, (2,1)), 3, 1, 
            self.model, self.cmodel, self.vmodel
        )
    
    def visualize_trajectory(self, q, dt):
        visualize_robot(
            q, dt, 3,
            self.model, self.cmodel, self.vmodel
        )


def omega_to_axisangle_vel_map(Re: np.ndarray, Rd: np.ndarray):
    nd, sd, ad = Rd.T
    ne, se, ae = Re.T
    L = -0.5*(
        VecToso3(nd.reshape(3,1)) @ VecToso3(ne.reshape(3,1)) + 
        VecToso3(sd.reshape(3,1)) @ VecToso3(se.reshape(3,1)) + 
        VecToso3(ad.reshape(3,1)) @ VecToso3(ae.reshape(3,1))
    )
    return L


def orientation_error(Re: np.ndarray, Rd: np.ndarray):
    nd, sd, ad = Rd.T
    ne, se, ae = Re.T
    error = 0.5*(
        np.cross(ne, nd) + 
        np.cross(se, sd) + 
        np.cross(ae, ad)
    )
    return error.reshape(-1,1)


def position_error(p: np.ndarray, pd: np.ndarray):
    return pd - p

    
if __name__ == "__main__":
    r = RobotArmModel()

    q_def = np.array([[0., -np.pi/4, 0., -3*np.pi/4 + np.pi/15, 0., 12*np.pi/18, np.pi/4]]).T

    r.visualize_configuration(q_def)

    R_def, p_def = r.fk_ee(q_def)
    T_def = Rp2Trans(R_def, p_def.ravel())
    
    q_guess = np.array([[0., 0., 0., -2.3658182, 0., np.pi/2, np.pi/4]]).T
    q_ik = r.ik(T_def, q_guess)
    print(np.array2string(q_def.T, precision=3))
    print(np.array2string(q_ik.T, precision=3))