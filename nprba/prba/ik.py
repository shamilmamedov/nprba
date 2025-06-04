import numpy as np
import casadi as cs
import pinocchio as pin

from nprba.utils.kinematics_numpy import free_joint

qpoases_opts = {'printLevel': 'none'}
qrqp_opts = {'print_header': False, 'print_iter': False, 'print_info': False}
solver_opts = {'qp_oases': qpoases_opts, 'qrqp': qrqp_opts}


class IKSolver:
    def __init__(self, model, data, e_marker_id, max_iter=500, alpha=1, eps_err=1e-4) -> None:
        """
        :param model: pinocchio model
        :param data: pinocchio data
        :param e_marker_id: id of the end-point marker
        """
        self.model = model
        self.data = data
        self.e_marker_id = e_marker_id

        self.max_iter = max_iter
        self.eps_err = eps_err # tolerance of FK error
        self.eps_delta_q = 1e-5 # tolerance on the change of q
        self.alpha = alpha # gain of the CLIK
        self.W_v = np.eye(3) # weights on the end-point erro ||J dq - v||_W

        self.qp_solver = 'qrqp'

        #TODO pass the position of the marker in b frame as input
        self.p_b_emarker = 1.92

    
    def position_ik(self, pd, qb, W_dq: np.ndarray = None, W_dq_diff: np.ndarray = None, mu: float = 1e-5, 
                    verbose: bool = False, init_with_prev_sol: bool = False                
    ):
        """ Solves position-level IK either for a single configuration or
        for the whole trajectory 

        :param pd: end-point position
        :param qb: configuration of the base joint
        :param q0: initial guess for the solution
        :param verbose: weather to print stats

        :return: inverse kinematics solution
        """
        if pd.shape[1] == 1 and qb.shape[1] == 1:
            return self._position_ik(pd, qb, verbose=verbose)
        else:
            ns = pd.shape[0]
            q_ik = np.zeros((ns, self.model.nq))
            for k, (pd_k, qb_k) in enumerate(zip(pd, qb)):
                if init_with_prev_sol and k > 0:
                    q_ik[[k],:] = self._position_ik(
                        pd_k[:,None], qb_k[:,None], q0=q_ik[[k-1],6:].T, W_dq=W_dq, mu=mu, verbose=verbose
                    ).T
                else:
                    q_ik[[k],:] = self._position_ik(
                        pd_k[:,None], qb_k[:,None], W_dq=W_dq, W_dq_diff=W_dq_diff, mu=mu, verbose=verbose
                    ).T
        return q_ik
    
    def _position_ik(self, pd, qb, q0: np.ndarray = None, 
                     W_dq: np.ndarray = None, W_dq_diff: np.ndarray = None,
                     mu: float = 1e-5, verbose: bool = False):
        """ Solves position-level inverse kinematics for a single measurement

        :param pd: end-point position
        :param qb: configuration of the base joint
        :param q0: initial guess for the solution
        :param verbose: weather to print stats

        :return: inverse kinematics solution
        """
        if q0 is None:
            q0 = np.zeros((self.model.nq - qb.size, 1))
            # q0 = self.initial_guess(pd, qb)
        q_cur = np.vstack((qb, q0))

        if W_dq is None:
            W_dq = self.joint_velocity_weighting_matrix_for_position_ik(self.model.nq - qb.size)

        error_norm = 100.
        dist_cur_prev = 100.
        iter = 0
        while dist_cur_prev > self.eps_delta_q and error_norm > self.eps_err and iter < self.max_iter:
            p = compute_marker_position(self.model, self.data, q_cur, self.e_marker_id)
            error = self.position_error(p, pd)
            Jp = compute_marker_jacobian(self.model, self.data, q_cur, self.e_marker_id)[:,6:]
            dq_rfem_cur = self._velocity_ik_unconstrained(Jp, error, W_dq, W_dq_diff, mu)

            q_prev = q_cur.copy() 
            q_cur[6:,:] += self.alpha*dq_rfem_cur

            dist_cur_prev = np.linalg.norm(q_cur - q_prev)
            iter += 1
            error_norm = np.linalg.norm(error)

        if verbose:
            print(f'Number of iterations: {iter}')
            print(f'|fk(q) - pd| = {error_norm}')
            print(f'|q_cur - q_prev| = {dist_cur_prev}')

        return q_cur

    def velocity_ik_constrained(self, v, dqb, q, W_dq=None, W_dq_diff=None):
        if v.shape[1] == 1 and dqb.shape[1] == 1:
            v, dqb, q = v.T, dqb.T, q.T

        ns = v.shape[0]
        dq = np.zeros((ns, self.model.nq))
        for k, (vk, dqbk, qk) in enumerate(zip(v, dqb, q)):
            Jk = compute_marker_jacobian(self.model, self.data, qk, self.e_marker_id)
            vbe_k = vk[:,None] - Jk[:,:6] @ dqbk[:,None]
            dq_refem_k = self._velocity_ik_constrained(Jk[:,6:], vbe_k, W_dq, W_dq_diff)
            dq[[k],:] = np.vstack((dqbk[:,None], dq_refem_k)).T
        
        if dq.shape[0] == 1:
            return dq.T
        else:
            return dq

    def velocity_ik_unconstrained(self, v, dqb, q, W_dq=None, W_dq_diff=None, mu: float = 1e-5):
        if v.shape[1] == 1 and dqb.shape[1] == 1:
            v, dqb, q = v.T, dqb.T, q.T

        ns = v.shape[0]
        dq = np.zeros((ns, self.model.nq))
        for k, (vk, dqbk, qk) in enumerate(zip(v, dqb, q)):
            Jk = compute_marker_jacobian(self.model, self.data, qk, self.e_marker_id)
            vbe_k = vk[:,None] - Jk[:,:6] @ dqbk[:,None]
            dq_refem_k = self._velocity_ik_unconstrained(Jk[:,6:], vbe_k, W_dq, W_dq_diff, mu)
            dq[[k],:] = np.vstack((dqbk[:,None], dq_refem_k)).T
        
        # if dq.shape[0] == 1:
        #     return dq.T
        # else:
        return dq

    def _velocity_ik_unconstrained(self, J, v, W_dq = None, W_dq_diff: np.ndarray = None, mu: float = 1e-5):
        # Design a weigting matrix
        if W_dq is None:
            nq = J.shape[1]
            W_dq = self.joint_velocity_weighting_matrix(nq)
        
        if W_dq_diff is not None:
            W_dq = W_dq + W_dq_diff

        qp = dict()
        Jp_W = self.W_v @ J
        error_W = self.W_v @ v
        H_qp = cs.DM(Jp_W.T @ Jp_W + mu*W_dq)
        g_qp = cs.DM(-error_W.T @ Jp_W)
        qp['h'] = H_qp.sparsity()
        QP_solver = cs.conic('S', self.qp_solver, qp,  solver_opts[self.qp_solver])
        x_opt = QP_solver(h=H_qp, g=g_qp)['x']
        # np.linalg.solve(H_qp, g_qp)
        return np.array(x_opt)

    def _velocity_ik_constrained(self, J, v, W_dq=None, W_dq_diff=None):
        # Design a weigting matrix
        nq = J.shape[1]
        step_size = 0.7
        if W_dq is None:
            W_dq = self.joint_velocity_weighting_matrix(nq, step_size)
        if W_dq_diff is None:
            W_dq_diff = 0.3*self.joint_velocity_difference_weighting_matrix(nq, step_size)
        W_dq = W_dq + W_dq_diff

        dq0 = np.zeros((nq,1))

        qp = dict()
        H_qp = cs.DM(W_dq)
        g_qp = cs.DM(-dq0.T)
        A_qp = cs.DM(J)
        lba = v
        uba = v
        
        qp['h'] = H_qp.sparsity()
        qp['a'] = A_qp.sparsity()
        QP_solver = cs.conic('S', self.qp_solver, qp,  solver_opts[self.qp_solver])
        x_opt = QP_solver(h=H_qp, g=g_qp, a=A_qp, lba=lba, uba=uba)['x']
        return np.array(x_opt)
    
    @staticmethod
    def position_error(p, pd):
        return pd - p

    @staticmethod
    def joint_velocity_weighting_matrix(nq, step_size=0.6, reversed=False):
        n_seg = nq // 2
        w_dq = np.tile(np.linspace(1, step_size*(n_seg+1), n_seg), (2,1)).T.reshape(-1)
        if reversed:
            w_dq = np.flip(w_dq)
        W_dq = np.diag(w_dq)
        return W_dq
    
    def joint_velocity_weighting_matrix_for_position_ik(self, nq):
        n_seg = nq // 2
        w_dq_z = np.concatenate((
            np.array([1., 1.]),
            1. + 0.5*np.arange(0, n_seg-2)
        ))
        w_dq = np.tile(w_dq_z[:n_seg], (2,1)).T.reshape(-1)
        W_dq = np.diag(w_dq)
        return W_dq
    
    @staticmethod
    def joint_velocity_difference_weighting_matrix(nq, step_size=0.6):
        n_seg = nq // 2
        w_dqz = np.linspace(1, step_size*(n_seg+1), n_seg)
        w_dq = np.tile(w_dqz, (2,1)).T.reshape(-1)
        W_dq_diff = np.diag(w_dq)
        np.fill_diagonal(W_dq_diff[:,2:], -w_dq[1:-1])
        np.fill_diagonal(W_dq_diff[2:,:], -w_dq[1:-1])
        return W_dq_diff

    def initial_guess(self, pd, qb):
        nq_rfem = self.model.nq - qb.size 
        n_seg = nq_rfem // 2

        # Get pose of the base joint
        Tb_k = free_joint(qb)[0]

        # Find the position of the end DLO assuming no deformation
        p_0seg = (Tb_k @ np.array([[self.p_b_emarker, 0., 0., 1.]]).T)[:3,:]

        # Get the deformation of the end
        delta_y = float((p_0seg - pd)[2])
        delta_z = float((p_0seg - pd)[1])

        # DLO length except the first segment
        l_ = self.p_b_emarker*(1 - 1/(2*n_seg))

        # Find angles assuming one segment of legnth l_
        alpha_y = np.arctan2(delta_y, l_)
        alpha_z = np.arctan2(delta_z, l_)

        # Divide angles by the number of segments
        delta_alpha_y = alpha_y/n_seg
        delta_alpha_z = alpha_z/n_seg
        return np.array([[-2*delta_alpha_z, 2*delta_alpha_y]*n_seg]).T


def compute_marker_position(model, data, q, frame_id):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    T_e_marker = data.oMf[frame_id]
    p = T_e_marker.translation[:,np.newaxis]
    return p


def compute_marker_jacobian(model, data, q, frame_id):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J = pin.getFrameJacobian(model, data, 
                frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return J[:3,:]


