import numpy as np
from numpy import cos, sin
from typing import Tuple
import casadi as cs

from nprba.utils.joints import JointType, JOINTTYPE_TO_NQ


def rotz(x):
    sx = sin(x)
    cx = cos(x)
    return np.array(
        [[cx, -sx, 0.],
        [sx, cx, 0.],
        [0., 0., 1.]]
    )


def roty(x):
    sx = sin(x)
    cx = cos(x)
    return np.array(
        [[cx, 0., sx],
        [0., 1., 0.],
        [-sx, 0., cx]]
    )

                    
def rotx(x):
    sx = sin(x)
    cx = cos(x)
    return np.array(
        [[1., 0., 0.,],
        [0., cx, -sx],
        [0., sx, cx]]
    )    


def rpy(phi):
    """
    phi = [phi_x phi_y phi_z].T

    :param phi: is 3x1 vector of rpy angles
    """
    return rotz(phi[2,0]) @ roty(phi[1,0]) @ rotx(phi[0,0])


def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation

    :param omg: A 3-vector
    :return: The skew symmetric representation of omg
    """
    return np.array([[0.,      -omg[2,0],  omg[1,0]],
                     [omg[2,0],       0., -omg[0,0]],
                     [-omg[1,0], omg[0,0],       0.]])


def inertia_vec2mat(I_vec: np.ndarray):
    """ Converts inertia matrix components stored as
    vector into a matrix

    I_vec = [Ixx Iyy Izz Ixy Ixz Iyz] = [6x1]

    I = [Ixx Ixy Ixz
         Ixy Iyy Iyz
         Ixz Iyz Izz]
    """
    return np.array([[I_vec[0,0], I_vec[3,0], I_vec[4,0]],
                      [I_vec[3,0], I_vec[1,0], I_vec[5,0]],
                      [I_vec[4,0], I_vec[5,0], I_vec[2,0]]])


def inertia_at_joint(R_ab, p_ba, m, I_b):
    """
    Compute the inertia in a body-fixed joint frame 'a'
    while inertia is defined in a body-fixed frame 'b'.
    (R_ab, p_ba) expresses the frame 'b' in the frame 'a'
    See "kim2012lie", Page 5
    :param m: body mass
    :param I: 3x3 inertia matrix
    :param R_ab: Rotation matrix
    :param p_ba: origin of 'b' w.r.t. 'a'
    :return: Spatial inertia matrix in 'a'
    """
    p = VecToso3(p_ba)
    return np.r_[np.c_[R_ab @ I_b @ R_ab.T + m * p.T @ p, m * p],
                 np.c_[m * p.T, m * np.eye(3)]]


def diag_inertia_matrix_to_central_second_moments(j):
    """
    j = [jxx, jyy, jzz]
    """
    jxx, jyy, jzz = j[0,0], j[1,0], j[2,0]
    L1 = 0.5*(-jxx + jyy + jzz)
    L2 = 0.5*(jxx - jyy + jzz)
    L3 = 0.5*(jxx + jyy - jzz)
    return np.array([[L1, L2, L3]]).T


def central_second_moments_to_diag_inertia_matrix(L):
    jxx = L[1,0] + L[2,0]
    jyy = L[0,0] + L[2,0]
    jzz = L[0,0] + L[1,0]
    return np.diag(np.array([jxx, jyy, jzz]))


def Rp2Trans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix
    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs
    """
    if isinstance(p, cs.SX):
        return cs.vertcat(
            cs.horzcat(R, p), 
            cs.horzcat(cs.SX.zeros(1, 3), 1.)
        )
    return np.r_[np.c_[R, p], [[0., 0., 0., 1.]]]


def Trans2Rp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector
    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.
    """
    return T[0:3, 0:3], T[0:3, [3]]


def TransInv(T):
    """Inverts a homogeneous transformation matrix

    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    """
    R, p = Trans2Rp(T)
    Rt = R.T
    if isinstance(T, cs.SX):
        return cs.vertcat(
            cs.horzcat(Rt, -Rt @ p),
            cs.horzcat(cs.SX.zeros(1, 3), 1.)
        )
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0., 0., 0., 1.]]]


def ad(V):
    """Calculate the 6x6 matrix [adV] of the given 6-vector

    :param V: A 6-vector spatial velocity
    :return: The corresponding 6x6 matrix [adV]

    Used to calculate the Lie bracket [V1, V2] = [adV1]V2
    """
    omgmat = VecToso3(V[:3,:])
    if isinstance(V, cs.SX):
        return cs.vertcat(
            cs.horzcat(omgmat, cs.SX.zeros(3, 3)),
            cs.horzcat(VecToso3(V[3:,:]), omgmat)
        )
    return np.r_[np.c_[omgmat, np.zeros((3, 3))],
                 np.c_[VecToso3(V[3:,:]), omgmat]]


def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    """
    R, p = Trans2Rp(T)
    if isinstance(T, cs.SX):
        return cs.vertcat(
            cs.horzcat(R, cs.SX.zeros(3, 3)),
            cs.horzcat(cs.mtimes(VecToso3(p), R), R)
        )
    return np.r_[np.c_[R, np.zeros((3, 3))],
                  np.c_[np.dot(VecToso3(p), R), R]]


def jcalc(type: JointType, q, dq):
    jtype_jcalc = {
        JointType.U_ZY: universal_ZY_joint,
        JointType.P_XYZ: P_XYZ_joint,
        JointType.FREE: free_joint,
    }

    if type in jtype_jcalc:
        return jtype_jcalc[type](q, dq)
    else:
        raise NotImplementedError


def universal_ZY_joint(
    q: np.ndarray, 
    dq: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Perform basic joint computations

    :param q: joint configuration
    :param dq: joint velocities [optional]

    :return: Homogenous transformation matrix R, 
             joint mapping matrix (joint Jacobian) S 
             and its derivatuive
    """
    assert q.shape == (2,1)

    R = rotz(q[0,0]) @ roty(q[1,0])
    p = np.zeros((3,1))
    T = Rp2Trans(R, p)

    S = np.array(
        [[-np.sin(q[1,0]), 0.], 
        [0., 1.], 
        [np.cos(q[1,0]), 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]]
    )
    # You don't always need velocity related term
    if dq is None:
        dS = None
    else:
        dS = np.array([[-np.cos(q[1,0])*dq[1,0], 0.], 
                        [0., 0.], 
                        [-np.sin(q[1,0])*dq[1,0], 0.],
                        [0., 0.],
                        [0., 0.],
                        [0., 0.]])
    return T, S, dS


def P_XYZ_joint(
    q: np.ndarray,
    dq: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param q: joint configuration
    :param dq: joint velocities [optional]

    :return: Homogenous transformation matrix R, 
             joint mapping matrix (joint Jacobian) S 
             and its derivatuive
    """
    assert q.shape == (3,1)

    R = np.eye(3)
    T = Rp2Trans(R, q)

    S = np.vstack((np.zeros((3,3)), np.eye(3)))
    if dq is None:
        dS = None
    else:
        dS = np.zeros((6,3))
    return T, S, dS


def free_joint(
        q: np.ndarray,
        dq: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param q: joint configuration (expressed in parent frame)
    :param dq: joint velocities (expressed in body frame)

    We chose the Pinocchio convention for the free joint, that is
    q = [global_base_position, global_base_orientation]
    :return: Homogenous transformation matrix R,
             joint mapping matrix (joint Jacobian) S
             and its derivative
    """
    assert q.shape == (6,1)

    q1, q2, q3 = q[3,0], q[4,0], q[5,0]
    
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s3, c3 = np.sin(q3), np.cos(q3)
    R = np.array([[c1*c2, -s1*c3 + s2*s3*c1, s1*s3 + s2*c1*c3],
                   [s1*c2, s1*s2*s3 + c1*c3, s1*s2*c3 - s3*c1],
                   [-s2, s3*c2, c2*c3]])
    T = Rp2Trans(R, q[:3, 0])

    S_rot = np.array([[-s2, 0., 1.], [s3*c2, c3, 0.], [c2*c3, -s3, 0.]])
    S = np.r_[np.c_[np.zeros((3,3)), S_rot],
               np.c_[R.T, np.zeros((3,3))]]


    if dq is None:
        dS = None
    else:
        dq1, dq2, dq3 = dq[3,0], dq[4,0], dq[5,0]
        dS_rot = np.array([
            [-c2*dq2, 0, 0], 
            [-s2*s3*dq2 + c2*c3*dq3, -s3*dq3, 0], 
            [-s2*c3*dq2 - s3*c2*dq3, -c3*dq3, 0]
        ])
        dS_trans = np.array([
            [-s1*c2*dq1 - s2*c1*dq2, -s1*s2*dq2 + c1*c2*dq1, -c2*dq2],
            [-s1*s2*s3*dq1 + s1*s3*dq3 + s2*c1*c3*dq3 + s3*c1*c2*dq2 - c1*c3*dq1, 
                s1*s2*c3*dq3 + s1*s3*c2*dq2 - s1*c3*dq1 + s2*s3*c1*dq1 - s3*c1*dq3, 
                -s2*s3*dq2 + c2*c3*dq3
            ],
            [-s1*s2*c3*dq1 + s1*c3*dq3 - s2*s3*c1*dq3 + s3*c1*dq1 + c1*c2*c3*dq2, 
                -s1*s2*s3*dq3 + s1*s3*dq1 + s1*c2*c3*dq2 + s2*c1*c3*dq1 - c1*c3*dq3,
                -s2*c3*dq2 - s3*c2*dq3
            ]
        ])

        dS = np.r_[np.c_[np.zeros((3,3)), dS_rot],
                    np.c_[dS_trans, np.zeros((3,3))]]
    return T, S, dS


def euler_zyx_derivative_to_angular_velocity(phi_e):
    phi, theta, psi = phi_e[:,0]
    c_phi = cos(phi)
    s_phi = sin(phi)
    c_theta = cos(theta)
    s_theta = sin(theta)
    return np.array([[0., -s_phi, c_phi*c_theta],
                     [0., c_phi, s_phi*c_theta],
                     [1., 0., -s_theta]])


def derivative_of_euler_zyx_derivative_to_angular_velocity(phi_e, dphi_e):
    phi, theta, psi = phi_e[:,0]
    dphi, dtheta, dpsi = dphi_e[:,0]
    c_phi = cos(phi)
    s_phi = sin(phi)
    c_theta = cos(theta)
    s_theta = sin(theta)
    return np.array(
        [[0., -c_phi*dphi, -s_phi*c_theta*dphi - c_phi*s_theta*dtheta],
         [0., -s_phi*dphi, c_phi*c_theta*dphi - s_phi*s_theta*dtheta],
         [0., 0., -c_theta*dtheta]]
    )


def compute_angular_velocity_from_euler_zyx_derivative(phi_e, dphi_e):
    return euler_zyx_derivative_to_angular_velocity(phi_e) @ dphi_e


def compute_S():
    """ Computes the joint Jacobian aka geometric body-fixed Jacobian S
    and its derivative dS for joints using Euler rotation matrices """
    import sympy as sp
    sp.init_printing(use_unicode=True)

    joint_type = 'free'
    t  = sp.symbols('t')
    q1 = sp.Function('q1')
    q2 = sp.Function('q2')
    q3 = sp.Function('q3')
    s1 = sp.sin(q1(t))
    c1 = sp.cos(q1(t))
    s2 = sp.sin(q2(t))
    c2 = sp.cos(q2(t))
    s3 = sp.sin(q3(t))
    c3 = sp.cos(q3(t))

    Rx = sp.Matrix([[1, 0., 0.],
                    [0., c3, -s3],
                    [0., s3, c3]])

    Ry = sp.Matrix([[c2, 0., s2],
                    [0., 1., 0.],
                    [-s2, 0., c2]])

    Rz = sp.Matrix([[c1, -s1, 0.],
                    [s1, c1, 0.],
                    [0., 0., 1.]])

    if joint_type == '2D':
        R = Rz * Ry
    elif joint_type == 'free':
        R = Rz * Ry * Rx

    dR = sp.diff(R, t)
    print(R)
    #print(dR.shape)

    def so3toVec(R):
        """
        Converts an so(3) matrix to vec
        Source: Modern Robotics - https://github.com/NxRLab/ModernRobotics
        """
        return sp.Matrix([R[2, 1], R[0, 2], R[1, 0]])

    RTdR = R.T * dR
    RTdR_vec = sp.simplify(so3toVec(RTdR))
    print('test', RTdR_vec)
    if joint_type == '2D':
        col1 = RTdR_vec.subs([(sp.Derivative(q1(t), t), 1), (sp.Derivative(q2(t), t), 0)])
        col2 = RTdR_vec.subs([(sp.Derivative(q1(t), t), 0), (sp.Derivative(q2(t), t), 1)])
        S = col1.row_join(col2)
    elif joint_type == 'free':
        col1 = RTdR_vec.subs([(sp.Derivative(q1(t), t), 1), (sp.Derivative(q2(t), t), 0), (sp.Derivative(q3(t), t), 0)])
        col2 = RTdR_vec.subs([(sp.Derivative(q1(t), t), 0), (sp.Derivative(q2(t), t), 1), (sp.Derivative(q3(t), t), 0)])
        col3 = RTdR_vec.subs([(sp.Derivative(q1(t), t), 0), (sp.Derivative(q2(t), t), 0), (sp.Derivative(q3(t), t), 1)])
        S = col1.row_join(col2).row_join(col3)

    dS = sp.simplify(sp.diff(S, t))
    dR_T = sp.simplify(sp.simplify(sp.diff(R.T, t)))

    def get_string_from_sym(sym_exp):
        str_exp = str(sym_exp)
        replace_list = [['q1(t)', 'q1'], ['q2(t)', 'q2'], ['q3(t)', 'q3'],
                        ['cos(q1)', 'c1'], ['sin(q1)', 's1'],
                        ['cos(q2)', 'c2'], ['sin(q2)', 's2'],
                        ['cos(q3)', 'c3'], ['sin(q3)', 's3'],
                        ['cos', 'jnp.cos'], ['sin', 'jnp.sin'],
                        ['1.0*', ''], ['Derivative(q1, t)', 'dq1'],
                        ['Derivative(q2, t)', 'dq2'], ['Derivative(q3, t)', 'dq3']
                        ]
        for i in range(len(replace_list)):
            str_exp = str_exp.replace(replace_list[i][0], replace_list[i][1])
        return str_exp

    R_str = get_string_from_sym(R)
    RT_str = get_string_from_sym(R.T)
    dRT_str = get_string_from_sym(dR_T)
    S_str = get_string_from_sym(S)
    dS_str = get_string_from_sym(dS)
    print('R:', R_str)
    print('R.T:', RT_str)
    print('dR_T:', dRT_str)
    print('S:', S_str)
    print('dS:', dS_str)


def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


if __name__ == "__main__":
    compute_S()
    print("DEBUGGG")

