import yaml
import numpy as np
import pytest

from nprba.robot_arms.franka_panda import RobotArmModel, orientation_error
from nprba.utils.kinematics_jax import rotx, roty, rotz
from nprba.utils.kinematics_numpy import Rp2Trans, Trans2Rp


N_TESTS = 50
Q_0 = np.array([[0., -np.pi/4, 0., -3*np.pi/4, 0., np.pi/2, np.pi/4]]).T
R_bee_0, p_bee_0 = RobotArmModel().fk_ee(Q_0)
T_bee_0 = Rp2Trans(R_bee_0, p_bee_0.ravel())

with open('configs/franka_limits.yaml', 'r') as file:
    PANDA_LIMITS = yaml.safe_load(file)
DQ_MIN = PANDA_LIMITS['dq_min']
DQ_MAX = PANDA_LIMITS['dq_max']
DDQ_MIN = PANDA_LIMITS['ddq_min']
DDQ_MAX = PANDA_LIMITS['ddq_max']


def test_ik_trans():
    r = RobotArmModel()

    # Sanity test: give an initial guess the real solution
    q_init_guess = Q_0.copy()
    q_ik = r.ik(T_bee_0, q_init_guess)
    np.testing.assert_array_almost_equal(q_init_guess, q_ik)

    for _ in range(N_TESTS):
        delta_p = np.random.uniform(-0.25, 0.25, size=(3,1))
        T_rnd = T_bee_0.copy()
        T_rnd[:3,[3]] += delta_p
        R_rnd, p_rnd = Trans2Rp(T_rnd)

        q_ik = r.ik(T_rnd, q_init_guess.copy())
        R_ik, p_ik = r.fk_ee(q_ik)
        e_o = orientation_error(R_rnd, R_ik)
        np.testing.assert_almost_equal(e_o, np.zeros((3,1)), decimal=3)
        np.testing.assert_almost_equal(p_rnd, p_ik, decimal=3)


def test_ik_rot():
    r = RobotArmModel()

    q_init_guess = Q_0.copy()
    for _ in range(N_TESTS):
        delta_abg = np.random.uniform(-np.pi/18, np.pi/18, size=(3,1))
        delta_R = rotx(delta_abg[0,0]) @ roty(delta_abg[1,0]) @ rotz(delta_abg[2,0])
        T_rnd = T_bee_0.copy()
        T_rnd[:3,:3] = delta_R @ T_rnd[:3,:3]
        R_rnd, p_rnd = Trans2Rp(T_rnd)

        q_ik = r.ik(T_rnd, q_init_guess.copy())
        R_ik, p_ik = r.fk_ee(q_ik)
        e_o = orientation_error(R_rnd, R_ik)
        np.testing.assert_almost_equal(e_o, np.zeros((3,1)), decimal=3)
        np.testing.assert_almost_equal(p_rnd, p_ik, decimal=3)


def test_velocity_ik():
    """ 
    NOTE I am comparing ee velocities instead of joint velocities
    because the robot is redundant and the solution of the 
    Jacobain inverse is not unique, it chooses the solution where
    the joint velocities are smallest
    """
    r = RobotArmModel()

    q = Q_0.copy()
    for _ in range(N_TESTS):
        dq_rnd = [np.random.uniform(l, h) for l, h in zip(DQ_MIN, DQ_MAX)]
        dq_rnd = np.array(dq_rnd)[:, np.newaxis]

        v = r.ee_velocity(q, dq_rnd)
        dq_ik = r.velocity_ik(q, v)
        v_ik = r.ee_velocity(q, dq_ik)
        np.testing.assert_almost_equal(v, v_ik)


def test_acceleration_ik():
    r = RobotArmModel()

    q = Q_0.copy()
    for _ in range(N_TESTS):
        dq_rnd = [np.random.uniform(l, h) for l, h in zip(DQ_MIN, DQ_MAX)]
        dq_rnd = np.array(dq_rnd)[:, np.newaxis]

        ddq_rnd = [np.random.uniform(l, h) for l, h in zip(DDQ_MIN, DDQ_MAX)]
        ddq_rnd = np.array(ddq_rnd)[:, np.newaxis]

        a = r.ee_acceleration(q, dq_rnd, ddq_rnd)
        ddq_ik = r.acceleration_ik(q, dq_rnd, a)
        a_ik = r.ee_acceleration(q, dq_rnd, ddq_ik)
        np.testing.assert_almost_equal(a, a_ik)