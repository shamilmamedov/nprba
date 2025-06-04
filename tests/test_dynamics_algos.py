import numpy as np
import pinocchio as pin
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from nprba.prba.algorithms_jax import (
    rnea_lie_jax,
    mass_matrix_rnea_jax,
    aba_lie_jax)
from nprba.prba.models import JointType
import nprba.prba.models as models
import nprba.utils.prba as prba


# seed for python random number generator
np.random.seed(42)

N_TESTS = 50
N_SEGS = [2, 4, 6]
P_markers = [0.81, 1.60, 2.42]
BASE_JOINT_TYPES = [JointType.U_ZY, JointType.P_XYZ, JointType.FREE]


def load_dlo_params():
    dlo_params = {
        'length': 2.42,
        'diameter_inner': 0.004,
        'diameter_outer': 0.006,
        'density': 2710,
        'youngs_modulus': 7.0e+10,
        'shear_modulus': 2.7e+10,
        'normal_damping_coef': 1.0e+9,
        'tangential_damping_coef': 1.0e+9
    }
    return prba.DLOParameters(**dlo_params)



def test_rnea_serial_chain():
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)

            # Pin model
            model_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            data_pin = model_pin.createData()

            # Get custom spherical pendulum model descriptio
            model_own = models.create_prba_custom_model(prba_params)

            # Random values for positions, velocities and accelerations
            nq = model_pin.nq
            o_nqx1 = np.zeros((nq,1))
            Q_rnd = np.pi/2*np.random.uniform(-1,1, size=(N_TESTS, nq, 1))
            DQ_rnd = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, nq, 1))
            DDQ_rnd = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, nq, 1))

            for k, (q_rnd, dq_rnd, ddq_rnd) in enumerate(zip(Q_rnd, DQ_rnd, DDQ_rnd)):
                # Test gravity vector
                g_pin = pin.rnea(model_pin, data_pin, q_rnd, o_nqx1, o_nqx1)
                g_own_jax = rnea_lie_jax(model_own, q_rnd, o_nqx1, o_nqx1).ravel()
                np.testing.assert_array_almost_equal(g_pin, g_own_jax, decimal=4)

                # Test whole nonlinear inverse dynamics
                tau_pin = pin.rnea(model_pin, data_pin, q_rnd, dq_rnd, ddq_rnd)
                tau_own_jax = rnea_lie_jax(model_own, q_rnd, dq_rnd, ddq_rnd).ravel()
                np.testing.assert_array_almost_equal(tau_pin, tau_own_jax, decimal=4)

                # Test building of the inertia matrix
                M_own_jax = mass_matrix_rnea_jax(model_own, q_rnd)
                M_pin = pin.crba(model_pin, data_pin, q_rnd)
                np.testing.assert_array_almost_equal(M_own_jax, M_pin, decimal=5)


def test_rnea_prba():
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)
            prba_params.compute_jforce_callbacks_from_dlo_params()

            # Pin model
            model_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            data_pin = model_pin.createData()

            # Get custom model description
            model_own = models.create_prba_custom_model(prba_params)
            cum_nqs = np.cumsum(model_own.jnqs)[:-1]

            # Random values for positions, velocities and accelerations
            nq = model_pin.nq
            Q_rnd = np.pi/2*np.random.uniform(-1,1, size=(N_TESTS, nq, 1))
            DQ_rnd = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, nq, 1))
            DDQ_rnd = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, nq, 1))

            for k, (q_rnd, dq_rnd, ddq_rnd) in enumerate(zip(Q_rnd, DQ_rnd, DDQ_rnd)):
                q_j = np.split(q_rnd, cum_nqs)
                dq_j = np.split(dq_rnd, cum_nqs)
                tau_sdes = np.vstack([clbk(x, dx) for clbk, x, dx in 
                                      zip(model_own.jforcecallbacks, q_j, dq_j)])
                
                # Test whole nonlinear inverse dynamics
                tau_pin = pin.rnea(model_pin, data_pin, q_rnd, dq_rnd, ddq_rnd)[:,None]
                tau_own_jax = rnea_lie_jax(model_own, q_rnd, dq_rnd, ddq_rnd)
                np.testing.assert_array_almost_equal(tau_pin + tau_sdes, tau_own_jax, decimal=4)


# @pytest.mark.skip(reason="Doesn't work for more than two joints")
def test_aba_serial_chain():
    """
    Tests "aba_LieAlgebra" - implementing forward dynamics
    of an N-link chain of rigid bodies - by comparing the dynamics to
    an analytical model derived in Pinocchio
    """
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)

            # Create pinocchio model
            model_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            data_pin = model_pin.createData()

            # Get custom model description
            model_own = models.create_prba_custom_model(prba_params)

            # Random values for positions, velocities and accelerations
            nq = model_pin.nq
            o_nqx1 = np.zeros((nq,1))
            Q_rnd = np.pi/2*np.random.uniform(-1,1, size=(N_TESTS, nq, 1))
            DQ_rnd = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, nq, 1))
            for k, (q_rnd, dq_rnd) in enumerate(zip(Q_rnd, DQ_rnd)):
                accel_pin = pin.aba(model_pin, data_pin, q_rnd, dq_rnd, o_nqx1)
                accel_aba_jax = aba_lie_jax(model_own, q_rnd, dq_rnd, simulation_mode='forward').ravel()

                np.testing.assert_allclose(accel_pin, accel_aba_jax)


def test_aba_prba():
    """
    Tests "aba_LieAlgebra" - implementing forward dynamics
    of an N-link chain of rigid bodies - by comparing the dynamics to
    an analytical model derived in Pinocchio
    """
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)
            prba_params.compute_jforce_callbacks_from_dlo_params()

            # Create pinocchio model
            model_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            data_pin = model_pin.createData()

            # Get custom model description
            model_own = models.create_prba_custom_model(prba_params)
            cum_nqs = np.cumsum(model_own.jnqs)[:-1]

            # Random values for positions, velocities and accelerations
            nq = model_pin.nq
            Q_rnd = np.pi/2*np.random.uniform(-1,1, size=(N_TESTS, nq, 1))
            DQ_rnd = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, nq, 1))
            for k, (q_rnd, dq_rnd) in enumerate(zip(Q_rnd, DQ_rnd)):
                q_j = np.split(q_rnd, cum_nqs)
                dq_j = np.split(dq_rnd, cum_nqs)
                tau_sdes = np.vstack([clbk(x, dx) for clbk, x, dx in 
                                      zip(model_own.jforcecallbacks, q_j, dq_j)])

                accel_pin = pin.aba(model_pin, data_pin, q_rnd, dq_rnd, -tau_sdes)
                accel_aba_jax = aba_lie_jax(model_own, q_rnd, dq_rnd, simulation_mode='forward').ravel()
                np.testing.assert_allclose(accel_pin, accel_aba_jax)


def test_hybrid_dynamics_serial_chain():
    """ Another way of testing would be assigning torques and comuputing
    acceleratioans with forward dynamics. Then take the accelerations of 
    the base and torques of the passive joints, and reconstruct torques
    in the base joint and compute accelerations in the other joints. 
    Obviously the results should be identical to what was provided to
    the forward dynamics algorithm in the beginning
    """
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)

            # Create pin model
            m_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            d_pin = m_pin.createData()

            # Create custom model 
            m_custom = models.create_prba_custom_model(prba_params)
            nq_bjoint = m_custom.jnqs[0]

            # Random values for positions, velocities and accelerations
            nq = m_pin.nq
            Q_rnd = np.pi*np.random.uniform(-1,1, size=(N_TESTS, nq, 1))
            DQ_rnd = 2*np.pi*np.random.uniform(-1, 1, size=(N_TESTS, nq, 1))
            TAU_base_rnd = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, nq_bjoint, 1))
            TAU = np.concatenate((TAU_base_rnd, np.zeros((N_TESTS, nq-nq_bjoint, 1))), axis=1)

            for k, (q_rnd, dq_rnd, tau_rnd) in enumerate(zip(Q_rnd, DQ_rnd, TAU)):
                ddq = pin.aba(m_pin, d_pin, q_rnd, dq_rnd, tau_rnd)[:,None]
                ddq_base = ddq[:nq_bjoint,:]

                ddq_aba_jax = aba_lie_jax(m_custom, q_rnd, dq_rnd, ddq_base, simulation_mode='hybrid')
                np.testing.assert_allclose(ddq_aba_jax[nq_bjoint:], ddq[nq_bjoint:], rtol=1e-5, atol=1e-2)
        

def test_hybrid_dynamics_prba():
    """
    Test hybrid dynamics for the RFEM model in Pinocchio and JAX.

    The difference from the serial_chain test in the presence of passive elastic torques
    """
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)
            prba_params.compute_jforce_callbacks_from_dlo_params()

            # Create pin model
            m_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            d_pin = m_pin.createData()

            # Create custom model 
            m_custom = models.create_prba_custom_model(prba_params)
            cum_nqs = np.cumsum(m_custom.jnqs)[:-1]
            nq_bjoint = m_custom.jnqs[0]

            # Random values for positions, velocities and accelerations
            nq = m_pin.nq
            Q_rnd = np.pi*np.random.uniform(-jnp.pi, jnp.pi, size=(N_TESTS, nq, 1))
            DQ_rnd = 2*np.pi*np.random.uniform(-jnp.pi, jnp.pi, size=(N_TESTS, nq, 1))
            TAU_base_rnd = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, nq_bjoint, 1))
            TAU = np.concatenate((TAU_base_rnd, np.zeros((N_TESTS, nq-nq_bjoint, 1))), axis=1)

            for k, (q_rnd, dq_rnd, tau_rnd) in enumerate(zip(Q_rnd, DQ_rnd, TAU)):
                q_j = np.split(q_rnd, cum_nqs)
                dq_j = np.split(dq_rnd, cum_nqs)
                tau_sdes = np.vstack([clbk(x, dx) for clbk, x, dx in 
                                      zip(m_custom.jforcecallbacks, q_j, dq_j)])
                
                tauk = tau_rnd - tau_sdes
                ddq = pin.aba(m_pin, d_pin, q_rnd, dq_rnd, tauk)[:,None]
                ddq_base = ddq[:nq_bjoint,:]

                ddq_aba_jax = aba_lie_jax(m_custom, q_rnd, dq_rnd, ddq_base, simulation_mode='hybrid')
                np.testing.assert_allclose(ddq_aba_jax[nq_bjoint:], ddq[nq_bjoint:], rtol=1e-5, atol=1e-2)


if __name__ == "__main__":
    pass