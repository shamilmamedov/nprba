import numpy as np
import pinocchio as pin
from jax.config import config
config.update("jax_enable_x64", True)


from nprba.prba.algorithms_jax import (
    fwd_kinematics,
    fwd_velocity_kinematics,
    compute_markers_velocities,
    compute_markers_positions,
    compute_markers_positions_and_velocities)
from nprba.utils.kinematics_jax import Trans2Rp
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


def test_fwd_kinematics():
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)

            # Pin model
            model_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            data_pin = model_pin.createData()

            # Get custom model description
            model_own = models.create_prba_custom_model(prba_params)

            # Configurations
            Q = np.random.uniform(-np.pi/2, np.pi/2, size=(N_TESTS, model_pin.nq, 1))
            for q in Q:
                pin.framesForwardKinematics(model_pin, data_pin, q)

                o_T_j, _ = fwd_kinematics(model_own, q)
                
                for j, Tj_own in enumerate(o_T_j):
                    Rj_pin = data_pin.oMi[j+1].rotation
                    pj_pin = data_pin.oMi[j+1].translation

                    Rj_own, pj_own = Trans2Rp(Tj_own)

                    np.testing.assert_array_almost_equal(Rj_pin, Rj_own, decimal=5)
                    np.testing.assert_array_almost_equal(pj_pin, pj_own.ravel(), decimal=5)


def test_compute_marker_positions():
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)

            # Pin model
            model_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            data_pin = model_pin.createData()

            # Get custom model description
            model_own = models.create_prba_custom_model(prba_params)

            # Configurations
            Q = np.random.uniform(-np.pi/2, np.pi/2, size=(N_TESTS, model_pin.nq, 1))
            for q in Q:
                pin.framesForwardKinematics(model_pin, data_pin, q)

                p_markers = compute_markers_positions(model_own, q)
                
                for m in range(3):
                    p_pin = data_pin.oMf[m+1].translation
                    p_own = p_markers[m]

                    np.testing.assert_array_almost_equal(p_pin, p_own, decimal=5)


def test_fwd_velocity_kinematics():
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)

            # Pin model
            model_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            data_pin = model_pin.createData()

            # Get custom model description
            model_own = models.create_prba_custom_model(prba_params)

            # Configurations
            Q = np.pi/2*np.random.uniform(-1, 1, size=(N_TESTS, model_pin.nq, 1))
            dQ = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, model_pin.nq, 1))
            for q, dq in zip(Q, dQ):
                pin.forwardKinematics(model_pin, data_pin, q, dq)
                pin.updateFramePlacements(model_pin, data_pin)

                Vj, _ = fwd_velocity_kinematics(model_own, q, dq)
                for j, vj_own in enumerate(Vj):
                    vj_pin = np.concatenate((
                        data_pin.v[j+1].angular, 
                        data_pin.v[j+1].linear
                    ))
                    
                    np.testing.assert_array_almost_equal(vj_pin, vj_own.ravel())


def test_compute_markers_velocities():
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)

            # Pin model
            model_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            data_pin = model_pin.createData()

            # Get custom model description
            model_own = models.create_prba_custom_model(prba_params)

            # Configurations
            Q = np.pi/2*np.random.uniform(-1, 1, size=(N_TESTS, model_pin.nq, 1))
            dQ = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, model_pin.nq, 1))
            for q, dq in zip(Q, dQ):
                pin.forwardKinematics(model_pin, data_pin, q, dq)
                pin.updateFramePlacements(model_pin, data_pin)

                dp_markers = compute_markers_velocities(model_own, q, dq)
                for m in range(3):
                    v_ = pin.getFrameVelocity(
                        model_pin, 
                        data_pin, 
                        model_pin.getFrameId(f'marker_{m+1}'), 
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                    )
                    dp_pin = v_.linear
                    np.testing.assert_array_almost_equal(dp_markers[m], dp_pin)


def test_compute_markers_positions_and_velocities():
    dlo_params = load_dlo_params()

    for bjoint in BASE_JOINT_TYPES:
        for n_seg in N_SEGS:
            prba_params = models.PRBAParameters(n_seg, dlo_params, bjoint, P_markers)

            # Pin model
            model_pin, _, _ = models.create_prba_pinocchio_model(prba_params)
            data_pin = model_pin.createData()

            # Get custom model description
            model_own = models.create_prba_custom_model(prba_params)

            # Configurations
            Q = np.pi/2*np.random.uniform(-1, 1, size=(N_TESTS, model_pin.nq, 1))
            dQ = np.pi*np.random.uniform(-1, 1, size=(N_TESTS, model_pin.nq, 1))
            for q, dq in zip(Q, dQ):
                pin.forwardKinematics(model_pin, data_pin, q, dq)
                pin.updateFramePlacements(model_pin, data_pin)

                p_m, dp_m = compute_markers_positions_and_velocities(model_own, q, dq)
                for m in range(3):
                    p_pin = data_pin.oMf[m+1].translation
                    v_pin = pin.getFrameVelocity(
                        model_pin, 
                        data_pin, 
                        model_pin.getFrameId(f'marker_{m+1}'), 
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                    )
                    dp_pin = v_pin.linear

                    np.testing.assert_array_almost_equal(p_pin, p_m[m], decimal=5)
                    np.testing.assert_array_almost_equal(dp_pin, dp_m[m], decimal=5)


if __name__ == '__main__':
    pass