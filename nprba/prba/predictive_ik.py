import casadi as cs
from pinocchio import casadi as cpin
import pinocchio as pin

from nprba.prba import models


def create_predictive_ik_solver(
    prba_params,
    N: int = 3,
    dt: float = 0.004,
    mu_q: float = 1e-1,
    mu_dq: float = 1e-2
):
    """ Create a predictive IK solver for the DLO with the given parameters
    
    :param prba_params: prba parameters that define the DLO
    :param N: number of look-ahead steps

    :return: predictive IK solver, lower and upper bounds for the decision variables
    """
    n_seg = prba_params.n_seg
    n_q = 2*n_seg

    # Create symbolic variables
    q_prba = cs.SX.sym('q_prba', n_q, N)
    dq_prba = cs.SX.sym('dq_prba', n_q, N)

    q_b = cs.SX.sym('q_b', 6, N)
    dq_b = cs.SX.sym('dq_b', 6, N)
    p_e = cs.SX.sym('p_e', 3, N)
    dp_e = cs.SX.sym('dp_e', 3, N)

    # Dynamics constraints via multiple shooting
    g = []
    for i in range(N-1):
        rhs = q_prba[:,i] + dt*dq_prba[:,i]
        g.append(q_prba[:,i+1] - rhs)

    g = cs.vertcat(*g)

    # Cost function
    cost = 0
    f_output = generate_output_casadi_fcn(prba_params)
    w_pe, w_dpe = 5., 1.
    W_y = cs.diag([w_pe, w_pe, w_pe, w_dpe, w_dpe, w_dpe])
    for i in range(N):
        q = cs.vertcat(q_b[:,i], q_prba[:,i])
        dq = cs.vertcat(dq_b[:,i], dq_prba[:,i])
        y_pred = f_output(q, dq)
        y_true = cs.vertcat(p_e[:,i], dp_e[:,i])
        y_error = y_pred - y_true
        if i == 0:
            pred_error_term = 2*y_error.T @ W_y @ y_error
        else:
            pred_error_term = y_error.T @ W_y @ y_error

        cost += (pred_error_term + 
                mu_dq*cs.sumsqr(dq_prba[:,i]) + 
                mu_q*cs.sumsqr(q_prba[:,i])
        )

    # Decision variables
    w = cs.vertcat(cs.vec(q_prba), cs.vec(dq_prba))
    lb_q_prba = -cs.pi/4*cs.DM.ones((n_q, N))
    ub_q_prba = cs.pi/4*cs.DM.ones((n_q, N))
    lb_dq_prba = -cs.inf*cs.DM.ones((n_q, N))
    ub_dq_prba = cs.inf*cs.DM.ones((n_q, N))
    lb_w = cs.vertcat(cs.vec(lb_q_prba), cs.vec(lb_dq_prba))
    ub_w = cs.vertcat(cs.vec(ub_q_prba), cs.vec(ub_dq_prba))

    # Parameters
    p = cs.vertcat(cs.vec(q_b), cs.vec(dq_b), cs.vec(p_e), cs.vec(dp_e))

    # Define nlp
    nlp = {'x': w, 'p': p, 'f': cost, 'g': g}

    # Create solver instance
    opts = {'ipopt': {'max_iter': 100, 'print_level': 0}, 'print_time': 0}
    solver = cs.nlpsol('solver', 'ipopt', nlp, opts)
    return solver, lb_w, ub_w


def generate_output_casadi_fcn(prba_params, e_marker_name='marker_1'):
    """ Generate a casadi function that computes the DLO's 
    end-point position and velocity

    :param prba_params: prba parameters that define the DLO
    
    :return: casadi function that computes the DLO's end-point position and velocity
    """
    # Create pinocchio model and data for the DLO
    m, _, _ = models.create_prba_pinocchio_model(prba_params)

    cmodel = cpin.Model(m)
    cdata = cmodel.createData()

    # Create casadi symbols for the DLO's joint positions and velocities
    cq = cs.SX.sym('q', m.nq, 1)
    cdq = cs.SX.sym('dq', m.nv, 1)

    # Compute the forward of the rigid body chain
    cpin.forwardKinematics(cmodel, cdata, cq, cdq)
    cpin.updateFramePlacements(cmodel, cdata)

    # Get the end-point position and velocity
    e_marker_id = cmodel.getFrameId(e_marker_name)

    p_e = cdata.oMf[e_marker_id].translation
    v_e = cpin.getFrameVelocity(
        cmodel, 
        cdata, 
        e_marker_id, 
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    dp_e = v_e.linear

    # Create the output vector and casadi function
    y = cs.vertcat(p_e, dp_e)
    f = cs.Function('f', [cq, cdq], [y])
    return f