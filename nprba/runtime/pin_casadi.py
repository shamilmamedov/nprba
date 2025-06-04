import casadi as cs
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from acados_template import AcadosModel, AcadosSim
from casados_integrator import CasadosIntegrator
from typing import List, Union

import learning_dlo_dynamics.rfem.models as models
from learning_dlo_dynamics.rfem import algorithms_casadi as algos
from learning_dlo_dynamics.dlo_models import (
    LinearJointForce, NonlinearJointForce
)


def get_jit_opts():
    return {
        "compiler": "shell", 
        "jit": True, 
        "jit_options": {"compiler": "gcc"}
    }


def silu(x):
    return x / (1 + np.exp(-x))


def relu(x):
    return cs.fmax(0., x)


def make_casadi_func_for_linear_joint_force(jf: LinearJointForce):
    q = cs.MX.sym('q', 2)
    v = cs.MX.sym('v', 2)

    delta_q = q - np.asarray(jf.q_offset)/jf.q_offset_scaling
    k = (np.asarray(jf.sqrt_k) / np.asarray(jf.sqrt_k_scaling))**2
    d = (np.asarray(jf.sqrt_d) / np.asarray(jf.sqrt_d_scaling))**2
    out = k * delta_q + d * v

    return cs.Function('linear_joint_force', [q, v], [out]) 


def make_casadi_func_for_nn_joint_force(jf: NonlinearJointForce):
    q = cs.MX.sym('q', 2)
    v = cs.MX.sym('v', 2)

    state = cs.vertcat(q, v)
    W1 = np.asarray(jf.W1)
    W2 = np.asarray(jf.W2)
    h = silu(W1 @ state)
    out = W2 @ h
    return cs.Function('nn_joint_force', [q, v], [out])


def make_casadi_func_for_nonlinear_joint_force(jf: NonlinearJointForce):
    q = cs.MX.sym('q', 2)
    v = cs.MX.sym('v', 2)

    lin = make_casadi_func_for_linear_joint_force(jf.linear_joint_force)
    nn = make_casadi_func_for_nn_joint_force(jf)

    out = lin(q, v) + nn(q, v)
    return cs.Function('nonlinear_joint_force', [q, v], [out])


def make_casadi_func_for_joint_force(jf: Union[LinearJointForce, NonlinearJointForce]):
    if isinstance(jf, LinearJointForce):
        return make_casadi_func_for_linear_joint_force(jf)
    elif isinstance(jf, NonlinearJointForce):
        return make_casadi_func_for_nonlinear_joint_force(jf)
    else:
        raise ValueError("Invalid joint force type")



def simple_floating_base_hybrid_dynamics(
    model, data, q, dq, ddq_base, tau
):
    """
    Implements hybrid dynamics for a floating base system with elastic joints.
    It leverages the pinocchio library and its avaialble routines to compute the
    forward and inverse dynamics, as well as the mass matrix.

    :param model: pinocchio model
    :param data: pinocchio data
    :param q: joint positions of all angles including the base
    :param dq: joint velocities of all angles including the base
    :param ddq_base: acceleration of the base
    :param tau: joint torques of all angles except the base (of passive joints)

    :return: acceleration of all joints including the base
    """
    nq_base = ddq_base.shape[0]

    # Get dummy acceleration for non-base joints
    ddq = np.zeros((model.nq, 1))
    ddq[:nq_base] = ddq_base

    # Call inverse dynamics 
    C_tilde = pin.rnea(model, data, q, dq, ddq).reshape(-1, 1)

    # Call crba to get the mass matrix
    M = pin.crba(model, data, q)

    # Compute the acceleration of non-base joints
    ddq[nq_base:] = np.linalg.solve(M[nq_base:, nq_base:], tau - C_tilde[nq_base:])

    # Calculate torques for the base joint
    tau_base = C_tilde[:nq_base] + M[:nq_base, nq_base:] @ ddq[nq_base:]

    # Put together the torques and compute the full acceleration
    tau_full = np.vstack((tau_base, tau))
    return pin.aba(model, data, q, dq, tau_full).reshape(-1, 1)


def make_casadi_func_for_prb_ode(rfem_params, joint_forces):
    # Create pinocchio model
    model, _, _ = models.create_rfem_pinocchio_model(rfem_params)

    # Create casadi model and data
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    # Create joint forces
    joint_forces_cs = [
        make_casadi_func_for_joint_force(jf) for jf in joint_forces
    ]

    # Define the symbolic variables
    nq = model.nq
    nq_base = model.joints[1].nq
    nq_prb = nq - nq_base

    q_prb = cs.SX.sym('q_prb', nq_prb)
    dq_prb = cs.SX.sym('dq_prb', nq_prb)
    q_b = cs.SX.sym('q_b', nq_base)
    dq_b = cs.SX.sym('dq_b', nq_base)
    ddq_b = cs.SX.sym('ddq_b', nq_base)
    # tau_elastic = cs.SX.sym('tau_elastic', nq_prb)

    q = cs.vertcat(q_b, q_prb)
    dq = cs.vertcat(dq_b, dq_prb)
    ddq = cs.vertcat(ddq_b, cs.SX.zeros(nq_prb))

    # Elastic joint forces
    qi_prb = cs.vertsplit(q_prb, 2)
    dqi_prb = cs.vertsplit(dq_prb, 2)
    tau_elastic = -cs.vertcat(
        *[jf(qi, dqi) for jf, qi, dqi in zip(joint_forces_cs, qi_prb, dqi_prb)]
    )

    # Hybrid dynamics
    C_tilde = cpin.rnea(cmodel, cdata, q, dq, ddq)
    M = cpin.crba(cmodel, cdata, q)
    ddq_prb = cs.solve(M[nq_base:, nq_base:], tau_elastic - C_tilde[nq_base:])
    tau_b = C_tilde[:nq_base] + M[:nq_base, nq_base:] @ ddq_prb
    tau_full = cs.vertcat(tau_b, tau_elastic)
    ddq = cpin.aba(cmodel, cdata, q, dq, tau_full)

    # Create ODE function
    x = cs.vertcat(q_prb, dq_prb)
    u = cs.vertcat(q_b, dq_b, ddq_b)
    out = cs.vertcat(dq_prb, ddq[nq_base:])

    return cs.Function('prb_ode', [x, u], [out])


def make_casadi_func_for_prb_ode_2(rfem_params, joint_forces, jit = False):
    joint_forces_cs = [
        make_casadi_func_for_joint_force(jf) for jf in joint_forces
    ]
    rfem_params.set_joint_force_callbacks(
        [lambda x, dx: 0. * x] + joint_forces_cs
    )
    robot_descr = models.create_rfem_custom_model(rfem_params)

    # Define the symbolic variables
    nq = robot_descr.n_q
    nq_base = robot_descr.jnqs[0]
    nq_prb = nq - nq_base

    q_prb = cs.SX.sym('q_prb', nq_prb)
    dq_prb = cs.SX.sym('dq_prb', nq_prb)
    q_b = cs.SX.sym('q_b', nq_base)
    dq_b = cs.SX.sym('dq_b', nq_base)
    ddq_b = cs.SX.sym('ddq_b', nq_base)

    q = cs.vertcat(q_b, q_prb)
    dq = cs.vertcat(dq_b, dq_prb)

    # Hybrid dynamics
    ddq_expr = algos.aba_lie(
        robot_descr, 
        q, dq, ddq_b, simulation_mode='hybrid'
    )

    # ODE
    x = cs.vertcat(q_prb, dq_prb)
    u = cs.vertcat(q_b, dq_b, ddq_b)
    out = cs.vertcat(dq_prb, ddq_expr[nq_base:])
    if jit:
        jit_opts = get_jit_opts()
        return cs.Function('prb_ode', [x, u], [out], jit_opts)
    else:
        return cs.Function('prb_ode', [x, u], [out])


def make_casadi_func_for_prb_fk(rfem_params, e_marker_name='marker_1'):
    # Create pinocchio model and data for the DLO
    model, _, _ = models.create_rfem_pinocchio_model(rfem_params)

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    # Create casadi symbols for the DLO's joint positions and velocities
    nq = model.nq
    nq_base = model.joints[1].nq
    nq_prb = nq - nq_base

    q_prb = cs.SX.sym('q_prb', nq_prb)
    dq_prb = cs.SX.sym('dq_prb', nq_prb)
    q_b = cs.SX.sym('q_b', nq_base)
    dq_b = cs.SX.sym('dq_b', nq_base)
    ddq_b = cs.SX.sym('ddq_b', nq_base)

    cq = cs.vertcat(q_b, q_prb)
    cdq = cs.vertcat(dq_b, dq_prb)

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
    x = cs.vertcat(q_prb, dq_prb)
    u = cs.vertcat(q_b, dq_b, ddq_b)

    y = cs.vertcat(p_e, dp_e)
    return cs.Function(
        'output', [x, u], [y], 
        # {"compiler": "shell", 
        # "jit": True, 
        # "jit_options": {"compiler": "gcc"}
        # }
    )


def make_casadi_fun_for_MLP(eqx_MLP, activation: str):
    x = cs.SX.sym('x', eqx_MLP.in_size)

    out = x
    n_layers = len(eqx_MLP.layers)
    for i, layer in enumerate(eqx_MLP.layers):
        W = np.array(layer.weight)
        out = W @ out 
        if layer.use_bias:
            b = np.array(layer.bias)
            out += b

        if i < n_layers - 1:
            if activation == 'relu':
                out = relu(out)
            elif activation == 'tanh':
                out = np.tanh(out)
            else:
                raise NotImplementedError()

    return cs.Function('MLP', [x], [out])


def make_casadi_func_for_LTI(eqx_ode, jit = False):
    B_dq = np.array(eqx_ode.B_dq)
    A_dq = np.array(eqx_ode.A_dq)
    c = np.array(eqx_ode.c)

    nq = A_dq.shape[0]
    nu = B_dq.shape[1]

    q = cs.SX.sym('q', nq)
    dq = cs.SX.sym('dq', nq)
    u = cs.SX.sym('u', nu)
    x = cs.vertcat(q, dq)

    ddq = A_dq @ x + B_dq @ u
    dx = cs.vertcat(dq, ddq) + c
    if jit:
        jit_opts = get_jit_opts()
        return cs.Function('LTI', [x, u], [dx], jit_opts)
    else:
        return cs.Function('LTI', [x, u], [dx])


def make_casadi_func_for_NODE(eqx_ode, activation: str, jit = False):
    vector_field = make_casadi_fun_for_MLP(eqx_ode.vector_field, activation)

    nq = 2*eqx_ode.rfem_params.n_seg
    nu = 18
    q = cs.SX.sym('q', nq)
    dq = cs.SX.sym('dq', nq)
    u = cs.SX.sym('u', nu)
    x = cs.vertcat(q, dq)

    xu = cs.vertcat(x, u[3:])
    ddq = vector_field(xu)
    dx = cs.vertcat(dq, ddq)
    if jit:
        jit_opts = get_jit_opts()
        return cs.Function('NODE', [x, u], [dx], jit_opts)
    else:
        return cs.Function('NODE', [x, u], [dx])
    



def build_acados_model_for_prb(ode: cs.Function):
    model_name = "prb_model"
    
    # Symbolic vars
    nx = ode.size_in(0)
    nu = ode.size_in(1)

    x = cs.SX.sym("x", nx)
    dx = cs.SX.sym("dx", nx)
    u = cs.SX.sym("u", nu)
    f_expl = ode(x, u)
    f_impl = dx - f_expl

    # Create acados model
    p = []

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = dx
    model.u = u
    model.p = p
    model.name = model_name

    return model


def create_acados_integrator(model, integrator_opts, dt, integrator_type="IRK", use_cython=True):
    sim = AcadosSim()
    sim.model = model

    # set simulation time
    sim.solver_options.T = dt

    # set options
    sim.solver_options.sens_forw = False
    sim.solver_options.sens_algebraic = False
    sim.solver_options.sens_hess = False
    sim.solver_options.sens_adj = False

    if integrator_type == "GNSF":
        sim.solver_options.integrator_type = "GNSF"
        sim.solver_options.sens_hess = False
    elif integrator_type == "RK4":
        sim.solver_options.integrator_type = "ERK"
    else:
        sim.solver_options.integrator_type = "IRK"

    if integrator_type == "RK4":
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 1
    else:
        sim.solver_options.num_stages = integrator_opts["num_stages"]
        sim.solver_options.num_steps = integrator_opts["num_steps"]

    sim.solver_options.newton_iter = integrator_opts["newton_iter"]

    if integrator_opts["collocation_scheme"] == "radau":
        sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"
    elif integrator_opts["collocation_scheme"] == "legendre":
        sim.solver_options.collocation_type = "GAUSS_LEGENDRE"
    else:
        raise Exception(
            "integrator_opts['collocation_scheme'] must be radau or legendre."
        )

    sim.solver_options.newton_tol = (
        integrator_opts["tol"] / integrator_opts["num_steps"]
    )
    sim.code_export_directory = f'c_generated_code_{model.name}_{sim.solver_options.integrator_type}'

    # create
    casados_integrator = CasadosIntegrator(sim, use_cython=use_cython, code_reuse=False)
    return casados_integrator


def create_casadi_integrator(ode, integrator_opts, dt):
    nx = ode.size_in(0)
    nu = ode.size_in(1)
    
    x = cs.SX.sym("x", nx)
    u = cs.SX.sym("u", nu)
    f_expl_expr = ode(x, u)

    # casadi_integrator = cs.integrator(
    #     "casadi_integrator",
    #     "collocation",
    #     {"x": x, "p": u, "ode": f_expl_expr},
    #     {
    #         "tf": dt,
    #         "collocation_scheme": integrator_opts["collocation_scheme"],
    #         "number_of_finite_elements": integrator_opts["num_steps"],
    #         "interpolation_order": integrator_opts["num_stages"],
    #         "rootfinder_options": {"abstol": integrator_opts["tol"]},
    #         "simplify": True,
    #         "compiler": "shell", 
    #         "jit": True, 
    #         "jit_options": {"compiler": "gcc"}
    #     },
    # )

    casadi_integrator = cs.integrator(
        'casadi_integrator',
        'cvodes',
        {"x": x, "p": u, "ode": f_expl_expr},
        {
            'tf': dt,
            'abstol': 1e-4,
            'reltol': 1e-4,
            "compiler": "shell", 
            "jit": True, 
            "jit_options": {"compiler": "gcc"}
        }
    )

    return casadi_integrator