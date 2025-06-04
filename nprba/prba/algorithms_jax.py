import jax
from typing import Tuple
import jax.numpy as jnp
import equinox as eqx

from functools import partial
from jax import lax


import nprba.utils.kinematics_jax as jutils
from nprba.prba.models import RobotDescription



def fwd_kinematics(model: RobotDescription, q: jnp.ndarray):
    """ Computes forward kinematics of a robot described 
    by model

    :param model: model decription
    :param q: configuration of the robot
    """
    k = 0
    q_idxs = []
    for nq in model.jnqs:
        q_idxs.append(jnp.arange(k,k+nq))
        k += nq

    # Joint trasnformations
    o_T_j = jnp.zeros((len(q_idxs), 4, 4))
    for k, qk_idxs in enumerate(q_idxs):
        Tjk, _, _ = jutils.jcalc_jax(model.jtypes[k], q[qk_idxs])
        Tk = lax.dot(model.jplacements[k]['T'], Tjk)
        if model.jparents[k] != -1:
            o_T_j = o_T_j.at[k].set(
                lax.dot(o_T_j[model.jparents[k]], Tk)
            )
        else:
            o_T_j = o_T_j.at[k].set(Tk)

    # Frame transformations
    o_T_f = jnp.zeros((model.n_frames, 4, 4))
    for k in range(model.n_frames):
        Tk = model.fplacements[k]['T']
        o_T_f = o_T_f.at[k].set(
            jnp.dot(o_T_j[model.fparents[k]], Tk)
        )

    return o_T_j, o_T_f


def fwd_velocity_kinematics(model: RobotDescription, q: jnp.ndarray, dq: jnp.ndarray):
    k = 0
    q_idxs = []
    for nq in model.jnqs:
        q_idxs.append(jnp.arange(k,k+nq))
        k += nq

    Vj = jnp.zeros((len(q_idxs), 6, 1))
    o_T_j = jnp.zeros((len(q_idxs), 4, 4))
    Tj, S, dS = zip(*[jutils.jcalc_jax(jti, q[qi_idxs], dq[qi_idxs]) for jti, qi_idxs in zip(model.jtypes, q_idxs)])
    for i, qi_idxs in enumerate(q_idxs):
        if i == 0:
            V_λ = jnp.zeros((6,1))
        else:
            V_λ = Vj[i-1]

        # Describe i-frame in λ-frame (parent-frame of i)
        T_λj = model.jplacements[i]['T']
        T_λi = lax.dot(T_λj, Tj[i])

        # Describe i-th joint frame in 0-frame
        if model.jparents[i] != -1:
            o_T_j = o_T_j.at[i].set(
                lax.dot(o_T_j[model.jparents[i]], T_λi)
            )
        else:
            o_T_j = o_T_j.at[i].set(T_λi)

        Ad_T_iλ = jutils.Adjoint(jutils.TransInv(T_λi))

        # Velocity and acceleration of i-th body
        Vj = Vj.at[i].set(
            lax.dot(Ad_T_iλ, V_λ) + lax.dot(S[i], dq[qi_idxs])
        )

    Vf = dict()
    o_Vf = dict()
    for k in range(model.n_frames):
        V_parent_joint = Vj[model.fparents[k]]
        
        Tk = model.fplacements[k]['T']
        Vf[k] = lax.dot(jutils.Adjoint(jutils.TransInv(Tk)), V_parent_joint)

        o_T_f = lax.dot(o_T_j[model.fparents[k]], Tk)
        o_R_f = jutils.Trans2Rp(o_T_f)[0]

        o_Vf[k] = lax.dot(jax.scipy.linalg.block_diag(o_R_f, o_R_f), Vf[k])

    return Vj, o_Vf


def fwd_joint_position_and_velocity_kinematics(model: RobotDescription, q: jnp.ndarray, dq: jnp.ndarray):
    k = 0
    q_idxs = []
    for nq in model.jnqs:
        q_idxs.append(jnp.arange(k,k+nq))
        k += nq

    Vj = jnp.zeros((len(q_idxs), 6, 1))
    o_T_j = jnp.zeros((len(q_idxs), 4, 4))
    Tj, S, dS = zip(*[jutils.jcalc_jax(jti, q[qi_idxs], dq[qi_idxs]) for jti, qi_idxs in zip(model.jtypes, q_idxs)])
    for i, qi_idxs in enumerate(q_idxs):
        if i == 0:
            V_λ = jnp.zeros((6,1))
        else:
            V_λ = Vj[i-1]

        # Describe i-frame in λ-frame (parent-frame of i)
        T_λj = model.jplacements[i]['T']
        T_λi = lax.dot(T_λj, Tj[i])

        # Describe i-th joint frame in 0-frame
        if model.jparents[i] != -1:
            o_T_j = o_T_j.at[i].set(
                lax.dot(o_T_j[model.jparents[i]], T_λi)
            )
        else:
            o_T_j = o_T_j.at[i].set(T_λi)

        Ad_T_iλ = jutils.Adjoint(jutils.TransInv(T_λi))

        # Velocity and acceleration of i-th body
        Vj = Vj.at[i].set(
            lax.dot(Ad_T_iλ, V_λ) + lax.dot(S[i], dq[qi_idxs])
        )
    return o_T_j, Vj


def compute_markers_positions_and_velocities(model, q, dq):
    o_T_j, Vj = fwd_joint_position_and_velocity_kinematics(model, q, dq)

    p_markers = []
    for k in range(model.n_frames):
        Tk = model.fplacements[k]['T']
        o_T_f = jnp.dot(o_T_j[model.fparents[k]], Tk)
        p_markers.append(jutils.Trans2Rp(o_T_f)[1].T)

    dp_markers = []
    for k in range(model.n_frames):
        V_parent_joint = Vj[model.fparents[k]]
        
        Tk = model.fplacements[k]['T']
        Vf = lax.dot(jutils.Adjoint(jutils.TransInv(Tk)), V_parent_joint)

        o_T_f = jnp.dot(o_T_j[model.fparents[k]], Tk)
        o_R_f = jutils.Trans2Rp(o_T_f)[0]

        o_Vf = lax.dot(jax.scipy.linalg.block_diag(o_R_f, o_R_f), Vf)
        dp_markers.append(o_Vf[3:,:].T)
    return jnp.vstack(p_markers), jnp.vstack(dp_markers)


def compute_markers_positions(model, q):
    """ Computes markers positions for a given configuration
    marker positions are organized as follows
    [p1_x, p1_y, p1_x, p2_x, p2_y, p2_z, ...., pn_x, pn_y, pn_z] 

    :param model: model description
    :param q: configuration of the robot

    :return: jax array with markets positions
    """
    o_T_j, o_T_f = fwd_kinematics(model, q)
    p_markers = []
    for T in o_T_f:
        p_markers.append(jutils.Trans2Rp(T)[1].T)
    return jnp.vstack(p_markers)

vcompute_markers_positions = eqx.filter_vmap(compute_markers_positions, in_axes=(None, 0))       


def compute_markers_velocities(model, q, dq):
    """Computes linear velocities of the markers"""
    Vj, o_Vf = fwd_velocity_kinematics(model, q, dq)
    dp_markers = []
    for key, value in o_Vf.items():
        dp_markers.append(value[3:,:].T)
    return jnp.vstack(dp_markers)


def rnea_lie_jax(
    model: RobotDescription, 
    q: jnp.ndarray, 
    dq: jnp.ndarray, 
    ddq: jnp.ndarray, 
    g0: jnp.ndarray = jnp.array([[0., 0., -9.81]]).T
):
    # Pars inputs for convineice vars
    k = 0
    q_j, dq_j, ddq_j = [], [], []
    for nq in model.jnqs:
        q_j.append(q[k:k+nq,:])
        dq_j.append(dq[k:k+nq,:])
        ddq_j.append(ddq[k:k+nq,:])
        k += nq
    
    o_3x1 = jnp.zeros((3,1))
    o_6x1 = jnp.zeros((6,1))
    V, dV = [], []
    T_λi, Ad_T_iλ, ad_V = [], [], [] 
    Tj, S, dS = zip(*[jutils.jcalc_jax(jti, qi, dqi) for jti, qi, dqi in zip(model.jtypes, q_j, dq_j)])
    for i in range(model.n_bodies):
        if i == 0:
            V_λ = jnp.zeros((6,1))
            dV_λ = jnp.vstack((o_3x1, -g0))
        else:
            V_λ = V[i-1]
            dV_λ = dV[i-1]

        # Describe i-frame in λ-frame (parent-frame of i)
        T_λj = model.jplacements[i]['T']
        T_λi.append(T_λj @ Tj[i])
        Ad_T_iλ.append(jutils.Adjoint(jutils.TransInv(T_λi[i])))

        # Velocity and acceleration of i-th body
        V.append(Ad_T_iλ[i] @ V_λ + S[i] @ dq_j[i])
        ad_V.append(jutils.ad(V[i]))
        dV.append(Ad_T_iλ[i] @ dV_λ + ad_V[i] @ S[i] @ dq_j[i] + 
                  dS[i] @ dq_j[i] + S[i] @ ddq_j[i])
        
    taus = []
    F = o_6x1  
    for i in reversed(range(model.n_bodies)):
        I = model.inertias[i]['I']
        if i == model.n_bodies-1:
            F = I @ dV[i] - ad_V[i].T @ (I @ V[i])
        else:
            F = I @ dV[i] - ad_V[i].T @ (I @ V[i]) + Ad_T_iλ[i+1].T @ F
        taus.append(S[i].T @ F + model.jforcecallbacks[i](q_j[i], dq_j[i]))

    taus.reverse()
    return jnp.vstack(taus)


rnea_lie_jax_vmap = eqx.filter_vmap(rnea_lie_jax, in_axes=(None, 0, 0, 0, 0))


def mass_matrix_rnea_jax(model: RobotDescription, q: jnp.ndarray) -> jnp.ndarray:
    # Prepare variables to call RNEA in parallel
    nq = q.shape[0]
    I_nq = jnp.eye(nq)[:,:,None]
    O_nqxnqx1 = jnp.zeros((nq,nq,1))
    Q = jnp.repeat(q[None,:,:], nq, axis=0)
    G0 = jnp.zeros((nq,3,1))

    # Build mass matrix column by column
    # Modify the model such that joint force callback doesn't
    # mess up mass matrix computation
    new_model = model._replace(
        jforcecallbacks = [lambda x, dx: 0. * x for _ in range(model.n_bodies)]
    )

    return rnea_lie_jax_vmap(new_model, Q, O_nqxnqx1, I_nq, G0)[:,:,0]


def fwd_dynamics_rnea_jax(
    model: RobotDescription,
    q: jnp.ndarray, 
    dq: jnp.ndarray,
) -> jnp.ndarray:
    nq = q.shape[0]
    O_nqx1 = jnp.zeros((nq,1))
    M = mass_matrix_rnea_jax(model, q)
    n = rnea_lie_jax(model, q, dq, O_nqx1)
    return jnp.linalg.solve(M, -n)


def aba_lie_jax(
    model: RobotDescription,
    q: jnp.ndarray,
    dq: jnp.ndarray,
    ddq_0: jnp.ndarray = None,
    simulation_mode='forward'
):
    """
    Implementation of the "Articulated Body Algorithm" in Lie Algebra formulation using same notation as in
    "Lie Group Formulation of Articulated Rigid Body Dynamics" by Junggon Kim.

    :param model: Specifies multibody dynamics structure as detailed in Featherstone
    :param q: generalized position specifies rotation between joint coordinate systems
    :param dq: generalized velocity
    :param ddq_0: acceleration of first joint frame w.r.t to robot base frame
    :param g0: gravitational acceleration
    :param simulation_mode: This implementation of ABA allows to prescribe for the
                            first joint the acceleration ddq_0 instead of the joint torque.
                            If set to 'hybrid', the twist of the first body is calculated using
                            hybrid dynamics. [forward, hybrid]
    :return: generalized acceleration

    Modern robotics library
    ------------------------
    T       : RpToTrans(R, p)
    T^{-1}  : TransInv(T)
    Ad_T    : Adjoint(T)
    Ad_T^*  : Adjoint(T).T
    ad_V    : ad(V)
    ad_V^*  : ad(V).T
    """

    # Pars inputs for readability
    # cum_nqs = jnp.cumsum(jnp.array(model.jnqs))[:-1]
    # q_j = jnp.split(q, cum_nqs)
    # dq_j = jnp.split(dq, cum_nqs)
    # ddq_j = [jnp.zeros((nq,1)) for nq in model.jnqs]

    k = 0
    q_j, dq_j, ddq_j = [], [], []
    for nq in model.jnqs:
        q_j.append(q[k:k+nq,:])
        dq_j.append(dq[k:k+nq,:])
        ddq_j.append(jnp.zeros((nq,1)))
        k += nq

    # Gravitational acceleration viewed from base frame, see Featherstone - Page 130
    g0_ = -9.81
    dV_g_B = jnp.array([[0., 0., 0., 0., 0., -g0_]]).T

    tau  = [jnp.zeros((nq,1)) for nq in model.jnqs]
    # Define spatial velocity - V[0:3] is rot. vel. ω and V[3:6] is transl. vel. v
    V, dV, eta  = [], [], []
    beta = [None]*model.n_bodies
    B_hat = [None]*model.n_bodies

    # Homogeneous transforms
    T_λi = []

    # Adjoints and inertias
    I_art = [None]*model.n_bodies
    Ψ = [None]*model.n_bodies
    Pi = [None]*model.n_bodies
    ad_V, Ad_T_iλ  = [], []

    # Joint transform and jacobians
    Tj, S, dS = zip(*[jutils.jcalc_jax(jti, qi, dqi) for jti, qi, dqi in zip(model.jtypes, q_j, dq_j)])

    """ FORWARD RECURSION 1 - Calculate body velocities from base to leafs"""
    V_λ = jnp.zeros((6,1))
    for i in range(model.n_bodies):
        if i > 0:
            V_λ = V[i-1]

        # Describe i-frame in λ-frame (parent-frame of i)
        T_λj = model.jplacements[i]['T']
        T_λi.append(lax.dot(T_λj, Tj[i]))
        Ad_T_iλ.append(jutils.Adjoint(jutils.TransInv(T_λi[i]))) # Ad_T_iλ := Ad_T_λi_inv

        # Velocity and acceleration of i-th body
        V.append(jnp.dot(Ad_T_iλ[i], V_λ) + jnp.dot(S[i], dq_j[i]))
        ad_V.append(jutils.ad( V[i]))
        eta.append(ad_V[i] @ S[i] @ dq_j[i] + jnp.dot(dS[i], dq_j[i]))

    """ BACKWARD RECURSION - Calculate articulated inertias 'I_art' and bias forces 'B_hat' """
    for i in reversed(range(model.n_bodies)):
        # Transform body inertia in b-frame (fixed to COG) to i-frame
        I_body = model.inertias[i]['I']
        # Compute articulated body inertia
        if i == (model.n_bodies-1):
            I_art[i] = I_body
            B_hat[i] = -1 * ad_V[i].T @ I_body @ V[i]
        else:
            I_art[i] = I_body + Ad_T_iλ[i+1].T @ Pi[i+1] @ Ad_T_iλ[i+1]
            B_hat[i] = -1 * ad_V[i].T @ I_body @ V[i] + Ad_T_iλ[i+1].T @ beta[i+1] #- F_ext

        Ψ[i] = jnp.linalg.inv( S[i].T @ I_art[i] @ S[i] )
        Pi[i] = I_art[i] - I_art[i] @ S[i] @ Ψ[i] @ S[i].T @ I_art[i]
        tau[i] = -model.jforcecallbacks[i](q_j[i], dq_j[i])
        tmp1 = (eta[i] + S[i] @ Ψ[i] @ ( tau[i] - S[i].T @ ( I_art[i] @ eta[i] + B_hat[i] ) ))
        beta[i] = B_hat[i] + jnp.dot(I_art[i], tmp1)

    """ FOWARD RECURSION 2 - Calculate accelerations """
    dV_λ = dV_g_B
    for i in range(model.n_bodies):
        if (i == 0) and (simulation_mode == 'hybrid'):
            # In this ABA implementation, we can prescribe the acceleration
            # of the first joint (acceleration between first body and base frame)
            ddq_j[0] = ddq_0
        else:
            ddq_j[i] = Ψ[i] @ ( tau[i] - S[i].T @ I_art[i] @ ( Ad_T_iλ[i] @ dV_λ + eta[i] ) - S[i].T @ B_hat[i] ) # Joint acceleration

        dV_λ = Ad_T_iλ[i] @ dV_λ + S[i] @ ddq_j[i] + eta[i] # Acceleration of i-frame viewed in the i-frame
        #F[i] = I_art[i] @ dV[i] + B_hat[i] # Spatial force - Optional computation
        #tau_i = S[i].T @ F[i]

    return jnp.vstack(ddq_j)


def ode_jax(model: RobotDescription, x: jnp.ndarray) -> jnp.ndarray:
    q, dq = jnp.split(x, 2)
    return jnp.vstack((dq, 
                      fwd_dynamics_rnea_jax(model, q, dq)))


def hybrid_dynamics_rnea_jax(
    model: RobotDescription,
    q: jnp.ndarray,
    dq: jnp.ndarray,
    ddq: jnp.ndarray,
    tau: jnp.ndarray = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ Implements hybrid dynamics using RNEA algorithm.
    NOTE it is assumed that the prescribed accelerations are for the
    first m joints only!
    NOTE applied applied torques to RFEM are zero
    """
    n_ddq = ddq.shape[0]
    n_tau = model.n_q - n_ddq
    o_ntaux1 = jnp.zeros((n_tau,1))

    # Solve inverse dynamics and find a bias force that guarantees 
    # given acceleration at the specified joints
    t1_ = jnp.vstack((ddq, o_ntaux1))
    C_tilde = rnea_lie_jax(model, q, dq, t1_)

    # Get the inertia matrix
    H = mass_matrix_rnea_jax(model, q)

    # Solve forward dynamics for remaining joints
    ddq_rem = jnp.linalg.solve(H[n_ddq:,n_ddq:], -C_tilde[n_ddq:,:])

    # Get torques that caused the acceleration at the joint
    tau = C_tilde + H @ jnp.vstack((jnp.zeros_like(ddq), ddq_rem))
    return ddq_rem, tau[:n_ddq,:]


def compute_static_equilibrium(model: RobotDescription) -> jnp.ndarray:
    """ Computes the equilibrium of the RFEM model 

    NOTE if the stiffness is zero, then the algorithm might
    diverge
    """
    def equil_fcn(q_rfem):
        qb = jnp.zeros((model.jnqs[0],1))
        q = jnp.vstack((qb, q_rfem))
        o_nqx1 = jnp.zeros((model.n_q, 1))
        return rnea_lie_jax(model, q, o_nqx1, o_nqx1)[model.jnqs[0]:,:]


    o_nqx1 = jnp.zeros((model.n_q, 1))
    prnea = partial(rnea_lie_jax, model, dq=o_nqx1, ddq=o_nqx1)
    
    @jax.jit
    def step(q_eq, Rk):
        jac_Rk = jax.jacfwd(equil_fcn)(q_eq).squeeze()
        q_eq = q_eq - jnp.linalg.solve(jac_Rk, Rk)
        Rk = equil_fcn(q_eq)
        return q_eq, Rk

    # Initial guess for equilibrium
    q_eq = 0.01*jnp.ones((sum(model.jnqs[1:]), 1))
    Rk = equil_fcn(q_eq)
    while 0.5*jnp.linalg.norm(Rk) > 1e-3:
        q_eq, Rk = step(q_eq, Rk)
    
    return q_eq


def rfem_ode_jax(
    model: RobotDescription, 
    x: jnp.ndarray,
    u: jnp.ndarray
) -> jnp.ndarray:
    q, dq = jnp.split(x, 2)
    return jnp.vstack((dq, 
                      aba_lie_jax(model, q, dq, u, simulation_mode='hybrid')))
    


if __name__ == "__main__":
    pass
