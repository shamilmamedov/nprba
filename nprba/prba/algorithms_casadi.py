import casadi as cs
import numpy as np

from nprba.prba.models import JointType, RobotDescription
from nprba.utils.kinematics_numpy import jcalc, Adjoint
import nprba.utils.kinematics_numpy as kutils


def Rp2Trans(R, p):
    return cs.vertcat(
        cs.horzcat(R, p),
        cs.DM([0., 0., 0., 1.])
    )


def aba_lie(
    model,
    q: np.ndarray,
    dq: np.ndarray,
    ddq_0: np.ndarray = None,
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
    k = 0
    q_j, dq_j, ddq_j = [], [], []
    for nq in model.jnqs:
        q_j.append(q[k:k+nq,:])
        dq_j.append(dq[k:k+nq,:])
        ddq_j.append(cs.SX.zeros((nq,1)))
        k += nq
    
    # Gravitational acceleration viewed from base frame, see Featherstone - Page 130
    g0_ = -9.81
    dV_g_B = np.array( [[0., 0., 0., 0., 0., -g0_]] ).T

    tau  = [np.zeros((nq,1)) for nq in model.jnqs]
    # Define spatial velocity - V[0:3] is rot. vel. ω and V[3:6] is transl. vel. v
    V         = [np.zeros((6,1)) for _ in range(model.n_bodies)]
    dV        = [np.zeros((6,1)) for _ in range(model.n_bodies)]
    #F         = [np.zeros((6,1)) for _ in range(model.n_bodies)]
    beta      = [np.zeros((6,1)) for _ in range(model.n_bodies)]
    eta       = [np.zeros((6,1)) for _ in range(model.n_bodies)]
    B_hat     = [np.zeros((6,1)) for _ in range(model.n_bodies)]

    # Jacobians
    S         = [np.zeros((6,nq)) for nq in model.jnqs]
    dS        = [np.zeros((6,nq)) for nq in model.jnqs]

    # Homogeneous transforms
    T_λi    = [np.eye(4) for _ in range(model.n_bodies)]

    # Adjoints and inertias
    Ψ       = [np.zeros((nq,nq)) for nq in model.jnqs]
    I_art   = [np.zeros((6,6)) for _ in range(model.n_bodies)]
    Pi      = [np.zeros((6,6)) for _ in range(model.n_bodies)]
    ad_V    = [np.zeros((6,6)) for _ in range(model.n_bodies)]
    Ad_T_iλ = [np.zeros((6,6)) for _ in range(model.n_bodies)]


    """ FORWARD RECURSION 1 - Calculate body velocities from base to leafs"""
    for i in range(model.n_bodies):
        Tj, S[i], dS[i] = kutils.jcalc(model.jtypes[i], q_j[i], dq_j[i])

        if i == 0:
            V_λ = np.zeros((6,1))
        else:
            V_λ = V[i-1]

        # Describe i-frame in λ-frame (parent-frame of i)
        T_λj = np.array(model.jplacements[i]['T'])
        T_λi[i] = T_λj @ Tj
        Ad_T_iλ[i] = kutils.Adjoint(kutils.TransInv(T_λi[i])) # Ad_T_iλ := Ad_T_λi_inv

        # Velocity and acceleration of i-th body
        V[i] = Ad_T_iλ[i] @ V_λ + S[i] @ dq_j[i]
        ad_V[i] = kutils.ad( V[i] )
        eta[i] = ad_V[i] @ S[i] @ dq_j[i] + dS[i] @ dq_j[i]

    """ BACKWARD RECURSION - Calculate articulated inertias 'I_art' and bias forces 'B_hat' """
    for i in reversed(range(model.n_bodies)):
        # Transform body inertia in b-frame (fixed to COG) to i-frame
        I_body = np.array(model.inertias[i]['I'])
        # Compute articulated body inertia
        if i == (model.n_bodies-1):
            I_art[i] = I_body
            B_hat[i] = -1 * ad_V[i].T @ I_body @ V[i]
        else:
            I_art[i] = I_body + Ad_T_iλ[i+1].T @ Pi[i+1] @ Ad_T_iλ[i+1]
            B_hat[i] = -1 * ad_V[i].T @ I_body @ V[i] + Ad_T_iλ[i+1].T @ beta[i+1] #- F_ext

        Ψ[i] = cs.inv( S[i].T @ I_art[i] @ S[i] )
        Pi[i] = I_art[i] - I_art[i] @ S[i] @ Ψ[i] @ S[i].T @ I_art[i]
        tau[i] = -model.jforcecallbacks[i](q_j[i], dq_j[i])
        tmp1 = eta[i] + S[i] @ Ψ[i] @ ( tau[i] - S[i].T @ ( I_art[i] @ eta[i] + B_hat[i] ) )
        beta[i] = B_hat[i] + I_art[i] @ tmp1

    """ FOWARD RECURSION 2 - Calculate accelerations """
    for i in range(model.n_bodies):

        if i == 0:
            dV_λ = dV_g_B
        else:
            dV_λ = dV[i-1]

        if ( i == 0 ) and ( simulation_mode == 'hybrid' ):
            # In this ABA implementation, we can prescribe the acceleration
            # of the first joint (acceleration between first body and base frame)
            ddq_j[0] = ddq_0
        else:
            ddq_j[i] = Ψ[i] @ ( tau[i] - S[i].T @ I_art[i] @ ( Ad_T_iλ[i] @ dV_λ + eta[i] ) - S[i].T @ B_hat[i] ) # Joint acceleration

        dV[i] = Ad_T_iλ[i] @ dV_λ + S[i] @ ddq_j[i] + eta[i] # Acceleration of i-frame viewed in the i-frame
        #F[i] = I_art[i] @ dV[i] + B_hat[i] # Spatial force - Optional computation
        #tau_i = S[i].T @ F[i]

    return cs.vertcat(*ddq_j)