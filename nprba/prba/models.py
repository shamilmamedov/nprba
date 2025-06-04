import yaml
import numpy as np
import pinocchio as pin
import hppfcl as fcl
from collections import namedtuple
from typing import List
import jax.numpy as jnp
import pandas as pd

import nprba.utils.kinematics_jax as jutils
import nprba.utils.prba as prba
from nprba.utils.joints import JointType, JOINTTYPE_TO_NQ


class PRBAParameters:
    def __init__(self, 
        n_seg: int, 
        dlo_params: prba.DLOParameters,
        base_joint_type: JointType = JointType.FREE,
        p_markers: List = None,
        rpy_ee_b: jnp.ndarray = None,
        p_ee_b: jnp.ndarray = None
    ) -> None:
        self.n_seg = n_seg
        self.base_joint_type = base_joint_type

        if p_markers is not None:
            self.marker_positions = p_markers
        else:
            self.marker_positions = []

        if rpy_ee_b is None:
            rpy_ee_b = jnp.array([[0., 0., 0.]]).T
        self.rpy_ee_b = rpy_ee_b

        if p_ee_b is None:
            p_ee_b = jnp.array([[0., 0., 0.]]).T
        self.p_ee_b = p_ee_b

        L = dlo_params.length
        self.lengths = prba.compute_rfe_lengths(L, n_seg)
        self.m, self.rc, self.I = prba.compute_rfe_inertial_parameters(
            self.lengths, dlo_params
        )
        self.joint_force_callbacks = [lambda x, dx : 0.*x for _ in range(n_seg+1)]

        self.k_, self.d_ = prba.compute_rfe_sde_parameters_for_uniform_discretization(
            n_seg, dlo_params
        )

    def set_ee_to_b_transform(self, rpy, p):
        self.rpy_ee_b = rpy
        self.p_ee_b = p

    def set_marker_positions(self, m_positions):
        self.marker_positions = m_positions

    def set_masses(self, masses):
        self.m = masses

    def set_coms(self, rc):
        self.rc = rc

    def set_inertias(self, I):
        self.I = I

    def set_joint_force_callbacks(self, jfc):
        self.joint_force_callbacks = jfc

    def compute_jforce_callbacks_from_dlo_params(self):
        # Zero because the first joint is not part of the rfem and has zero stiffness
        self.joint_force_callbacks = (
            [lambda x, dx: 0. * x] + 
            [lambda x, dx: ki * x + di * dx for ki, di in zip(self.k_, self.d_)]
        )
    

def create_prba_pinocchio_model(
    prba_params: PRBAParameters, 
    add_ee_ref_joint: bool = False,
    add_panda: bool = False,
    add_ur10: bool = False,
    dlo_radius: float = 0.010
):
    base_joint_constructor = {
        JointType.U_ZY: create_universal_joint(),
        JointType.P_XYZ: pin.JointModelTranslation(),
        JointType.FREE: create_free_joint(),
        JointType.FIXED: None
    }

    # Compute robots end-effector to dlo beginning transformation
    T_ee_b = pin.SE3.Identity()
    T_ee_b.rotation = np.asarray(jutils.rpy(prba_params.rpy_ee_b))
    T_ee_b.translation = np.asarray(prba_params.p_ee_b)

    # Instantiate model
    if add_panda and add_ur10: 
        raise ValueError("Cannot add both panda and ur10 to the model")

    if add_panda:
        urdf_path = 'robots/panda_description/panda_arm.urdf'
        model, cmodel, vmodel = pin.buildModelsFromUrdf(urdf_path)
        parent_frame_id = model.getFrameId('table_link')
        parent_joint_id = 0
        joint_placement = model.frames[parent_frame_id].placement
    elif add_ur10:
        urdf_path = 'robots/ur10_description/ur10.urdf'
        model, cmodel, vmodel = pin.buildModelsFromUrdf(urdf_path)
        parent_frame_id = model.getFrameId('world')
        parent_joint_id = 0
        joint_placement = model.frames[parent_frame_id].placement
    else:
        model = pin.Model()
        model.name = "MRFEM"
        cmodel = pin.GeometryModel()
        vmodel = pin.GeometryModel()
        parent_frame_id = 0
        parent_joint_id = 0
        joint_placement = pin.SE3.Identity()

    # Add prba to the model
    n_seg = prba_params.n_seg
    rfe_lengths = np.asarray(prba_params.lengths)
    rfe_m = [float(x) for x in prba_params.m]
    rfe_rc = [np.asarray(x) for x in prba_params.rc]
    rfe_I = [np.asarray(x) for x in prba_params.I]
    p_markers = prba_params.marker_positions
    base_joint = base_joint_constructor[prba_params.base_joint_type]

    # Joint and body placement
    body_placement_default = pin.SE3.Identity()
    body_placement_default.rotation = pin.rpy.rpyToMatrix(0., np.pi/2, 0.)
    
    # Implement universal joint by combining two revolute joints
    universal_joint = create_universal_joint()
    jtypes = [base_joint] + [universal_joint]*n_seg
    jids = []
    for k, (jtype, lxk, mk, rck, Ik) in enumerate(zip(jtypes, rfe_lengths, rfe_m, rfe_rc, rfe_I)):
        # Add joint to the model
        joint_name = 'prba_joint_' + str(k+1)
        joint_id = model.addJoint(
            parent_joint_id,
            jtype,
            joint_placement,
            joint_name
        )
        jids.append(joint_id)

        try:
            body_inertia = pin.Inertia(mk, rck, Ik)
        except RuntimeError:
            body_inertia = pin.Inertia(mk, rck, 0.5*(Ik + Ik.T))

        body_placement = body_placement_default.copy()
        shape_placement = body_placement_default.copy()
        body_placement.translation[0] = lxk
        shape_placement.translation[0] = lxk/2
        if k == 0:
            body_placement = T_ee_b * body_placement
            shape_placement = T_ee_b * shape_placement
        model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())

        # Define geometry for visualizzation
        geom_name = "rfe_" + str(k+1)
        shape = fcl.Cylinder(dlo_radius, float(lxk))
        geom_obj = pin.GeometryObject(
            geom_name, joint_id, shape, shape_placement
        )
        # geom_obj.meshColor = np.array([0.729, 0.729, 0.729, 1.])
        geom_obj.meshColor = np.array([0.65, 0.65, 0.65, 1.])
        cmodel.addGeometryObject(geom_obj)
        vmodel.addGeometryObject(geom_obj)

        # Adding the next joint
        # NOTE transformation from parent to joint assumed I
        parent_joint_id = joint_id
        joint_placement = pin.SE3.Identity()
        joint_placement.translation[0] = lxk
        if k == 0:
            joint_placement = T_ee_b * joint_placement

    fparents = prba.compute_marker_frames_parent_joints(rfe_lengths, p_markers)
    fplacements = prba.compute_marker_frames_placements(rfe_lengths, fparents, p_markers)
    for k, (pj_idx, f_pos) in enumerate(zip(fparents, fplacements)):
        # Attach frame in the middle of the link
        frame_name = 'marker_' + str(k+1)
        frame_placement = pin.SE3.Identity()
        frame_placement.translation[0] = np.array(f_pos)
        frame = pin.Frame(
            frame_name, jids[pj_idx], jids[pj_idx], frame_placement, pin.FrameType.OP_FRAME
        )
        model.addFrame(frame)

        shape = fcl.Sphere(0.01)
        geom_obj = pin.GeometryObject(frame_name, jids[pj_idx], shape, frame_placement)
        geom_obj.meshColor = np.array([0.31, 0.42, 0.949, 1.])
        cmodel.addGeometryObject(geom_obj)
        vmodel.addGeometryObject(geom_obj)

    if add_ee_ref_joint:
        joint_placement = model.frames[parent_frame_id].placement
        joint_parent_id = 0
        joint_name = 'ee_ref_joint'
        joint_id = model.addJoint(
            joint_parent_id,
            pin.JointModelTranslation(),
            joint_placement,
            joint_name
        )

        body_inertia = pin.Inertia.Zero()
        model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())

        geom_name = "ee_ref"
        shape = fcl.Sphere(0.01)
        shape_placement = pin.SE3.Identity()
        geom_obj = pin.GeometryObject(geom_name, joint_id, shape, shape_placement)
        geom_obj.meshColor = np.array([1.,0.1,0.1,1.])
        cmodel.addGeometryObject(geom_obj)
        vmodel.addGeometryObject(geom_obj)

    return model, cmodel, vmodel


def create_universal_joint(axis_1: str = 'Z', axis_2: str = 'Y'):
    """ Creates universal joint using Composite joint
    """
    name_to_impl = {
        'Z': pin.JointModelRZ(),
        'Y': pin.JointModelRY(),
        'X': pin.JointModelRX(),
    }

    universal_joint = pin.JointModelComposite()
    universal_joint.addJoint(name_to_impl[axis_1])
    universal_joint.addJoint(name_to_impl[axis_2])
    return universal_joint


def create_free_joint():
    """ Creates free joint using Composite joint
    """
    free_joint = pin.JointModelComposite()
    free_joint.addJoint(pin.JointModelTranslation())
    free_joint.addJoint(pin.JointModelSphericalZYX())
    return free_joint



""" For robot description named tuple is used, because it is more neat
in terms of accessing fields and it's immutable!
n_bodies -- number of bodies
n_joints -- number of joints
n_q -- the dimension of the configuration vector
n_frames -- number of frames
jnqs[i] -- number of elements in the joint config vector
jtypes[i] -- joint type of joint i
jparents[i] -- parent link of joint i
jplacements[jparents[i]] -- transfoormation from i joints parent to joint axes
fparents[i] -- parent joint of the frame i
fplacements[fparents[i]] -- transformation from frames parent to frame
inertias[i] -- inertia pcreate_nlink_spherical_pendulumarameters of link i
jforcecallbacks[i] -- force callback function of joint i

NOTE compared to classical robot data structure given by Featherstone
here parents[i] can rather be refered to frames and it contains as a
last element the transformation from the last joint to the end-effector!
"""
RobotDescription = namedtuple(
    'Robot',
    ['n_bodies', 'n_joints', 'n_q', 'n_frames',
     'jtypes', 'jnqs', 'jparents', 'jplacements',
     'fparents', 'fplacements',
     'inertias', 'jforcecallbacks']
)


def create_prba_custom_model(prba_params: PRBAParameters):
    n_seg = prba_params.n_seg
    base_joint_type = prba_params.base_joint_type
    rfe_m = prba_params.m
    rfe_rc = prba_params.rc
    rfe_I = prba_params.I
    jforcecallbacks = prba_params.joint_force_callbacks
    p_markers = prba_params.marker_positions

    jparents = [-1] + [k for k in range(n_seg)]
    jtypes = [base_joint_type] + [JointType.U_ZY for _ in range(n_seg)]
    jnqs = [JOINTTYPE_TO_NQ[x] for x in jtypes]

    R_ee_b = jutils.rpy(prba_params.rpy_ee_b)
    p_ee_b = prba_params.p_ee_b
    T_ee_b = jutils.Rp2Trans(R_ee_b, p_ee_b)
    jlxs = prba.compute_sde_joint_placements(prba_params.lengths, frame = 'parent')
    jplacements = ([{'T': jutils.Rp2Trans(jnp.eye(3), jnp.array([[jlxs[0], 0., 0.]]).T)}] + # from base to first joint
        [{'T': T_ee_b @ jutils.Rp2Trans(jnp.eye(3), jnp.array([[jlxs[1], 0., 0.]]).T)}] + 
        [{'T': jutils.Rp2Trans(jnp.eye(3), jnp.array([[lxk, 0., 0.]]).T)} for lxk in jlxs[2:]]
    )

    inertias = [{'I': jutils.inertia_at_joint(R_ab=jnp.eye(3), p_ba=rck, m=mk, I_b=Ik)}
                for mk, rck, Ik in zip(rfe_m, rfe_rc, rfe_I)]

    # Describe frames which correspond to sensor placements
    fparents = prba.compute_marker_frames_parent_joints(prba_params.lengths, p_markers)
    flxs = prba.compute_marker_frames_placements(prba_params.lengths, fparents, p_markers)
    fplacements = [{'T': jutils.Rp2Trans(jnp.eye(3), jnp.array([[lxk, 0., 0.]]).T)} for lxk in flxs]

    # Descriptive params
    n_bodies = n_seg+1
    n_joints = n_seg+1
    n_q = sum(jnqs)
    n_frames = len(fparents)

    model = RobotDescription(
        n_bodies,
        n_joints,
        n_q,
        n_frames,
        jtypes,
        jnqs,
        jparents,
        jplacements,
        fparents,
        fplacements,
        inertias,
        jforcecallbacks
    )
    return model


def load_aluminium_rod_params():
    """ Loads parameters for alimunium rod
    """
    path_to_yaml = 'config/long-alrod-physical-params.yaml'
    return load_dlo_params_from_yaml(path_to_yaml)


def load_dlo_params_from_yaml(path: str):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return prba.DLOParameters(**data)


def get_prba_params_for_short_alrod(
        n_seg: int = 7, 
        base_joint: JointType = JointType.FREE,
        p_markers: list = None 
    ):
    phys_params_path = 'config/short-alrod-physical-params.yaml'
    dlo_params = load_dlo_params_from_yaml(phys_params_path)

    if p_markers is None: p_markers = [dlo_params.length]
    prba_params = PRBAParameters(n_seg, dlo_params, base_joint, p_markers)
    prba_params.compute_jforce_callbacks_from_dlo_params()
    return prba_params



if __name__ == "__main__":
    pass
