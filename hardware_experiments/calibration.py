from addict import Dict
import warnings
import yaml
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation 
from pytransform3d.plot_utils import make_3d_axis, plot_box
from pytransform3d.transform_manager import TransformManager
import pandas as pd
from typing import List
import rosbag
import modern_robotics.core as mr
import cv2


from nprba.robot_arms.franka_panda import RobotArmModel


METHOD = cv2.CALIB_HAND_EYE_ANDREFF
# cv2.CALIB_HAND_EYE_TSAI 
# cv2.CALIB_HAND_EYE_PARK (best)
# cv2.CALIB_HAND_EYE_HORAUD
# cv2.CALIB_HAND_EYE_ANDREFF !!!!
# cv2.CALIB_HAND_EYE_DANIILIDIS

N_CONFIGS = 135
DLO_PARENT_FRAME_NAME = 'panda_link7'


def data_sanity_check(datapaths: str):
    panda = RobotArmModel()

    n_config = []
    R_bee, p_bee = [], []
    R_vp, p_vp = [], []
    R_vt, p_vt = [], []
    for k, file_path in enumerate(datapaths):
        try:
            with rosbag.Bag(file_path) as bag:
                qk = get_franka_average_joint_position(bag)
                R_bee_k, p_bee_k = panda.fk_ee(qk)
                p_bee.append(p_bee_k.copy())
                R_bee.append(R_bee_k.copy())

                dct_joint0 = get_vicon_average_transform(bag, "dlo_base")
                T_vp = dct_joint0.T_vo

                R_vp_k, p_vp_k = mr.TransToRp(T_vp)
                p_vp.append(p_vp_k.copy())
                R_vp.append(R_vp_k.copy())

                dct_table = get_vicon_average_transform(bag, "table")
                T_vt = dct_table.T_vo

                R_vt_k, p_vt_k = mr.TransToRp(T_vt)
                p_vt.append(p_vt_k.copy())
                R_vt.append(R_vt_k.copy())

                n_config.append(k+1)
        except FileNotFoundError:
            pass

    delta_pbee = np.diff(np.array(p_bee), axis=0).squeeze()
    delta_pvp = np.diff(np.array(p_vp), axis=0).squeeze()
    delta_pbee_norm = np.linalg.norm(delta_pbee, axis=1)
    delta_pvp_norm = np.linalg.norm(delta_pvp, axis=1)

    conf_ang_err = []
    for k in range(1, len(p_bee)):
        R_error_1 = R_bee[k].dot(R_bee[k-1].T)
        R_error_2 = R_vp[k].dot(R_vp[k-1].T)
        theta_error_1 = np.arccos((np.trace(R_error_1) - 1.)/2.)
        theta_error_2 = np.arccos((np.trace(R_error_2) - 1.)/2.)
        abs_angle_diff_in_deg = np.rad2deg(abs(theta_error_2 - theta_error_1))
        conf_ang_err.append(abs_angle_diff_in_deg)

    columns = ['conf', 'abs_trans_err_pbee_pvp', 'abs_rot_err_Rbe_Rvp', 
               'abs_trans_err_pvt']
    df = pd.DataFrame(columns=columns)
    df['conf'] = n_config[1:]
    df['abs_trans_err_pbee_pvp'] = np.abs(delta_pvp_norm - delta_pbee_norm)*1e3
    df['abs_trans_err_pvt'] = np.linalg.norm(np.diff(p_vt, axis=0), axis=1)*1e3
    df['abs_rot_err_Rbe_Rvp'] = conf_ang_err
    print(df.to_string(index=False))


def estimate_setup_transforms(datapaths: str):
    """
    This script uses data from the Franka robot and Vicon system to estimate the
    unknown transform T_fb (b: vicon frame, f: flange) and T_to (t: table, o: robot zero frame)
    """
    # Specify path of the rosbag files
    transforms = get_transforms_from_rosbag(datapaths)

    # Output: cam2gripper
    R_fb, p_fb = cv2.calibrateHandEye(
        R_gripper2base = transforms.R_be.tolist(),
        t_gripper2base = transforms.p_be.tolist(),
        R_target2cam = transforms.R_pv.tolist(),
        t_target2cam = transforms.p_pv.tolist(),
        method = METHOD
    )
    T_fb = mr.RpToTrans(R_fb, p_fb)

    T_to = compute_table_to_panda_zero_transform(transforms, T_fb)[0]

    return T_fb, T_to


def remove_outlier_transforms(transforms):
    outliers = []
    prev_row = transforms.iloc[0]
    for k in range(1, transforms.shape[0]):
        cur_row = transforms.iloc[k]

        l1 = np.linalg.norm(cur_row.p_be - prev_row.p_be)
        l2 = np.linalg.norm(cur_row.p_vp - prev_row.p_vp)
        abs_diff_in_mm = float(abs(l2-l1)*1000)

        R_error_1 = cur_row.R_be.dot(prev_row.R_be.T)
        R_error_2 = cur_row.R_vp.dot(prev_row.R_vp.T)
        theta_error_1 = np.arccos((np.trace(R_error_1) - 1.)/2.)
        theta_error_2 = np.arccos((np.trace(R_error_2) - 1.)/2.)
        abs_angle_diff_in_deg = np.rad2deg(abs(theta_error_2 - theta_error_1))

        if abs_diff_in_mm > 1 or abs_angle_diff_in_deg > 1.5:
            outliers.append(k+1)
        prev_row = cur_row
        
    return transforms.drop(outliers, axis=0)


def get_transforms_from_rosbag(file_path_list):
    """
    Read in data from rosbag to get the transforms required for hand-eye calibration.
    Inside the rosbag should be data (tf transforms) of different static poses of the Franka panda arm
    and the tf transform of a vicon object attached to the robot's end-effector.
    """
    # Instantiate panda robot instance for computing forward kinematics
    panda = RobotArmModel(ee_frame_name=DLO_PARENT_FRAME_NAME)

    dct_T_be = Dict({'R_be': [], 'p_be': []})
    dct_T_pv = Dict({'R_pv': [], 'p_pv': [], 'p_vp': [], 'R_vp': []})
    dct_T_tv = Dict({'R_tv': [], 'p_tv': []})

    # Open the rosbag file and read the transform messages from some topics
    # T_bee = np.zeros((N_CONFIGS, 4, 4))
    # T_vdlo = np.zeros((N_CONFIGS, 4, 4))
    # T_vt = np.zeros((N_CONFIGS, 4, 4))
    for k, file_path in enumerate(file_path_list):
        try:
            with rosbag.Bag(file_path) as bag:
                qk = get_franka_average_joint_position(bag)
                R_be, p_be = panda.fk_ee(qk)
                dct_T_be.p_be.append(p_be.copy())
                dct_T_be.R_be.append(R_be.copy())

                try:
                    dct_joint0 = get_vicon_average_transform(bag, "dlo_base")
                except ValueError:
                    dct_joint0 = get_vicon_average_transform(bag, "pool_noodle_base")

                dct_T_pv.p_pv.append(dct_joint0.p_ov)
                dct_T_pv.p_vp.append(dct_joint0.p_vo)
                dct_T_pv.R_pv.append(dct_joint0.R_ov)
                dct_T_pv.R_vp.append(dct_joint0.R_vo)

                dct_table = get_vicon_average_transform(bag, "table")
                dct_T_tv.p_tv.append(dct_table.p_ov)
                dct_T_tv.R_tv.append(dct_table.R_ov)
        except FileNotFoundError:
            pass

    df = pd.DataFrame(dct_T_be)
    df2 = pd.DataFrame(dct_T_pv)
    df3 = pd.DataFrame(dct_T_tv)

    return pd.concat([df, df2, df3], axis=1)


def get_franka_average_joint_position(bag):
    q_list = []
    for topic, msg, t in bag.read_messages(topics=["/joint_states"]):
        q_list.append(np.array(msg.position))
    Q = np.vstack(q_list)
    if np.any(np.std(Q, axis=0) > 1e-5):
        warnings.warn("std of the q is above 1e-5 rad!!!")
    return np.mean(Q, axis=0)


def get_vicon_average_transform(bag, object_name):
    """
    Loads data of a vicon object stored inside a rosbag and
    averages over all quaternion vectors and translation vectors

    Hint: R_ov is the rotation from vicon world frame (v) to vicon object frame (o)

    :param bag: rosbag file containing data collected from a STATIONARY vicon object
    :param object_name: Name of the vicon object that has been specified in Vicon Explorer
    :return: Dict() containing transforms
    """
    object_path = f"/vicon/{object_name}/{object_name}"

    quat_list = []
    trans_list = []    
    for topic, msg, t in bag.read_messages(topics=[object_path]):
        trans = msg.transform.translation
        rot = msg.transform.rotation
        trans_list.append(np.array([trans.x, trans.y, trans.z]))
        quat_list.append(np.array([rot.x, rot.y, rot.z, rot.w]))

    dct_tmp = Dict()
    dct_tmp.p_vo = np.mean(np.array(trans_list), axis=0)
    dct_tmp.R_vo = np.array(Rotation.from_quat(quat_list).mean().as_matrix())
    dct_tmp.T_vo = mr.RpToTrans(dct_tmp.R_vo, dct_tmp.p_vo.ravel())
    dct_tmp.T_ov = mr.TransInv(dct_tmp.T_vo)
    dct_tmp.R_ov, dct_tmp.p_ov = mr.TransToRp(dct_tmp.T_ov)
    dct_tmp.p_ov = dct_tmp.p_ov.reshape(-1,1)
    return dct_tmp


def compute_table_to_panda_zero_transform(df, T_fb):
    """ Computes average transformation from the tabel to robot base

    :param df: dataframe containing a list of the transforms T_be, T_pv, T_tv
    :param T_fb: Estimated transform from robot flange to limb origin 

    :return: transformation from table to base
             a list of transformation from the vicon to robot base
    """
    T_vb_list = []
    R_tb_list = []
    p_tb_list = []

    # for i in range(len(df.R_be)):
    for index, row in df.iterrows():
        T_be_i = mr.RpToTrans(row.R_be, row.p_be)
        T_pv_i = mr.RpToTrans(row.R_pv, row.p_pv)
        T_tv_i = mr.RpToTrans(row.R_tv, row.p_tv)

        T_vt_i = mr.TransInv(T_tv_i) # np.linalg.inv(T_tv_i)
        T_tb_i = mr.TransInv(T_be_i @ T_fb @ T_pv_i @ T_vt_i)
        R_tb_i, p_tb_i = mr.TransToRp(T_tb_i)
        R_tb_list.append(R_tb_i)
        p_tb_list.append(p_tb_i)

        T_vb_list.append(T_vt_i @ T_tb_i)# np.linalg.inv(T_bt_i))

    R_tb = Rotation.from_matrix(np.stack(R_tb_list)).mean().as_matrix()
    p_tb = np.mean(np.vstack(p_tb_list), axis=0)
    T_tb = mr.RpToTrans(R_tb, p_tb.flatten())

    return T_tb, T_vb_list


def plot_dlo_transforms():
    """ Plots transformations between the between the frame
    attached to the beginning of the dlo and to markers along the dlo
    """
    list_obj_transf = get_dlo_transforms()

    tm = TransformManager(strict_check=True)
    for i, T_λi in enumerate(list_obj_transf):
        tm.add_transform(f'{i+1}', f'{0}', T_λi)

    plt.figure(figsize=(8, 12))

    ax = make_3d_axis(ax_s=1)
    ax = tm.plot_frames_in('0', ax=ax, s=0.5)
    ax.view_init(30, 20)
    plt.show()


def get_dlo_transforms(bag_path: str, vicon_markers_names: list):
    """ Computes homogenous transformation from the beginning of the dlo
    to vicon frames along the dlo
    """
    # Vicon object names
    nf = len(vicon_markers_names) - 1 
    trans_names = [f'p_bm{x}' for x in range(1, nf)] + ['p_be']
    transf_m0_to_mk = [None for _ in trans_names]

    # Get transforms from the vicon frame to marker frames
    T_vm = [None for _ in trans_names]
    with rosbag.Bag(bag_path) as bag:
        T_vm0 = get_vicon_average_transform(bag, vicon_markers_names[0]).T_vo
        for k, vicon_object_name in enumerate(vicon_markers_names[1:]):
            try:
                T_vm[k] = get_vicon_average_transform(bag, vicon_object_name).T_vo
            except ValueError:
                pass

    # Calculate transforms between the marker frames
    T_m0v = mr.TransInv(T_vm0)
    for k, T in enumerate(T_vm):
        if T is not None:
            transf_m0_to_mk[k] = T_m0v @ T

    # Create dictonary and remove all the None's
    trans_dict = dict(zip(trans_names, transf_m0_to_mk))
    trans_dict = {key: value[:3,3] for key, value in trans_dict.items() if value is not None}
    return trans_dict


def save_transforms_to_yaml(transforms: List, transform_names: List, filename: str):
    # Dictionary of transfroms
    data = dict()
    
    # Add transforms into dict
    for t, t_name in zip(transforms, transform_names):
        t_list = t.tolist()
        data[t_name] = t_list
    
    # Write the dictionary to a YAML file
    file_path = 'config/' + filename
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def calibration_error_on_train_dataset(datapaths: str, T_t_b: np.ndarray, T_l7_dlo: np.ndarray):
    # Get panda arm instance
    panda = RobotArmModel(ee_frame_name=DLO_PARENT_FRAME_NAME)

    theta_error_list = []
    p_error_norm_list = []
    for k, filepath in enumerate(datapaths):
        try:
            with rosbag.Bag(filepath) as bag:
                # Get transform from base to link 7
                qk = get_franka_average_joint_position(bag) 
                R_b_l7, p_b_l7 = panda.fk_ee(qk)
                T_b_l7 = mr.RpToTrans(R_b_l7, p_b_l7.ravel())

                # Get transform from vicon to table
                T_v_t = get_vicon_average_transform(bag, 'table').T_vo

                # Get transform from  vicon to DLO base using estimated
                # transforms
                T_b_dlo_red = T_b_l7 @ T_l7_dlo
                R_b_dlo_pred, p_b_dlo_pred = mr.TransToRp(T_b_dlo_red)

                # Get tansform from vicon to DLO base
                try:
                    T_v_dlo = get_vicon_average_transform(bag, "dlo_base").T_vo
                except ValueError:
                    T_v_dlo = get_vicon_average_transform(bag, "pool_noodle_base").T_vo
                T_b_dlo = mr.TransInv(T_v_t @ T_t_b) @ T_v_dlo
                R_b_dlo, p_b_dlo = mr.TransToRp(T_b_dlo)

                # Position error
                p_error_norm = np.linalg.norm(p_b_dlo_pred - p_b_dlo)
                p_error_norm_list.append(p_error_norm)
                
                # Rotation error
                # See https://tinyurl.com/mvxwua83
                R_error = R_b_dlo_pred.dot(R_b_dlo.T)
                theta_error = np.arccos((np.trace(R_error) - 1.)/2.)
                theta_error_list.append(theta_error)
        except FileNotFoundError:
            pass

    avrg_theta_error = sum(theta_error_list)/len(theta_error_list)
    avrg_p_error_norm = sum(p_error_norm_list)/len(p_error_norm_list)
    print('Average calibration errors')
    print("Average position error:", 
            np.array2string(avrg_p_error_norm*1000., precision=3, separator=','), "mm")
    print("Rotation error:", f"{avrg_theta_error:.5f} rad = {np.rad2deg(avrg_theta_error):.4f} deg")


def calibrate_setup(setup_name: str):
    setup_to_path = {
        'long-alrod': '../../dataset/DLO/long-alrod-calibration/',
        'short-alrod': '../../dataset/DLO/short-alrod-calibration/',
        'pool-noodle': '../../dataset/DLO/pool-noodle-calibration/'
    }
    if setup_name in setup_to_path:
        dataset_dir = setup_to_path[setup_name]
    else:
        raise ValueError

    data_paths = [dataset_dir + f'conf{x}.bag' for x in range(1, N_CONFIGS+1)]

    T_fb, T_to = estimate_setup_transforms(data_paths)
    calibration_error_on_train_dataset(data_paths, T_to, T_fb)

    filename_to_save = f'{setup_name}-calibrated-kinematics.yaml'
    transforms = [T_to, T_fb]
    transform_names = ['T_to', 'T_fb']
    save_transforms_to_yaml(transforms, transform_names, filename_to_save)


def calibrate_vicon_marker_locations(setup_name: str):
    n_vicon_frames = 4
    alrod_vicon_markers_names = (
        ['dlo_base'] + 
        [f'dlo_markers_{str(x)}' for x in range(1, n_vicon_frames)] + 
        ['dlo_end']
    )
    pool_noodle_vicon_markers_names = (
        ['pool_noodle_base', 'pool_noodle_middle', 'pool_noodle_end']
    )

    setup_to_path = {
        'long-alrod': '../../dataset/DLO/long-alrod-marker-locations.bag',
        'short-alrod': '../../dataset/DLO/short-alrod-marker-locations.bag',
        'pool-noodle': '../../dataset/DLO/pool-noodle-marker-locations.bag'
    }
    if setup_name in setup_to_path:
        bag_path = setup_to_path[setup_name]
    else:
        raise ValueError

    if setup_name == 'pool-noodle':
        vicon_markers_names = pool_noodle_vicon_markers_names
    else:
        vicon_markers_names = alrod_vicon_markers_names  

    dlo_transforms_dict = get_dlo_transforms(bag_path, vicon_markers_names)
    filename_to_save = f'{setup_name}-vicon-marker-locations.yaml'
    save_transforms_to_yaml(
        [*dlo_transforms_dict.values()], 
        [*dlo_transforms_dict.keys()], 
        filename_to_save
    )


def main(setup_name: str):
    calibrate_setup(setup_name)
    calibrate_vicon_marker_locations(setup_name)


if __name__ == "__main__":
    # data_sanity_check()
    # main()

    setup_name = 'short-alrod'
    calibrate_setup(setup_name)
    # calibrate_vicon_marker_locations(setup_name)