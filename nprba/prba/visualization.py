from panda3d_viewer import Viewer, ViewerConfig
from pinocchio.visualize import Panda3dVisualizer, MeshcatVisualizer
import time
import matplotlib.pyplot as plt
from matplotlib import animation
import jax.numpy as jnp
import numpy as np
import os
import cv2
import shutil


from nprba.prba import models

PANDA3D_CONFIG = ViewerConfig()
PANDA3D_CONFIG.enable_antialiasing(True, multisamples=5)
PANDA3D_CONFIG.enable_shadow(False)
PANDA3D_CONFIG.show_axes(False)
PANDA3D_CONFIG.show_grid(True)
PANDA3D_CONFIG.show_floor(True)
PANDA3D_CONFIG.enable_spotlight(False)
PANDA3D_CONFIG.enable_hdr(True)
# PANDA3D_CONFIG.set_window_size(1125, 1000)
PANDA3D_CONFIG.set_window_size(1920, 1080)


def setup_visualizer(model, geom_model, vis_model=None,):
    if vis_model is None:
        vis_model = geom_model

    viz = Panda3dVisualizer(model, geom_model, vis_model)
    viewer = Viewer(config=PANDA3D_CONFIG)
    viewer.set_background_color(((255, 255, 255)))
    # viewer.reset_camera((5, 0, 3.0), look_at=(2,0,0))
    # viewer.reset_camera((4., 5.5, 1.5), look_at=(1,-0.5,1.4))
    viewer.reset_camera((4.5, 3.5, 1.5), look_at=(0.5,-0.5,1.0))
    viz.initViewer(viewer=viewer)
    viz.loadViewerModel(group_name=f'{model.name}')

    # set rfe colors
    for k, geom in enumerate(viz.visual_model.geometryObjects):
        if 'rfe' in geom.name or 'ee_ref' in geom.name:
            rgba = vis_model.geometryObjects[k].meshColor
            rgba = (rgba[0], rgba[1], rgba[2], rgba[3])
            viz.viewer.set_material(viz.visual_group, geom.name, rgba)

    return viz, viewer


def visualize_robot(q, dt, n_replays, model, geom_model, vis_model=None):
    viz, viewer = setup_visualizer(model, geom_model, vis_model)
    
    for _ in range(n_replays):
        viz.display(q[0, :])
        time.sleep(2)
        viz.play(q[1:, :], dt)
        time.sleep(1)
    viz.viewer.stop()


def create_demo(name, q, dt, model, geom_model, vis_model=None):
    viz, viewer = setup_visualizer(model, geom_model, vis_model)

    # create a folder to store the images
    capture_dir = 'tmp'
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    nsteps = len(q)
    for i in range(nsteps):
        viz.display(q[i])
        viz.viewer.save_screenshot (f'{capture_dir}/img_{i}.png')
    viz.viewer.stop()

    # create a video from the images
    images = [img for img in os.listdir(capture_dir) if img.startswith("img_")]
    images = sorted(images, key=lambda x: int(x.split('_')[1].split('.')[0]))
    frame = cv2.imread(os.path.join(capture_dir, images[0]))
    height, width, layers = frame.shape

    output_video = f'{name}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video = cv2.VideoWriter(output_video, fourcc, int(1/dt), (width, height))

    for image in images:
        img_path = os.path.join(capture_dir, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()

    # remove directory called tmp
    shutil.rmtree(capture_dir)


def visualize_dlo_rollouts(
    prba_params, X, U, Y, Q_panda=None, dlo_radius: float = 0.01
):
    # Visualize DLO motion
    add_panda = Q_panda is not None
    model, gmodel, vmodel = models.create_prba_pinocchio_model(
        prba_params, 
        add_ee_ref_joint=True,
        add_panda=add_panda,
        dlo_radius=dlo_radius
    )
    j = 250//X.shape[1] 
    # dt = j*0.004
    dt = 0.004
    n_replays = 2
    n_rollouts = len(X)
    for r in range(n_rollouts):
        q_b = jnp.split(U[r], 3, axis=1)[0]
        q_prba = jnp.split(X[r], 2, axis=1)[0]
        p_e = jnp.split(Y[r], 2, axis=1)[0]
        if add_panda:
            q = jnp.concatenate((Q_panda[r], q_b, q_prba, p_e), axis=1)
        else:
            q = jnp.concatenate((q_b, q_prba, p_e), axis=1)
        visualize_robot(
            np.array(q), dt, n_replays, model, gmodel, vmodel
        )


def create_demo_with_dlo_rollouts(
    dlo, prba_params, X, U, Y, Q_panda=None, dlo_radius: float = 0.01, horizon: int = 1
):
    # Visualize DLO motion
    add_panda = Q_panda is not None
    model, gmodel, vmodel = models.create_prba_pinocchio_model(
        prba_params, 
        add_ee_ref_joint=True,
        add_panda=add_panda,
        dlo_radius=dlo_radius
    )
    j = (horizon*250)//X.shape[1] 
    dt = j*0.004
    n_rollouts = len(X)
    for r in range(n_rollouts):
        q_b = jnp.split(U[r], 3, axis=1)[0]
        q_prba = jnp.split(X[r], 2, axis=1)[0]
        p_e = jnp.split(Y[r], 2, axis=1)[0]
        if add_panda:
            q = jnp.concatenate((Q_panda[r], q_b, q_prba, p_e), axis=1)
        else:
            q = jnp.concatenate((q_b, q_prba, p_e), axis=1)
        create_demo(f'demos/{dlo}/rollout_{r}', np.array(q), dt, model, gmodel, vmodel)
