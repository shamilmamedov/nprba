import jax.numpy as jnp
import jax
import jax.random as jrandom
import copy
import matplotlib.pyplot as plt
import pandas as pd


class JAXMinMaxScalar:
    """ Scikit scalar uses numpy arrays or stores the min and max values
    of the dataset in numpy arrays. In this implementation Jax arratys
    are used to avoid mixing numpy and jax arrays during training
    """
    def __init__(self, feature_range: tuple) -> None:
        self.min, self.max = feature_range

    def fit(self, x):
        nx = x.shape[-1]
        x = jnp.reshape(x, (-1, nx))
        self._data_min = jnp.min(x, axis=0)
        self._data_max = jnp.max(x, axis=0)

    def transform(self, x):
        x_std = jnp.divide((x - self._data_min), (self._data_max - self._data_min))  
        return x_std * (self.max - self.min) + self.min
    
    def vtransform(self, x):
        return jax.vmap(self.transform)(x)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


def design_minmax_input_scaler(U: jnp.ndarray):
    U_scaler = JAXMinMaxScalar(feature_range=(-1,1))
    U_scaler.fit(U)
    return U_scaler


class DLOTrajectory:
    def __init__(
        self, 
        traj_path: str,
        additional_markers: list[str] = None,
        n_traj: int = None
    ) -> None:
        axes = ['x', 'y', 'z']
        self.dt = 0.004
        self.n_traj = n_traj
        self.data = pd.read_csv(traj_path)
        self.t = jnp.array(self.data[['t']].values)

        self.p_b_cols = [f'p_b_{x}' for x in axes]
        self.phi_b_cols = [f'phi_b_{x}' for x in reversed(axes)]
        self.dp_b_cols = [f'dp_b_{x}' for x in axes]
        self.dphi_b_cols = [f'dphi_b_{x}' for x in reversed(axes)]
        self.ddp_b_cols = [f'ddp_b_{x}' for x in axes]
        self.ddphi_b_cols = [f'ddphi_b_{x}' for x in reversed(axes)]
        self.p_e_cols = [f'p_e_{x}' for x in axes]
        self.dp_e_cols = [f'dp_e_{x}' for x in axes]
        self.q_p_cols = [f'q_{i}' for i in range(1, 8)]

        self.p_b = jnp.array(self.data[self.p_b_cols].values)
        self.phi_b = jnp.array(self.data[self.phi_b_cols].values)
        self.q_b = jnp.hstack((self.p_b, self.phi_b))
        self.dp_b = jnp.array(self.data[self.dp_b_cols].values)
        self.dphi_b = jnp.array(self.data[self.dphi_b_cols].values)
        self.dq_b = jnp.hstack((self.dp_b, self.dphi_b))
        self.ddp_b = jnp.array(self.data[self.ddp_b_cols].values)
        self.ddphi_b = jnp.array(self.data[self.ddphi_b_cols].values)
        self.ddq_b = jnp.hstack((self.ddp_b, self.ddphi_b))
        self.p_e = jnp.array(self.data[self.p_e_cols].values)
        self.dp_e = jnp.array(self.data[self.dp_e_cols].values)
        self.q_p = jnp.array(self.data[self.q_p_cols].values)

        self._get_additional_markers_data(additional_markers)
        
    def _get_additional_markers_data(self, additional_markers):
        # if additional_markers is None:
        #     additional_markers = ['m1', 'm2', 'm3']
        
        # p_m_cols = [f'p_{m}_{x}' for m in additional_markers for x in ['x', 'y', 'z']]
        # dp_m_cols = [f'dp_{m}_{x}' for m in additional_markers for x in ['x', 'y', 'z']]
        if additional_markers is None:
            axes = ['x', 'y', 'z']
            self.p_m1_cols = [f'p_m1_{x}' for x in axes]
            self.dp_m1_cols = [f'dp_m1_{x}' for x in axes]
            self.p_m2_cols = [f'p_m2_{x}' for x in axes]
            self.dp_m2_cols = [f'dp_m2_{x}' for x in axes]
            self.p_m3_cols = [f'p_m3_{x}' for x in axes]
            self.dp_m3_cols = [f'dp_m3_{x}' for x in axes]    

            self.p_m1 = jnp.array(self.data[self.p_m1_cols].values)
            self.dp_m1 = jnp.array(self.data[self.dp_m1_cols].values)
            self.p_m2 = jnp.array(self.data[self.p_m2_cols].values)
            self.dp_m2 = jnp.array(self.data[self.dp_m2_cols].values)
            self.p_m3 = jnp.array(self.data[self.p_m3_cols].values)
            self.dp_m3 = jnp.array(self.data[self.dp_m3_cols].values)

    def get_sliced_copy(self, idxs):
        sliced_obj = copy.copy(self)

        # Slice attributes using the provided indices
        for attr_name in dir(self):
            if not attr_name.startswith("__") and not callable(getattr(self, attr_name)):
                attr = getattr(self, attr_name)
                if isinstance(attr, jnp.ndarray):
                    setattr(sliced_obj, attr_name, attr[idxs,:])
        return sliced_obj

    def __len__(self):
        return self.t.shape[0]
    
    def plot(self, variable):
        if isinstance(variable, list):
            vars, lbls = [], []
            for v in variable:
                if hasattr(self, v):
                    vars.append(getattr(self, v))
                    lbls.append(getattr(self, v + '_cols'))
            plot_trajs([self.t]*len(vars), vars, lbls)
        else:
            if hasattr(self, variable):
                plot_trajs(self.t, getattr(self, variable), getattr(self, variable + '_cols'))


def plot_trajs(T, Y, lbls: list):
    # to treat rollouts and trajectories similarly
    if not isinstance(T, list):
        T, Y = [T], [Y]

    # Sanity check
    n_cols = Y[0].shape[1] // 3

    _, axs = plt.subplots(3, n_cols, sharex=True)
    axs = axs.T.reshape(-1)
    for t, y, l in zip(T, Y, lbls):
        for k, ax in enumerate(axs):
            ax.plot(t, y[:,k], label=l[k])
            # ax.set_ylabel()
            ax.grid(alpha=0.25)
            ax.legend()
    plt.tight_layout()
    plt.show()


def sliding_window(traj_length:int, window: int):
    n_windows = traj_length//window
    idxs = jnp.arange(0, n_windows*window, window)[:,None] + jnp.arange(window)[None, :]
    return idxs  


def rolling_window(traj_length:int, window: int):
    n_windows = traj_length - window + 1
    idxs = jnp.arange(n_windows)[:, None] + jnp.arange(window)[None, :]
    return idxs


def dataloader(arrays, batch_size, *, key):
    """Takes a list of arrays whose first dimension is the data point number
       and randomly arranges the data points in batches.
       Returns a generator for a list of arrays of batched data.
    """
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)

    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


