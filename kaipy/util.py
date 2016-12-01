import numpy as np


def h5md_pos(h5_dh, ts=None, folded=True):
    """ Sorted position from H5MD file.

    Returns the positions of all particles for timestep(s) `ts` either
    in folded or unfoled coordinates.

    Parameters
    ----------
    h5_dh: h5py file handle
    ts: int or array like
        Timestep (range) for which the coordinates should be returned.
    """
    if isinstance(ts, np.ndarray) or isinstance(ts, list) or ts is None:
        if ts is None:
            h5_pos = h5_dh["particles/atoms/position/value"][:, :, :]
            h5_id = h5_dh["particles/atoms/id/value"][:, :, :]
            h5_image = h5_dh["particles/atoms/image/value"][:, :, :]
        else:
            h5_pos = h5_dh["particles/atoms/position/value"][np.min(ts):np.max(ts), :, :]
            h5_id = h5_dh["particles/atoms/id/value"][np.min(ts):np.max(ts), :, :]
            h5_image = h5_dh["particles/atoms/image/value"][np.min(ts):np.max(ts), :, :]
        # number of timesteps: n_ts
        n_ts = h5_pos.shape[0]
        result = np.zeros(h5_pos.shape)
        for i in range(n_ts):
            sorted_pos = h5_pos[i, np.argsort(h5_id[i,:,:].flatten()), :]
            if not folded:
                h5_box = h5_dh["particles/atoms/box/edges"]
                sorted_image = h5_image[i, np.argsort(h5_id[i,:,:].flatten()), :]
                sorted_pos += sorted_image * h5_box[:]
            result[i, :, :] = sorted_pos
        return result
    else:
        h5_pos = h5_dh["particles/atoms/position/value"][ts, :, :]
        h5_id = h5_dh["particles/atoms/id/value"][ts, :, :]
        h5_image = h5_dh["particles/atoms/image/value"][ts, :, :]
        id_flat = h5_id.ravel()
        sorted_pos = h5_pos[np.argsort(id_flat), :]
        if not folded:
            h5_box = h5_dh["particles/atoms/box/edges"]
            sorted_image = h5_image[np.argsort(id_flat)]
            sorted_pos += sorted_image * h5_box[:]
        return sorted_pos
