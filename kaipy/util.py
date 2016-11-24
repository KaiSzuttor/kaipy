import numpy as np


def h5md_pos(h5_dh, ts, folded=True):
    """ Sorted position from H5MD file.

    Returns the positions of all particles for timestep `ts` either
    in folded or unfoled coordinates.

    Parameters
    ----------
    h5_dh: h5py file handle
    ts: int or array like
        Timestep (range) for which the coordinates should be returned.
    """
    h5_pos = h5_dh["particles/atoms/position/value"][ts,:,:]
    h5_id = h5_dh["particles/atoms/id/value"][ts,:,:]
    h5_image = h5_dh["particles/atoms/image/value"][ts,:,:]
    if type(ts) is np.ndarray or type(ts) is list:
        result = np.zeros(h5_pos.shape)
        for i in range(len(ts)):
            sorted_pos = np.array([x for (y, x) in sorted(zip(h5_id[i,:,:], h5_pos[i,:,:]))])
            if not folded:
                h5_box = h5_dh["particles/atoms/box/edges"]
                sorted_image = np.array([x for (y, x) in sorted(zip(h5_id[i,:,:], h5_image[i,:,:]))])
                sorted_pos += sorted_image * h5_box[:]
            result[i,:,:] = sorted_pos
        return result
    else:
        sorted_pos = np.array([x for (y, x) in sorted(zip(h5_id, h5_pos))])
        if not folded:
            h5_box = h5_dh["particles/atoms/box/edges"]
            sorted_image = np.array([x for (y, x) in sorted(zip(h5_id, h5_image))])
            sorted_pos += sorted_image * h5_box[:]                                 
        return sorted_pos
