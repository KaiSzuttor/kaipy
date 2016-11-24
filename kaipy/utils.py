def h5md_pos(h5_dh, ts, folded=True):
    """ Sorted position from H5MD file.

    Returns the positions of all particles for timestep `ts` either
    in folded or unfoled coordinates.

    Parameters
    ----------
    h5_dh: h5py file handle
    ts: int
        Timestep for which the coordinates should be returned.
    """
    h5_pos = h5_dh["particles/atoms/position/value"][ts,:,:]
    h5_id = h5_dh["particles/atoms/id/value"][ts,:,:]
    h5_image = h5_dh["particles/atoms/image/value"][ts,:,:]
    sorted_pos = np.array([x for (y, x) in sorted(zip(h5_id, h5_pos))])
    if not folded:
        h5_box = h5_dh["particles/atoms/box/edges"]
        sorted_image = np.array([x for (y, x) in sorted(zip(h5_id, h5_image))])
        sorted_pos += sorted_image * box 
    return sorted_pos
