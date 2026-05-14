import numpy as np


def _choose_index_dtype(nbins):
    max_bins = int(np.max(nbins))
    if max_bins <= np.iinfo(np.int16).max:
        return np.int16
    if max_bins <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def _row_bytes_view(a):
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def sparse_histogramdd_large(
    data,
    bin_widths=None,
    edges=None,
    occupied_bins=None,
    ranges=None,
    chunk_size=500_000,
):
    """
    Efficient sparse high-dimensional histogram.

    New histogram:
        counts, edges, occupied_bins = sparse_histogramdd_large(
            data,
            bin_widths=bin_widths,
        )

    Reuse existing sparse bins:
        counts2, edges2, occupied_bins2 = sparse_histogramdd_large(
            new_data,
            edges=edges,
            occupied_bins=occupied_bins,
        )

    Returns
    -------
    counts : ndarray, shape (M,)
    edges : ndarray, shape (M, D, 2)
        edges[k, d] = [lower_edge, upper_edge]
    occupied_bins : ndarray, shape (M, D)
        Integer bin indices.
    """

    data = np.asarray(data)
    N, ndim = data.shape

    # ============================================================
    # Reuse mode
    # ============================================================
    if edges is not None:
        if occupied_bins is None:
            raise ValueError(
                "When edges is provided, occupied_bins must also be provided."
            )

        edges = np.asarray(edges)
        occupied_bins = np.asarray(occupied_bins)

        if edges.shape != (len(occupied_bins), ndim, 2):
            raise ValueError("edges must have shape (M, D, 2).")

        widths = edges[0, :, 1] - edges[0, :, 0]

        # Recover original grid origin from one known bin
        lo = edges[0, :, 0] - occupied_bins[0] * widths

        idx_dtype = occupied_bins.dtype
        old_bins = np.ascontiguousarray(occupied_bins.astype(idx_dtype, copy=False))

        old_keys = _row_bytes_view(old_bins)
        key_to_pos = {k.tobytes(): i for i, k in enumerate(old_keys)}

        counts = np.zeros(len(occupied_bins), dtype=np.int64)

        for start in range(0, N, chunk_size):
            x = data[start : start + chunk_size]

            idx = np.floor((x - lo) / widths).astype(idx_dtype)

            unique_idx, chunk_counts = np.unique(
                idx,
                axis=0,
                return_counts=True,
            )

            unique_keys = _row_bytes_view(unique_idx)

            for key, c in zip(unique_keys, chunk_counts):
                pos = key_to_pos.get(key.tobytes())
                if pos is not None:
                    counts[pos] += c

        return counts, edges, occupied_bins

    # ============================================================
    # New histogram mode
    # ============================================================
    if bin_widths is None:
        raise ValueError("Provide either bin_widths or edges + occupied_bins.")

    bin_widths = np.asarray(bin_widths, dtype=float)

    if len(bin_widths) != ndim:
        raise ValueError("bin_widths must have length equal to data dimension.")

    if ranges is None:
        lo = data.min(axis=0)
        hi = data.max(axis=0)
    else:
        lo = np.array([r[0] for r in ranges], dtype=float)
        hi = np.array([r[1] for r in ranges], dtype=float)

    nbins = np.ceil((hi - lo) / bin_widths).astype(np.int64)
    idx_dtype = _choose_index_dtype(nbins)

    hist = {}

    for start in range(0, N, chunk_size):
        x = data[start : start + chunk_size]

        idx = np.floor((x - lo) / bin_widths).astype(idx_dtype)

        valid = np.all((idx >= 0) & (idx < nbins), axis=1)
        idx = idx[valid]

        if len(idx) == 0:
            continue

        unique_idx, chunk_counts = np.unique(
            idx,
            axis=0,
            return_counts=True,
        )

        keys = _row_bytes_view(unique_idx)

        for key, c in zip(keys, chunk_counts):
            key = key.tobytes()
            hist[key] = hist.get(key, 0) + int(c)

    M = len(hist)

    occupied_bins = np.empty((M, ndim), dtype=idx_dtype)
    counts = np.empty(M, dtype=np.int64)

    for i, (key, c) in enumerate(hist.items()):
        occupied_bins[i] = np.frombuffer(key, dtype=idx_dtype, count=ndim)
        counts[i] = c

    # Optional but useful: deterministic ordering
    order = np.lexsort(occupied_bins.T[::-1])
    occupied_bins = occupied_bins[order]
    counts = counts[order]

    lower = lo + occupied_bins.astype(float) * bin_widths
    upper = lower + bin_widths
    edges = np.stack([lower, upper], axis=-1)

    return counts, edges, occupied_bins
