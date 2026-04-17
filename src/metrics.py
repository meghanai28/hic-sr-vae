import math

import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackError
from scipy.sparse.linalg import eigsh


def _prepare_contact_map(mat: np.ndarray, subsample: int | None = None) -> np.ndarray:
    out = np.asarray(mat, dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    # The upstream methods expect non-negative contact intensities.
    out = np.clip(out, 0.0, None)
    if subsample and out.shape[0] > subsample:
        step = max(1, out.shape[0] // subsample)
        idx = np.arange(0, out.shape[0], step, dtype=np.int64)[:subsample]
        out = out[np.ix_(idx, idx)]
    return out


def _upper_tri_csr(mat: np.ndarray) -> csr_matrix:
    return csr_matrix(np.triu(mat).astype(np.float64, copy=False))


def _gd_sqrtvc(m: csr_matrix) -> csr_matrix:
    # Adapted from the official GenomeDISCO implementation.
    mup = m.tocsr()
    mdown = mup.transpose().tocsr()
    mdown.setdiag(0)
    mtogether = (mup + mdown).tocsr()
    sums_sq = np.sqrt(np.asarray(mtogether.sum(axis=1)).reshape(-1))
    sums_sq[sums_sq == 0.0] = 1.0
    d_sq = sps.spdiags(1.0 / sums_sq, [0], mtogether.shape[0], mtogether.shape[1], format="csr")
    return sps.triu(d_sq.dot(mtogether.dot(d_sq)), format="csr")


def _gd_process_matrix(m: csr_matrix, norm: str) -> csr_matrix:
    if norm == "sqrtvc":
        return _gd_sqrtvc(m)
    if norm == "uniform":
        return m.tocsr()
    raise ValueError(f"Unsupported GenomeDISCO norm: {norm}")


def _gd_to_transition(mtogether: csr_matrix) -> csr_matrix:
    sums = np.asarray(mtogether.sum(axis=1)).reshape(-1)
    sums[sums == 0.0] = 1.0
    d = sps.spdiags(1.0 / sums, [0], mtogether.shape[0], mtogether.shape[1], format="csr")
    return d.dot(mtogether).tocsr()


def _genomedisco_reproducibility(
    m1_csr: csr_matrix,
    m2_csr: csr_matrix,
    *,
    tmin: int,
    tmax: int,
    transition: bool,
) -> float:
    # Adapted from upstream `genomedisco/comparison_types/disco_random_walks.py`.
    m1up = m1_csr.tocsr()
    m1down = m1up.transpose().tocsr()
    m1down.setdiag(0)
    m1 = (m1up + m1down).tocsr()

    m2up = m2_csr.tocsr()
    m2down = m2up.transpose().tocsr()
    m2down.setdiag(0)
    m2 = (m2up + m2down).tocsr()

    if transition:
        m1 = _gd_to_transition(m1)
        m2 = _gd_to_transition(m2)

    rowsums_1 = np.asarray(m1.sum(axis=1)).reshape(-1)
    rowsums_2 = np.asarray(m2.sum(axis=1)).reshape(-1)
    nonzero_total = 0.5 * (
        float(np.count_nonzero(rowsums_1 > 0.0)) + float(np.count_nonzero(rowsums_2 > 0.0))
    )
    if nonzero_total <= 0:
        return 0.0

    scores: list[float] = []
    rw1 = None
    rw2 = None
    for t in range(1, tmax + 1):
        if t == 1:
            rw1 = m1.copy()
            rw2 = m2.copy()
        else:
            rw1 = rw1.dot(m1)
            rw2 = rw2.dot(m2)
        if t >= tmin:
            diff = float(np.abs(rw1 - rw2).sum())
            scores.append(diff / nonzero_total)

    if not scores:
        return 0.0
    if tmin == tmax:
        auc = scores[0]
    else:
        auc = float(np.trapezoid(np.asarray(scores, dtype=np.float64), dx=1.0) / max(1, len(scores) - 1))
    return float(1.0 - auc)


def genomedisco_score(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    t_steps: int = 3,
    subsample: int = 1024,
    norm: str = "sqrtvc",
) -> float:
    """GenomeDISCO score using the official upstream algorithm."""
    a = _prepare_contact_map(mat_a, subsample=subsample)
    b = _prepare_contact_map(mat_b, subsample=subsample)
    a_csr = _gd_process_matrix(_upper_tri_csr(a), norm)
    b_csr = _gd_process_matrix(_upper_tri_csr(b), norm)
    return _genomedisco_reproducibility(a_csr, b_csr, tmin=t_steps, tmax=t_steps, transition=True)


def _hs_laplacian(mat: np.ndarray) -> np.ndarray:
    # Adapted from the official HiC-spector Python reproducibility script.
    deg = np.asarray(mat.sum(axis=1), dtype=np.float64).reshape(-1)
    nonzero = np.where(deg > 0)[0]
    if nonzero.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    deg = deg[nonzero]
    mat = mat[np.ix_(nonzero, nonzero)]
    inv_sqrt = 1.0 / np.sqrt(deg)
    norm = (inv_sqrt[:, None] * mat) * inv_sqrt[None, :]
    lap = np.eye(norm.shape[0], dtype=np.float64) - norm
    return 0.5 * (lap + lap.T)


def _hs_evec_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    d1 = float(np.dot(v1 - v2, v1 - v2))
    d2 = float(np.dot(v1 + v2, v1 + v2))
    return math.sqrt(min(d1, d2))


def _hs_ipr(evec: np.ndarray) -> float:
    denom = float(np.sum(evec**4))
    if denom <= 0:
        return 0.0
    return 1.0 / denom


def _hs_smallest_eigs(lap: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    n = lap.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    keep = min(rank, n)
    if keep == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((n, 0), dtype=np.float64)
    if n <= rank + 1:
        vals, vecs = np.linalg.eigh(lap)
        order = np.argsort(vals)
        vals = np.real(vals[order][:keep])
        vecs = np.real(vecs[:, order][:, :keep])
        return vals, vecs
    try:
        vals, vecs = eigsh(csr_matrix(lap), k=keep, which="SM")
        order = np.argsort(vals)
        vals = np.real(vals[order])
        vecs = np.real(vecs[:, order])
        return vals, vecs
    except ArpackError:
        # Some smoothed baseline matrices can be numerically awkward for ARPACK.
        vals, vecs = np.linalg.eigh(lap)
        order = np.argsort(vals)
        vals = np.real(vals[order][:keep])
        vecs = np.real(vecs[:, order][:, :keep])
        return vals, vecs


def hicspector_score(mat_a: np.ndarray, mat_b: np.ndarray, rank: int = 20, subsample: int = 512) -> float:
    """HiC-Spector score using the official upstream algorithm."""
    m1 = _prepare_contact_map(mat_a, subsample=subsample)
    m2 = _prepare_contact_map(mat_b, subsample=subsample)

    k1 = np.sign(m1).sum(axis=1)
    d1 = np.diag(m1)
    kd1 = ~((k1 == 1) & (d1 > 0))
    k2 = np.sign(m2).sum(axis=1)
    d2 = np.diag(m2)
    kd2 = ~((k2 == 1) & (d2 > 0))
    keep_nodes = np.nonzero((k1 + k2 > 0) & kd1 & kd2)[0]
    if keep_nodes.size == 0:
        return 0.0

    m1b = m1[np.ix_(keep_nodes, keep_nodes)]
    m2b = m2[np.ix_(keep_nodes, keep_nodes)]

    nz1 = np.where(m1b.sum(axis=1) > 0)[0]
    nz2 = np.where(m2b.sum(axis=1) > 0)[0]

    lap1 = _hs_laplacian(m1b)
    lap2 = _hs_laplacian(m2b)
    _, vecs1 = _hs_smallest_eigs(lap1, rank=rank)
    _, vecs2 = _hs_smallest_eigs(lap2, rank=rank)

    b1_extend = np.zeros((m1b.shape[0], vecs1.shape[1]), dtype=np.float64)
    b2_extend = np.zeros((m2b.shape[0], vecs2.shape[1]), dtype=np.float64)
    if vecs1.size:
        b1_extend[nz1, :] = vecs1
    if vecs2.size:
        b2_extend[nz2, :] = vecs2

    ipr_cut = 5.0
    ipr1 = np.asarray([_hs_ipr(b1_extend[:, i]) for i in range(b1_extend.shape[1])], dtype=np.float64)
    ipr2 = np.asarray([_hs_ipr(b2_extend[:, i]) for i in range(b2_extend.shape[1])], dtype=np.float64)
    b1_eff = b1_extend[:, ipr1 > ipr_cut]
    b2_eff = b2_extend[:, ipr2 > ipr_cut]
    num_evec_eff = min(b1_eff.shape[1], b2_eff.shape[1])
    if num_evec_eff == 0:
        return 0.0

    evd = np.asarray(
        [_hs_evec_distance(b1_eff[:, i], b2_eff[:, i]) for i in range(num_evec_eff)],
        dtype=np.float64,
    )
    l = math.sqrt(2.0)
    return float(abs(l - evd.sum() / num_evec_eff) / l)
