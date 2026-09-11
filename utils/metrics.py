import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_cloth_ids, g_cloth_ids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view and same clothing are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid, camid and cloth_id
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_cloth_id = q_cloth_ids[q_idx]

        # remove gallery samples that have the same pid and camid OR same pid and cloth_id with query
        order = indices[q_idx]  # select one row
        remove_same_cam = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        remove_same_cloth = (g_pids[order] == q_pid) & (g_cloth_ids[order] == q_cloth_id)
        remove = remove_same_cam | remove_same_cloth
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    # Pad to max_rank so callers can always index cmc[0], cmc[4], cmc[9]: a query
    # whose gallery is shorter than max_rank has already retrieved everything, so
    # every further rank stays a hit and the curve keeps its last value.
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    if all_cmc.shape[0] < max_rank:
        all_cmc = np.concatenate(
            [all_cmc, np.ones((max_rank - all_cmc.shape[0], all_cmc.shape[1]),
                              dtype=all_cmc.dtype)], axis=0)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def _prune_gallery(g_pids, g_camids, g_cloth_ids, q_pid, q_camid, q_cloth_id,
                   use_cloth):
    """CSCI's junk-removal rule, shared by both evaluators."""
    remove = (g_pids == q_pid) & (g_camids == q_camid)
    if use_cloth:
        remove = remove | ((g_pids == q_pid) & (g_cloth_ids == q_cloth_id))
    return np.invert(remove)


def _cmc_map(indices, matches, g_pids, g_camids, g_cloth_ids, q_pids, q_camids,
             q_cloth_ids, use_cloth, max_rank):
    """Shared CMC/mAP loop for the CC (``use_cloth``) and General rules.

    Each query may keep a different number of gallery entries, so the per-query
    CMC curves are accumulated in a fixed ``max_rank`` accumulator instead of
    being stacked (a ragged stack raises on numpy >= 1.24).
    """
    all_cmc = np.zeros(max_rank, dtype=np.float64)
    all_AP = []
    num_valid_q = 0.

    # pre-sort gallery attributes once; row q of these equals ``order`` above
    g_pids_order = g_pids[indices]
    g_camids_order = g_camids[indices]
    g_cloth_ids_order = g_cloth_ids[indices]

    for q_idx in range(len(q_pids)):
        # NOTE: row q_idx of these arrays is the gallery reordered by
        # ``indices[q_idx]``, so ``keep`` masks positions of the sorted gallery --
        # exactly like the original ``matches[q_idx][keep]`` formulation.
        keep = _prune_gallery(g_pids_order[q_idx], g_camids_order[q_idx],
                              g_cloth_ids_order[q_idx],
                              q_pids[q_idx], q_camids[q_idx], q_cloth_ids[q_idx],
                              use_cloth)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # query identity does not appear in gallery (or every match was pruned)
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        # pad to max_rank: a query whose gallery shrank (through pruning) yields a
        # shorter curve, and "no more gallery left" means every remaining rank is
        # a hit, i.e. the curve stays at its last value.
        if cmc.shape[0] < max_rank:
            cmc = np.concatenate([cmc, np.ones(max_rank - cmc.shape[0],
                                               dtype=cmc.dtype)])
        all_cmc += cmc[:max_rank]
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        all_AP.append(tmp_cmc.sum() / num_rel)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    return (all_cmc / num_valid_q).astype(np.float32), float(np.mean(all_AP))


def eval_func_LTCC(distmat, q_pids, g_pids, q_camids, g_camids, q_cloth_ids, g_cloth_ids, max_rank=50):
    """Clothes-Changing (CC) rule, aligned with ICCV-CSCI-Person-ReID/utils/metrics.py.

    Gallery samples sharing the query's identity AND either its camera or its
    clothing are treated as junk and removed::

        remove  = (g_pid == q_pid) & (g_camid == q_camid)
        remove |= (g_pid == q_pid) & (g_cloth == q_cloth)
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    return _cmc_map(indices, matches, g_pids, g_camids, g_cloth_ids,
                    q_pids, q_camids, q_cloth_ids, True, max_rank)


def eval_func_general(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Standard (General / SC) rule: only same-camera gallery entries are removed."""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # cloth is irrelevant under the General rule
    q_cloth_ids = np.zeros(len(q_pids), dtype=np.int64)
    g_cloth_ids = np.zeros(len(g_pids), dtype=np.int64)
    return _cmc_map(indices, matches, g_pids, g_camids, g_cloth_ids,
                    q_pids, q_camids, q_cloth_ids, False, max_rank)


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.cloth_ids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid, cloth_id = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.cloth_ids.extend(np.array(cloth_id))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_cloth_ids = np.asarray(self.cloth_ids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        g_cloth_ids = np.asarray(self.cloth_ids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_cloth_ids, g_cloth_ids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


class R1_mAP_eval_CC(R1_mAP_eval):
    """Clothes-Changing evaluator (same pruning rule as CSCI's ``R1_mAP_eval_LTCC``).

    ``compute`` is a CC-protocol drop-in replacement for ``R1_mAP_eval.compute``;
    ``compute_general`` evaluates the very same features under the Standard rule.
    """

    def _shared(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        qf = feats[:self.num_query]
        gf = feats[self.num_query:]
        q_pids = np.asarray(self.pids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_camids = np.asarray(self.camids[self.num_query:])
        q_cloth_ids = np.asarray(self.cloth_ids[:self.num_query])
        g_cloth_ids = np.asarray(self.cloth_ids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        return (qf, gf, distmat, q_pids, g_pids, q_camids, g_camids,
                q_cloth_ids, g_cloth_ids)

    def compute(self):
        qf, gf, distmat, q_pids, g_pids, q_camids, g_camids, q_cloth_ids, g_cloth_ids = self._shared()
        cmc, mAP = eval_func_LTCC(distmat, q_pids, g_pids, q_camids, g_camids,
                                  q_cloth_ids, g_cloth_ids)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

    def compute_general(self):
        qf, gf, distmat, q_pids, g_pids, q_camids, g_camids, _, _ = self._shared()
        cmc, mAP = eval_func_general(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



