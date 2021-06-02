import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


def df_to_csr_matrix(df, n_user):
    mtx_csr = csr_matrix((df.feedback, (df.sid, df.tid)), shape=(n_user, n_user))
    return mtx_csr


def df_to_csc_matrix(df, n_user):
    mtx_csc = csc_matrix((df.feedback, (df.sid, df.tid)), shape=(n_user, n_user))
    return mtx_csc


def set_mtx_triad(mtx1, mtx2):
    mtx = [mtx1 * mtx2,
           mtx1 * csr_matrix.transpose(mtx2),
           (csr_matrix.transpose(mtx1) * mtx2).tocsr(),
           (csr_matrix.transpose(mtx1) * csr_matrix.transpose(mtx2)).tocsr()]
    return mtx


# create matrix for feature extraction
def create_mtx(df, df_pos, df_neg, n_user):
    temp1 = csr_matrix((df.feedback, (df.sid, df.tid)), shape=(n_user, n_user))
    temp2 = csr_matrix((df.feedback, (df.tid, df.sid)), shape=(n_user, n_user))
    mtx_train_csr = temp1 + temp2
    mtx_train_csr.data = np.ones_like(mtx_train_csr.data)
    mtx_common = mtx_train_csr * mtx_train_csr

    mtx_cs = [df_to_csr_matrix(df_pos, n_user), df_to_csr_matrix(df_neg, n_user),
              df_to_csc_matrix(df_pos, n_user), df_to_csc_matrix(df_neg, n_user)]

    mtx_triad = []
    mtx_triad.extend(set_mtx_triad(mtx_cs[0], mtx_cs[0]))
    mtx_triad.extend(set_mtx_triad(mtx_cs[0], mtx_cs[1]))
    mtx_triad.extend(set_mtx_triad(mtx_cs[1], mtx_cs[0]))
    mtx_triad.extend(set_mtx_triad(mtx_cs[1], mtx_cs[1]))

    mtx = [mtx_cs, mtx_common, mtx_triad]
    return mtx


# set matrix for OBOE
def set_OBOE_mtx(mtx, scores, n_user, sign_thres, pre_analysis):
    if (0.5 + sign_thres[0]) > (0.5 - sign_thres[1]):
        scores = np.array([1 if x >= (0.5 + sign_thres[0]) else x for x in scores])
        scores = np.array([-1 if x < (0.5 - sign_thres[1]) else x for x in scores])
        scores = np.array([0 if (x != 1) & (x != -1) else x for x in scores])
    else:
        scores = np.array([-1 if x < (0.5 + sign_thres[0]) else x for x in scores])
        scores = np.array([1 if x >= (0.5 - sign_thres[1]) else x for x in scores])
        scores = np.array([2 if (x != 1) & (x != -1) else x for x in scores])
    # pp (np) means that tnode has positive(negative) scores and it will give positive (positive) scores to neighbors.
    pp_sid, pn_sid, np_sid, nn_sid = [], [], [], []
    pp_tid, pn_tid, np_tid, nn_tid = [], [], [], []
    pp_feedback, pn_feedback, np_feedback, nn_feedback = [], [], [], []

    def pass_rate(tn, nei, case, rate, ver):
        if ver == 0:
            a, b = 0, 1
        if case < 2:
            pp_sid.append(tn)
            pn_sid.append(tn)
            pp_tid.append(nei)
            pn_tid.append(nei)
            if case == 0:
                pp_feedback.append(rate[0][a])
                pn_feedback.append(rate[0][b])
            else:
                pp_feedback.append(rate[1][a])
                pn_feedback.append(rate[1][b])
        else:
            np_sid.append(tn)
            nn_sid.append(tn)
            np_tid.append(nei)
            nn_tid.append(nei)
            if case == 2:
                np_feedback.append(rate[2][a])
                nn_feedback.append(rate[2][b])
            else:
                np_feedback.append(rate[3][a])
                nn_feedback.append(rate[3][b])

    # tnode is node on which the surfer is staying.
    # we assume that tnode has both of the scores (positive, negative)
    # and will investigate all of the neighbors node of tnode.
    for tnode in range(n_user):
        neighs_pos = get_neighbors(mtx[0][0], tnode)
        neighs_neg = get_neighbors(mtx[0][1], tnode)
        for n in neighs_pos:
            # over threshold
            if (scores[n] == 1) | (scores[n] == 2):
                # tnode has positive score and balanced triangle
                pp_sid.append(tnode)
                pp_tid.append(n)
                pp_feedback.append(1.0)
                # tnode has negative score and unbalanced triangle.
                pass_rate(tnode, n, 2, pre_analysis, 0)
            # over threshold
            elif (scores[n] == -1) | (scores[n] == 2):
                nn_sid.append(tnode)
                nn_tid.append(n)
                nn_feedback.append(1.0)
                pass_rate(tnode, n, 0, pre_analysis, 0)
            # under threshold
            else:
                pass_rate(tnode, n, 0, pre_analysis, 0)
                pass_rate(tnode, n, 2, pre_analysis, 0)
        for n in neighs_neg:
            if (scores[n] == -1) | (scores[n] == 2):
                pn_sid.append(tnode)
                pn_tid.append(n)
                pn_feedback.append(1.0)
                pass_rate(tnode, n, 3, pre_analysis, 0)
            elif (scores[n] == 1) | (scores[n] == 2):
                np_sid.append(tnode)
                np_tid.append(n)
                np_feedback.append(1.0)
                pass_rate(tnode, n, 1, pre_analysis, 0)
            else:
                pass_rate(tnode, n, 1, pre_analysis, 0)
                pass_rate(tnode, n, 3, pre_analysis, 0)

    mtx_FS = [csr_matrix((pp_feedback, (pp_sid, pp_tid)), shape=(n_user, n_user)),
              csr_matrix((pn_feedback, (pn_sid, pn_tid)), shape=(n_user, n_user)),
              csr_matrix((np_feedback, (np_sid, np_tid)), shape=(n_user, n_user)),
              csr_matrix((nn_feedback, (nn_sid, nn_tid)), shape=(n_user, n_user))]
    nor_mtx_FS, no_neighs = normalize_OBOE_mtx(mtx_FS, n_user)
    return nor_mtx_FS, no_neighs


# normalize matrix for OBOE
def normalize_OBOE_mtx(mtx_FS, n_user):
    nor_mtx_FS = []
    # [0]: positive nodes which don't have neighbors
    # [1]: negative nodes which don't have neighbors
    no_neighs = []
    for i in [0, 2]:
        no_neighs_snode = []
        sid_p, sid_n, tid_p, tid_n, feedback_p, feedback_n = [], [], [], [], [], []
        for snode in range(n_user):
            neighs_p, neighs_feedback_p = get_neighbors_and_feedback(mtx_FS[0+i], snode)
            neighs_n, neighs_feedback_n = get_neighbors_and_feedback(mtx_FS[1+i], snode)
            neighs_len = float(sum(neighs_feedback_p) + sum(neighs_feedback_n))
            if neighs_len == 0:
                no_neighs_snode.append(snode)
                continue
            sid_p.extend([snode] * len(neighs_p))
            sid_n.extend([snode] * len(neighs_n))
            tid_p.extend(neighs_p)
            tid_n.extend(neighs_n)
            feedback_p.extend(neighs_feedback_p / neighs_len)
            feedback_n.extend(neighs_feedback_n / neighs_len)
        nor_mtx_FS.append(csr_matrix((feedback_p, (sid_p, tid_p)), shape=(n_user, n_user)))
        nor_mtx_FS.append(csr_matrix((feedback_n, (sid_n, tid_n)), shape=(n_user, n_user)))
        no_neighs.append(no_neighs_snode)
    return nor_mtx_FS, no_neighs


# calculate OBOE by product matrix to two vectors of scores
def calculate_RWR(mtx_FS, no_neighs, seed, n_user, m_iter, c):
    vec_scores_p = np.zeros(n_user)
    vec_scores_n = np.zeros(n_user)
    vec_scores_p[seed] = 1
    for i in range(m_iter):
        next_vec_scores_p = vec_scores_p * mtx_FS[0] + vec_scores_n * mtx_FS[2]
        next_vec_scores_n = vec_scores_p * mtx_FS[1] + vec_scores_n * mtx_FS[3]
        global_scores = np.append(vec_scores_p[no_neighs[0]] / (2 * n_user),
                                  vec_scores_n[no_neighs[1]] / (2 * n_user))
        global_score = sum(global_scores)
        next_vec_scores_p += global_score
        next_vec_scores_n += global_score
        vec_scores_p = next_vec_scores_p * (1 - c)
        vec_scores_n = next_vec_scores_n * (1 - c)
        vec_scores_p[seed] += c
    return vec_scores_p, vec_scores_n


def get_neighbors(mtx, idx):
    start = mtx.indptr[idx].astype(int)
    end = mtx.indptr[idx + 1].astype(int)
    return mtx.indices[start:end]


def get_neighbors_and_feedback(mtx, idx):
    start1 = mtx.indptr[idx].astype(int)
    end1 = mtx.indptr[idx + 1].astype(int)
    return mtx.indices[start1:end1], mtx.data[start1:end1]


def get_degree(mtx, idx):
    return mtx.indptr[idx + 1] - mtx.indptr[idx]
