import sys, os, tqdm, argparse
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from src import util
from src import matrix
from src import FExtra
from src import feature
import numpy as np



# extract features of train dataset for FExtra
def extract_features(flag, path, n_user):
    input_path = path + "_train.txt"
    output_path = path + "_train" + "_feature.txt"

    df_train = util.txt_to_df(input_path)
    df_train_pos, df_train_neg = util.split_sign(df_train)
    mtx = matrix.create_mtx(df_train, df_train_pos, df_train_neg, n_user)

    if flag:
        print("Start extracting features.........")
        features_train = feature.extract_features(output_path, df_train, mtx)
    else:
        if os.path.isfile(output_path):
            print("Load features.........")
            features_train = util.read_features_from_file(output_path)
        else:
            sys.exit("A file of features is not existed. Please extract them first.")
    return df_train, features_train, mtx


# predict FExtra scores of every node from the seed nodes in test dataset
def predict_FExtra_scores(flag, path, features_train, mtx, n_user):
    input_path = path + "_test.txt"
    output_path = path + "_FExtra.txt"

    df_test = util.txt_to_df(input_path)
    group_test = util.group_by_test(df_test)
    fextra = FExtra.FExtra(features_train, 23)

    if flag:
        print("Start predicting FExtra scores........")
        with open(output_path, 'w') as f:
            pbar = tqdm.tqdm(total=len(group_test.groups.keys()))
            for snode in group_test.groups.keys():
                dic_ans = {name: value for name, value in zip(df_test.loc[group_test.indices[snode]].tid,
                                                              df_test.loc[group_test.indices[snode]].feedback)}
                if len(dic_ans) == 0:
                    pass
                else:
                    util.save_predict(snode, dic_ans, mtx, fextra, f, n_user)
                pbar.update(1)
            pbar.close()
    else:
        print("Load FExtra scores........")
        if not os.path.isfile(output_path):
            sys.exit("A file of FExtra scores is not existed. Please predict them first.")

    del mtx[2]
    del mtx[1]
    del mtx[0][3]
    del mtx[0][2]
    return df_test, mtx


def process(path, result_path, mtx, df_train, df_test, data_name, n_user, c, sign_thres, pre_analysis, m_iter):
    print("Run OBOE......................")
    input_path = path + "_FExtra.txt"
    group_test = util.group_by_test(df_test)

    top_n = [10, 20]
    precision_top = {}; recall_top = {}; ndcg_top = {}
    precision_bottom = {}; recall_bottom = {}; ndcg_bottom = {}
    for n in top_n:
        precision_top[n] = 0
        recall_top[n] = 0
        ndcg_top[n] = 0.0
        precision_bottom[n] = 0
        recall_bottom[n] = 0
        ndcg_bottom[n] = 0
    num_snode, num_top_err, num_bottom_err = 0.0, 0.0, 0.0

    with open(input_path, 'r') as f:
        pbar = tqdm.tqdm(total=len(df_test))
        for snode in group_test.groups.keys():
            num_snode += 1.0
            temp = df_test.loc[group_test.indices[snode]]
            dic_ans = {name: value for name, value in zip(temp.tid,
                                                          temp.feedback)}
            if len(dic_ans) == 0:
                num_snode -= 1.0
                pass

            if (temp['feedback'] == 0).all():
                num_snode -= 1.0
                continue

            scores_snode, signs_snode = util.read_predict(snode, n_user, f)

            # overwrite scores and signs by real edge
            scores_snode[snode] = 1
            signs_snode[snode] = 1
            scores_snode = np.array(scores_snode)
            signs_snode = np.array(signs_snode)
            real_edge = df_train[df_train.sid == snode].tid.values
            real_sign = df_train[df_train.sid == snode].feedback.values
            scores_snode[real_edge] = real_sign
            signs_snode[real_edge] = real_sign

            snode_nodes = df_train[df_train.sid == snode].tid.values

            p_top, r_top, nd_top, p_bottom, r_bottom, nd_bottom = OBOE(
                snode, dic_ans, mtx, scores_snode, sign_thres, pre_analysis, m_iter, c, snode_nodes, top_n, n_user)

            if p_top is None:
                num_top_err += 1.0
            else:
                for n in top_n:
                    precision_top[n] += p_top[n]
                    recall_top[n] += r_top[n]
                    ndcg_top[n] += nd_top[n]
            if p_bottom is None:
                num_bottom_err += 1.0
            else:
                for n in top_n:
                    precision_bottom[n] += p_bottom[n]
                    recall_bottom[n] += r_bottom[n]
                    ndcg_bottom[n] += nd_bottom[n]
            pbar.update(len(dic_ans))
        pbar.close()

    with open(result_path, 'a') as res:
        res.write(data_name + "\n")
        res.write(str(sign_thres[0]) + " " + str(sign_thres[1]) + "\n")
        res.write("Reprobability: " + str(c) + "\n")
        res.write("Max_iter: " + str(m_iter) + "\n")
        for n in top_n:
            precision_top[n] = precision_top[n] / (num_snode-num_top_err)
            recall_top[n] = recall_top[n] / (num_snode-num_top_err)
            precision_bottom[n] = precision_bottom[n] / (num_snode-num_bottom_err)
            recall_bottom[n] = recall_bottom[n] / (num_snode-num_bottom_err)

        # yc: add
        for n in top_n:
            res.write("F1_top@" + str(n) + ": "
                      + str(2*precision_top[n]*recall_top[n]/(precision_top[n]+recall_top[n])) + "\n")
        for n in top_n:
            res.write("NDCG_top@" + str(n) + ": " + str(ndcg_top[n] / (num_snode-num_top_err)) + "\n")
        for n in top_n:
            res.write("F1_bottom@" + str(n) + ": "
                      + str(2*precision_bottom[n]*recall_bottom[n]/(precision_bottom[n]+recall_bottom[n])) + "\n")
        for n in top_n:
            res.write("NDCG_bottom@" + str(n) + ": " + str(ndcg_bottom[n] / (num_snode-num_bottom_err)) + "\n")
    print("Finish OBOE.")


def OBOE(snode, dic_ans, mtx, scores_snode, sign_thres, pre_analysis, m_iter, c, snode_nodes, top_n, n_user):
    nor_mtx_FS, no_neighs = matrix.set_OBOE_mtx(mtx, scores_snode, n_user, sign_thres, pre_analysis)
    vec_scores_p, vec_scores_n = matrix.calculate_RWR(nor_mtx_FS, no_neighs, snode, n_user, m_iter, c)
    vec_scores = vec_scores_p - vec_scores_n
    precision_top, recall_top, ndcg_top = util.evaluation_ranking(snode, dic_ans, vec_scores, snode_nodes, top_n, "top") # yc_add
    precision_bottom, recall_bottom, ndcg_bottom = util.evaluation_ranking(snode, dic_ans, vec_scores, snode_nodes, top_n, "bottom") # yc_add
    return precision_top, recall_top, ndcg_top, precision_bottom, recall_bottom, ndcg_bottom


def init(config):
    if config.dataset == "wiki":
        dataset_name = "Wikipedia"
        n_user = 7118
        pre_analysis = [(0.92, 0.08), (0.72, 0.28), (0.62, 0.38), (0.50, 0.50)]

    path = '.\\datasets\\' + dataset_name + '\\' + config.dataset
    result_path = '.\\results\\' + config.dataset + '_result.txt'

    """
        Extract_Train: if True, extract feature from train file newly and overwrite feature file
        Save_Predict: if True, calculate FExtra scores newly for test file and overwrite prediction file
    """
    if config.func == "extract":
        df_train, features_train, mtx = extract_features(True, path, n_user)
        return
    else:
        df_train, features_train, mtx = extract_features(False, path, n_user)

    if config.func == "predict":
        df_test, mtx = predict_FExtra_scores(True, path, features_train, mtx, n_user)
        return
    else:
        df_test, mtx = predict_FExtra_scores(False, path, features_train, mtx, n_user)

    if config.func == "run":
        process(path, result_path, mtx, df_train, df_test, config.dataset,
                n_user, config.c, [config.p_thres, config.n_thres], pre_analysis, config.m_iter)
    else:
        sys.exit("Function name is wrong. Please select one of (extract, predict, oboe).")


def parse_args():
    parser = argparse.ArgumentParser(description="Run OBOE.")

    parser.add_argument("--dataset", nargs="?", default="wiki",
                        help="Dataset name. Default is \"wiki\"")
    parser.add_argument("--func", default="run",
                        help="select a function of (extract, predict, run). Default is \"run\"")
    parser.add_argument("--p_thres", type=float, default=1.0,
                        help="Positive threshold (beta_+). Default is 1.0")
    parser.add_argument("--n_thres", type=float, default=0.6,
                        help="Negative threshold (beta_-). Default is 0.6")
    parser.add_argument("--c", type=float, default=0.4,
                        help="Restart probability. Default is 0.4.")
    parser.add_argument("--m_iter", type=int, default=50,
                        help="Number of maximum iterations. Default is 50.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    config = parse_args()
    init(config)
