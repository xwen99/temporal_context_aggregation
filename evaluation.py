import argparse
import io
import os

import h5py
import networkx as nx
import numpy as np
import torch
from networkx.algorithms.dag import dag_longest_path
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import FeatureDataset
from model import (GRUModule, LSTMModule, NetVLAD, NeXtVLAD, TCA, VideoComparator)
from utils import resize_axis


def calculate_similarities(query_features, target_feature, metric='euclidean', comparator=None):
    """
      Args:
        query_features: global features of the query videos
        target_feature: global features of the target video
        metric: distance metric of features
      Returns:
        similarities: the similarities of each query with the videos in the dataset
    """
    similarities = []
    if metric == 'euclidean':
        dist = np.nan_to_num(
            cdist(query_features, target_feature, metric='euclidean'))
        for i, v in enumerate(query_features):
            sim = np.round(1 - dist[i] / dist.max(), decimals=6)
            similarities.append(sim.item())
    elif metric == 'cosine':
        dist = np.nan_to_num(
            cdist(query_features, target_feature, metric='cosine'))
        for i, v in enumerate(query_features):
            sim = 1 - dist[i]
            similarities.append(sim.item())
    elif metric == 'chamfer':
        for query in query_features:
            sim = chamfer(query, target_feature, comparator)
            similarities.append(sim)
    else:
        for query in query_features:
            sim1 = chamfer(query, target_feature, comparator)
            sim2 = chamfer(target_feature, query, comparator)
            similarities.append((sim1 + sim2) / 2.0)
    return similarities


def chamfer(query, target_feature, comparator=False):
    query = torch.Tensor(query).cuda()
    target_feature = torch.Tensor(target_feature).cuda()
    simmatrix = torch.einsum('ik,jk->ij', [query, target_feature])
    if comparator:
        simmatrix = comparator(simmatrix).detach()
    sim = simmatrix.max(dim=1)[0].sum().cpu().item() / simmatrix.shape[0]
    return sim


def dp(query, target_feature, phi=10):
    n, m = query.shape[0], target_feature.shape[0]
    mismatch = 0
    sims = np.zeros((n, m))
    for i in range(0, n):
        sims[i, 0] = np.dot(query[i], target_feature[0])
    for j in range(0, m):
        sims[0, j] = np.dot(query[0], target_feature[j])
    for i in range(1, n):
        for j in range(1, m):
            sim = np.dot(query[i], target_feature[j])
            if mismatch >= phi:
                sims[i, j] = sim
                mismatch = 0
            else:
                top_left = sims[i - 1, j - 1] + sim
                top = sims[i - 1, j] + sim / 2.0
                left = sims[i, j - 1] + sim / 2.0
                continue
                if top_left >= max(top, left):
                    sims[i, j] = top_left
                else:
                    sims[i, j] = max(top, left)
                    mismatch += 1

    sim = np.sum(np.max(sims, axis=1)) / n
    return sim


def compute_dists(query, target_feature):
    query = torch.Tensor(query).cuda()
    target_feature = torch.Tensor(target_feature).cuda()
    sims = torch.einsum('ik,jk->ij', [query, target_feature]).cpu().numpy()
    unsorted_dists = 1 - sims
    idxs = np.argsort(unsorted_dists)
    rows = np.dot(np.arange(idxs.shape[0]).reshape(
        (idxs.shape[0], 1)), np.ones((1, idxs.shape[1]))).astype(int)
    sorted_dists = unsorted_dists[rows, idxs]
    return idxs, unsorted_dists, sorted_dists


def tn(query_features, refer_features, top_K=5, min_sim=0.80, max_step=10):
    """
      用于计算两组特征(已经做过l2-norm)之间的帧匹配结果
      Args:
        query_features: shape: [N, D]
        refer_features: shape: [M, D]
        top_K: 取前K个refer_frame
        min_sim: 要求query_frame与refer_frame的最小相似度
        max_step: 有边相连的结点间的最大步长
      Returns:
        path_query: shape: [1, L]
        path_refer: shape: [1, L]
    """
    node_pair2id = {}
    node_id2pair = {}
    node_id2pair[0] = (-1, -1)  # source
    node_pair2id[(-1, -1)] = 0
    node_num = 1

    DG = nx.DiGraph()
    DG.add_node(0)

    idxs, unsorted_dists, sorted_dists = compute_dists(query_features, refer_features)

    # add nodes
    for qf_idx in range(query_features.shape[0]):
        for k in range(top_K):
            rf_idx = idxs[qf_idx][k]
            sim = 1 - sorted_dists[qf_idx][k]
            if sim < min_sim:
                break
            node_id2pair[node_num] = (qf_idx, rf_idx)
            node_pair2id[(qf_idx, rf_idx)] = node_num
            DG.add_node(node_num)
            node_num += 1

    node_id2pair[node_num] = (query_features.shape[0],
                              refer_features.shape[0])  # sink
    node_pair2id[(query_features.shape[0], refer_features.shape[0])] = node_num
    DG.add_node(node_num)
    node_num += 1

    # link nodes

    for i in range(0, node_num - 1):
        for j in range(i + 1, node_num - 1):

            pair_i = node_id2pair[i]
            pair_j = node_id2pair[j]

            if(pair_j[0] > pair_i[0] and pair_j[1] > pair_i[1] and
               pair_j[0] - pair_i[0] <= max_step and pair_j[1] - pair_i[1] <= max_step):
                qf_idx = pair_j[0]
                rf_idx = pair_j[1]
                DG.add_edge(i, j, weight=1 - unsorted_dists[qf_idx][rf_idx])

    for i in range(0, node_num - 1):
        j = node_num - 1

        pair_i = node_id2pair[i]
        pair_j = node_id2pair[j]

        if(pair_j[0] > pair_i[0] and pair_j[1] > pair_i[1] and
                pair_j[0] - pair_i[0] <= max_step and pair_j[1] - pair_i[1] <= max_step):
            qf_idx = pair_j[0]
            rf_idx = pair_j[1]
            DG.add_edge(i, j, weight=0)

    longest_path = dag_longest_path(DG)
    if 0 in longest_path:
        longest_path.remove(0)  # remove source node
    if node_num - 1 in longest_path:
        longest_path.remove(node_num - 1)  # remove sink node
    path_query = [node_id2pair[node_id][0] for node_id in longest_path]
    path_refer = [node_id2pair[node_id][1] for node_id in longest_path]

    score = 0.0
    for (qf_idx, rf_idx) in zip(path_query, path_refer):
        score += 1 - unsorted_dists[qf_idx][rf_idx]

    return score


def query_vs_database(model, dataset, args):
    model = model.eval()
    comparator = None
    if args.use_comparator:
        comparator = VideoComparator()
        comparator.load_state_dict(torch.load('models/video_comparator.pth'))
        comparator = comparator.eval()
    if args.cuda:
        model = model.cuda()
        if args.use_comparator:
            comparator = comparator.cuda()
    print('loading features...')
    vid2features = h5py.File(args.feature_path, 'r')
    print('...features loaded')
    test_loader = DataLoader(
        FeatureDataset(vid2features, dataset.get_queries(),
                       padding_size=args.padding_size, random_sampling=args.random_sampling),
        batch_size=1, shuffle=False)
    # Extract features of the queries
    all_db, queries, queries_ids = set(), [], []
    for feature, feature_len, query_id in tqdm(test_loader):
        query_id = query_id[0]
        if feature.shape[1] > 0:
            if args.cuda:
                feature = feature.cuda()
                feature_len = feature_len.cuda()
            # queries.append(model(feature, feature_len).detach().cpu().numpy()[0])
            queries.append(model.encode(
                feature, feature_len).detach().cpu().numpy()[0])
            queries_ids.append(query_id)
            all_db.add(query_id)
    queries = np.array(queries)

    test_loader = DataLoader(
        FeatureDataset(vid2features, dataset.get_database(),
                       padding_size=args.padding_size, random_sampling=args.random_sampling),
        batch_size=1, shuffle=False)
    # Calculate similarities between the queries and the database videos
    similarities = dict({query: dict() for query in queries_ids})
    for feature, feature_len, video_id in tqdm(test_loader):
        video_id = video_id[0]
        # print('current video : {} {}'.format(video_id, feature.shape))
        if feature.shape[1] > 0:
            if args.cuda:
                feature = feature.cuda()
                feature_len = feature_len.cuda()
            # embedding = model(feature, feature_len).detach().cpu().numpy()
            embedding = model.encode(
                feature, feature_len).detach().cpu().numpy()[0]
            all_db.add(video_id)
            sims = calculate_similarities(
                queries, embedding, args.metric, comparator)
            for i, s in enumerate(sims):
                similarities[queries_ids[i]][video_id] = float(s)

    dataset.evaluate(similarities, all_db)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Name of evaluation dataset. Options: CC_WEB_VIDEO, VCDB, '
                             '\"FIVR-200K\", \"FIVR-5K\", \"EVVE\"')

    parser.add_argument('-pc', '--pca_components', type=int, default=1024,
                        help='Number of components of the PCA module.')
    parser.add_argument('-nc', '--num_clusters', type=int, default=256,
                        help='Number of clusters of the NetVLAD model')
    parser.add_argument('-od', '--output_dim', type=int, default=1024,
                        help='Dimention of the output embedding of the NetVLAD model')
    parser.add_argument('-nl', '--num_layers', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('-mp', '--model_path', type=str, required=True,
                        help='Directory of the .pth file containing model state dicts')

    parser.add_argument('-fp', '--feature_path', type=str, required=True,
                        help='Path to the .hdf5 file that contains the features of the dataset')
    parser.add_argument('-ps', '--padding_size', type=int, default=100,
                        help='Padding size of the input data at temporal axis')
    parser.add_argument('-rs', '--random_sampling', action='store_true',
                        help='Flag that indicates that the frames in a video are random sampled if max frame limit is exceeded')
    parser.add_argument('-m', '--metric', type=str, default='euclidean',
                        help='Metric that will be used for similarity calculation')
    parser.add_argument('-uc', '--use_comparator', action='store_true',
                        help='Flag that indicates that the video comparator is used')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    if 'CC_WEB' in args.dataset:
        from data import CC_WEB_VIDEO
        dataset = CC_WEB_VIDEO()
        eval_function = query_vs_database
    elif 'VCDB' in args.dataset:
        from data import VCDB
        dataset = VCDB()
        eval_function = query_vs_database
    elif 'FIVR' in args.dataset:
        from data import FIVR
        dataset = FIVR(version=args.dataset.split('-')[1].lower())
        eval_function = query_vs_database
    elif 'EVVE' in args.dataset:
        from data import EVVE
        dataset = EVVE()
        eval_function = query_vs_database
    else:
        raise Exception('[ERROR] Not supported evaluation dataset. '
                        'Supported options: \"CC_WEB_VIDEO\", \"VCDB\", \"FIVR-200K\", \"FIVR-5K\", \"EVVE\"')

    model = TCA(feature_size=args.pca_components, nlayers=args.num_layers)
    model.load_state_dict(torch.load(args.model_path))
    eval_function(model, dataset, args)


if __name__ == '__main__':
    main()
