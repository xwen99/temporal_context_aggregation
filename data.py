import io
import pickle as pk
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import KVReader
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, utils

from utils import resize_axis


class VCDBPairDataset(Dataset):
    def __init__(self,
                 annotation_path,
                 feature_path='hdfs://haruna/user/lab/wenxin.me/datasets/vcdb/features/imac/vcdb99709_resnet50_imac_pca1024',
                 padding_size=300,
                 random_sampling=False,
                 neg_num=1):
        self.feature_path = feature_path
        self.padding_size = padding_size
        self.random_sampling = random_sampling
        self.neg_num = neg_num
        self.features = h5py.File(self.feature_path, 'r', swmr=True)
        self.pairs = []
        self.vcdb = pk.load(open(annotation_path, 'rb'))
        for pair in self.vcdb['video_pairs']:
            vid1, vid2 = pair['videos'][0], pair['videos'][1]
            self.pairs.append([vid1, vid2])
        self.negs = self.vcdb['negs']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ns = random.sample(self.negs, self.neg_num)
        feat_a, feat_p, feat_n = self.features[self.pairs[index][0]][:], self.features[self.pairs[index][1]][:], [
            self.features[item][:] for item in ns]
        len_a, len_p, len_n = torch.Tensor([len(feat_a)]), torch.Tensor(
            [len(feat_p)]), torch.Tensor([len(item) for item in feat_n])
        a = resize_axis(feat_a, axis=0, new_size=self.padding_size, fill_value=0,
                        random_sampling=self.random_sampling).transpose(-1, -2)
        p = resize_axis(feat_p, axis=0, new_size=self.padding_size, fill_value=0,
                        random_sampling=self.random_sampling).transpose(-1, -2)
        n = torch.stack([resize_axis(item, axis=0, new_size=self.padding_size, fill_value=0,
                                     random_sampling=self.random_sampling).transpose(-1, -2) for item in feat_n])
        return a, p, n, len_a, len_p, len_n


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self,
                 vid2features,
                 videos,
                 padding_size=100,
                 random_sampling=False):
        super(FeatureDataset, self).__init__()
        self.vid2features = vid2features
        self.padding_size = padding_size
        self.random_sampling = random_sampling
        self.videos = videos
        self.keys = self.vid2features.keys()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        if self.videos[index] in self.keys:
            feat = self.vid2features[self.videos[index]][:]
            len_feat = len(feat)
            return resize_axis(feat, axis=0, 
                               new_size=self.padding_size, fill_value=0, 
                               random_sampling=self.random_sampling).transpose(-1, -2), len_feat, self.videos[index]
        else:
            return torch.Tensor([]), 0, 'None'


class CC_WEB_VIDEO(object):

    def __init__(self):
        with open('datasets/cc_web_video.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.database = dataset['vid2index']
        self.queries = dataset['queries']
        self.ground_truth = dataset['ground_truth']
        self.excluded = dataset['excluded']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(map(str, self.database.keys()))

    def calculate_mAP(self, similarities, all_videos=False, clean=False, positive_labels='ESLMV'):
        mAP = 0.0
        for query_set, labels in enumerate(self.ground_truth):
            query_id = self.queries[query_set]
            i, ri, s = 0.0, 0.0, 0.0
            if query_id in similarities:
                res = similarities[query_id]
                for video_id in sorted(res.keys(), key=lambda x: res[x], reverse=True):
                    video = self.database[video_id]
                    if (all_videos or video in labels) and (not clean or video not in self.excluded[query_set]):
                        ri += 1
                        if video in labels and labels[video] in positive_labels:
                            i += 1.0
                            s += i / ri
                positives = np.sum([1.0 for k, v in labels.items() if
                                    v in positive_labels and (not clean or k not in self.excluded[query_set])])
                mAP += s / positives
        return mAP / len(set(self.queries).intersection(similarities.keys()))

    def evaluate(self, similarities, all_db=None):
        if all_db is None:
            all_db = self.database

        print('=' * 5, 'CC_WEB_VIDEO Dataset', '=' * 5)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(
                not_found))
        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos'.format(len(all_db)))

        print('-' * 25)
        print('All dataset')
        print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}\n'.format(
            self.calculate_mAP(similarities, all_videos=False, clean=False),
            self.calculate_mAP(similarities, all_videos=True, clean=False)))

        print('Clean dataset')
        print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}'.format(
            self.calculate_mAP(similarities, all_videos=False, clean=True),
            self.calculate_mAP(similarities, all_videos=True, clean=True)))


class VCDB(object):

    def __init__(self):
        with open('datasets/vcdb.pickle', 'rb') as f:
            dataset = pk.load(f, encoding='latin1')
        self.database = dataset['index']
        self.queries = dataset['index'][:528]
        self.ground_truth = dict({query: set() for query in self.queries})
        for query in self.queries:
            self.ground_truth[query].add(query)
        for pair in dataset['video_pairs']:
            self.ground_truth[pair['videos'][0]].add(pair['videos'][1])
            self.ground_truth[pair['videos'][1]].add(pair['videos'][0])

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(self.database)

    def calculate_mAP(self, query, res, all_db):
        query_gt = self.ground_truth[query]
        query_gt = query_gt.intersection(all_db)

        i, ri, s = 0.0, 0, 0.0
        for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
            if video != query and video in all_db:
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        return s / len(query_gt)

    def evaluate(self, similarities, all_db=None):
        if all_db is None:
            all_db = self.database

        ans = []
        for query, res in similarities.items():
            ans.append(self.calculate_mAP(query, res, all_db))

        print('=' * 5, 'VCDB Dataset', '=' * 5)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(
                not_found))

        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos'.format(len(all_db)))

        print('-' * 16)
        print('VCDB mAP: {:.4f}'.format(np.mean(ans)))


class FIVR(object):

    def __init__(self, version='200k'):
        self.version = version
        with open('datasets/fivr.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.annotation = dataset['annotation']
        self.queries = dataset[self.version]['queries']
        self.database = dataset[self.version]['database']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(self.database)

    def calculate_mAP(self, query, res, all_db, relevant_labels):
        gt_sets = self.annotation[query]
        query_gt = set(sum([gt_sets[label]
                            for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(all_db)

        i, ri, s = 0.0, 0, 0.0
        for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
            if video != query and video in all_db:
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        return s / len(query_gt)

    def evaluate(self, similarities, all_db=None):
        if all_db is None:
            all_db = self.database

        DSVR, CSVR, ISVR = [], [], []
        for query, res in similarities.items():
            DSVR.append(self.calculate_mAP(query, res, all_db,
                                           relevant_labels=['ND', 'DS']))
            CSVR.append(self.calculate_mAP(query, res, all_db,
                                           relevant_labels=['ND', 'DS', 'CS']))
            ISVR.append(self.calculate_mAP(query, res, all_db,
                                           relevant_labels=['ND', 'DS', 'CS', 'IS']))

        print('=' * 5, 'FIVR-{} Dataset'.format(self.version.upper()), '=' * 5)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(
                not_found))

        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos'.format(len(all_db)))

        print('-' * 16)
        print('DSVR mAP: {:.4f}'.format(np.mean(DSVR)))
        print('CSVR mAP: {:.4f}'.format(np.mean(CSVR)))
        print('ISVR mAP: {:.4f}'.format(np.mean(ISVR)))


class EVVE(object):

    def __init__(self):
        with open('datasets/evve.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.events = dataset['annotation']
        self.queries = dataset['queries']
        self.database = dataset['database']
        self.query_to_event = {qname: evname
                               for evname, (queries, _, _) in self.events.items()
                               for qname in queries}

    def get_queries(self):
        return list(self.queries)

    def get_database(self):
        return list(self.database)

    def score_ap_from_ranks_1(self, ranks, nres):
        """ Compute the average precision of one search.
        ranks = ordered list of ranks of true positives (best rank = 0)
        nres  = total number of positives in dataset
        """
        if nres == 0 or ranks == []:
            return 0.0

        ap = 0.0

        # accumulate trapezoids in PR-plot. All have an x-size of:
        recall_step = 1.0 / nres

        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far

            # y-size on left side of trapezoid:
            if rank == 0:
                precision_0 = 1.0
            else:
                precision_0 = ntp / float(rank)
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0
        return ap

    def evaluate(self, similarities, all_db=None):
        results = {e: [] for e in self.events}
        if all_db is None:
            all_db = set(self.database).union(set(self.queries))

        not_found = 0
        for query in self.queries:
            if query not in similarities:
                not_found += 1
            else:
                res = similarities[query]
                evname = self.query_to_event[query]
                _, pos, null = self.events[evname]
                if all_db:
                    pos = pos.intersection(all_db)
                pos_ranks = []

                ri, n_ext = 0.0, 0.0
                for ri, dbname in enumerate(sorted(res.keys(), key=lambda x: res[x], reverse=True)):
                    if dbname in pos:
                        pos_ranks.append(ri - n_ext)
                    if dbname not in all_db:
                        n_ext += 1

                ap = self.score_ap_from_ranks_1(pos_ranks, len(pos))
                results[evname].append(ap)

        print('=' * 18, 'EVVE Dataset', '=' * 18)

        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(
                not_found))
        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos\n'.format(len(all_db - set(self.queries))))
        print('-' * 50)
        ap = []
        for evname in sorted(self.events):
            queries, _, _ = self.events[evname]
            nq = len(queries.intersection(all_db))
            ap.extend(results[evname])
            print('{0: <36} '.format(evname), 'mAP = {:.4f}'.format(
                np.sum(results[evname]) / nq))

        print('=' * 50)
        print('overall mAP = {:.4f}'.format(np.mean(ap)))
