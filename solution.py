from __future__ import print_function

import argparse
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

pd.set_option('display.max_colwidth', -1)

import re
import matplotlib.pyplot as plt
import sys

sys.path.append('./chinese-whispers-python')
import networkx as nx
import itertools as it


class SenseGraph:
    @staticmethod
    def load_word2vec(from_file=True):
        if not from_file:
            import gensim.downloader as api
            model_name = "word2vec-ruscorpora-300"
            model = api.load(model_name)
        else:
            from gensim.test.utils import datapath
            from gensim.models import KeyedVectors
            model_f = '/Users/ildarnurgaliev/gensim-data/word2vec-ruscorpora-300/word2vec-ruscorpora-300.gz'
            model = KeyedVectors.load_word2vec_format(datapath(model_f), binary=True)
        return model

    # Best: threshold 0.69, ckuster_type = 2, metric = ?, k =9, diff_th = 0.17
    def __init__(self, N=200, model=None, debug=False) -> None:
        self.model = self.load_word2vec() if not model else model
        self.N = N
        self.threshold_g = 0.68
        self.debug = debug
        self.word_clusters_map = dict()
        self.is_avg_emb = True
        self.cluster_type = 3
        self.diff_th = 0.2

        self.metric = self.cos_sim
        self.weight = np.array([0.5054758225891177, 0.21444731010852391, 0.29806156073853607, 0.9254550498748371,
                                0.8774148542375415, 0.6353913887524003, 0.717273600069991, 0.8052905943878901,
                                0.5165530756684633])

        self.k = 9
        self.min_clust = 5
        self.min_dist_dbscan = 0.78
        # Fluid cluster
        self.is_k_depends_g = False
        if self.cluster_type == 3:
            self.threshold_g = 0.38

    def cos_sim(self, word_emb, x):
        return self.model.cosine_similarities(word_emb, [x])[0]

    def prob_sim(self, avg_context_emb, sense_emb):
        return 1 / (1 + np.exp(-np.dot(avg_context_emb, sense_emb)))

    def top_k(self, word_pos):
        words = self.model.similar_by_vector(self.emb(word_pos), topn=self.N, restrict_vocab=None)
        if self.debug: print('top_k', len(words), 'TOP', words[:2], 'BOTTOM', words[-2:])
        return [_[0] for _ in words]

    @lru_cache(maxsize=None)
    def emb(self, word):
        return self.model.wv[word]

    def gen_pairs(self, words):
        pairs = []
        N = len(words)
        for a, b in list(it.combinations(words, 2)):
            sim = self.model.similarity(a, b)
            if sim > self.threshold_g:
                pairs.append((a, b, sim))
        if self.debug:
            print(len(pairs), 'cut-off {}'.format(N * (N - 1) // 2 - len(pairs)), pairs[:2], pairs[-2:])
        return pairs

    def get_cluster_emb(self, word_pos):
        if word_pos not in self.word_clusters_map:
            clusters = self.create_g_cluster(word_pos)
            # if len(clusters) > 15:
            #     prev_clusters = clusters
            #     prev_th = self.threshold
            #     self.threshold = 0.71
            #     clusters = self.create_g_cluster(word_pos)
            #     self.threshold = prev_th
            #     if len(clusters) <= 1:
            #         print('Return cluster')
            #         clusters = prev_clusters

            cluster_emb_list = self.cluster2emb(clusters, word_pos)
            self.word_clusters_map[word_pos] = cluster_emb_list
        return self.word_clusters_map[word_pos]

    def cluster2emb(self, clusters, word_pos):
        emb_cluster = []

        for words in clusters:
            emb_list = [self.emb(w) for w in words]
            word_emb = self.emb(word_pos)
            if self.is_avg_emb:
                emb = np.average(emb_list, axis=0)
            else:
                coef_list = [self.cos_sim(word_emb, _) for _ in emb_list]
                emb_list = np.array(emb_list)
                coef_list = np.array(coef_list).reshape(len(coef_list), 1)
                emb = np.sum(emb_list * coef_list, axis=0) / np.sum(coef_list, axis=0)
            assert emb.shape == (300,)
            emb_cluster.append(emb)
        emb_cluster = sorted(emb_cluster, key=lambda _: self.cos_sim(word_emb, _), reverse=True)
        return emb_cluster

    def create_g_cluster(self, word_pos):
        words = self.top_k(word_pos)[1:]

        if self.cluster_type < 4:
            pairs = self.gen_pairs(words)
            G = nx.Graph()
            G.add_weighted_edges_from(pairs)

        if self.cluster_type == 3:
            G = max(nx.connected_component_subgraphs(G), key=len)
            print('len_strip(G)', len(G))

        if self.cluster_type == 1:
            from networkx.algorithms.community import greedy_modularity_communities
            clusters = list(greedy_modularity_communities(G))
        elif self.cluster_type == 2:
            from chinese_whispers import chinese_whispers, aggregate_clusters
            chinese_whispers(G, iterations=20, weighting='log', seed=13)  # top, nolog, log
            clusters = aggregate_clusters(G).values()
        elif self.cluster_type == 3:
            from networkx.algorithms.community import asyn_fluidc
            if self.is_k_depends_g:
                clusters = list(asyn_fluidc(G, k=self.k - int((self.k - 8) * ((200 - len(G)) / 100))))
            else:
                clusters = list(asyn_fluidc(G, k=min(self.k, len(G))))
        elif self.cluster_type == 4:
            from collections import defaultdict
            from sklearn.cluster import KMeans

            X = [sg.emb(_) for _ in words[1:]]
            clusters = defaultdict(list)

            kmeans = KMeans(n_clusters=self.k, random_state=13)
            assigned_clusters = kmeans.fit_predict(X)

            for cl, w in zip(assigned_clusters, words): clusters[cl].append(w)
            clusters = list(clusters.values())
        elif self.cluster_type == 5:
            from collections import defaultdict
            from sklearn.cluster import DBSCAN

            X = [sg.emb(_) for _ in words[1:]]
            clusters = defaultdict(list)

            dbscan = DBSCAN(metric='l2', eps=self.min_dist_dbscan, min_samples=self.min_clust)
            assigned_clusters = dbscan.fit_predict(X)

            for cl, w in zip(assigned_clusters, words): clusters[cl].append(w)
            clusters = list(clusters.values())
        else:
            raise Exception('no cluster type', self.cluster_type)

        if self.debug:
            for i, cluster in enumerate(sorted(clusters, key=lambda e: len(e), reverse=True)):
                print('Cluster ID\tCluster Elements\n')
                print('{}\t{}\n'.format(i, cluster))
        print(word_pos, 'clusters', len(clusters))

        return clusters

    def select_word_pos(self, contextmorph, tags=('NOUN', 'VERB', 'ADJ', 'ADV', 'ADV', 'ADJ', 'DET', 'ADJ', 'SCONJ',
                                                  'INTJ', 'NUM', 'PART', 'ADP', 'PRON', 'X')):
        ctx = []
        for o in contextmorph.split():
            if o in self.model.wv:
                ctx.append(o)
                continue
            if 'NUM' in o:
                try:
                    num = int(o.split('_')[0])
                    if 900 < num and num <= 2020:
                        ctx.append('год_NOUN')
                    else:
                        # ctx.append('число_NOUN')
                        ctx.append('цифра_NOUN')
                except:
                    ctx.append('количество_NOUN')
            s = o.split('_')[0]
            for t in tags:
                t = s + '_' + t
                if t in sg.model.wv:
                    ctx.append(t)
        return ctx

    def disambiguate(self, wordpos, contextmorph):
        if type(contextmorph) != str:
            return 0, (np.zeros(self.k), np.zeros(self.k)) if self.cluster_type == 3 else (None, None)
        sense_embs = self.get_cluster_emb(wordpos)
        sense_sim_matrix = []

        contextmorph = self.select_word_pos(contextmorph)
        if not contextmorph:
            contextmorph = [wordpos]

        for emb in sense_embs:
            s = [self.cos_sim(self.emb(w), emb) for w in contextmorph]
            sense_sim_matrix.append(s)
        mat = np.array(sense_sim_matrix).reshape((len(sense_embs), -1))
        diff = np.abs(mat.max(axis=0) - mat.min(axis=0))
        select_idx = diff >= self.diff_th

        i = 1
        while not select_idx.any():
            select_idx = diff >= self.diff_th - i * 0.02
            i += 1
        # sense_emb & context_emb similarity
        ctx_emb_avg = np.average([self.emb(contextmorph[i]) for i in np.where(select_idx)[0]], axis=0)

        mat_new = [self.metric(ctx_emb_avg, e) for e in sense_embs]
        if self.cluster_type == 3:
            res = np.argmax(self.weight * mat_new)
        else:
            res = np.argmax(mat_new)

        if self.cluster_type == 3:
            cos_features = [self.cos_sim(ctx_emb_avg, e) for e in sense_embs]
            prob_features = [self.prob_sim(ctx_emb_avg, e) for e in sense_embs]
            features = (cos_features, prob_features)
        else:
            features = (None, None)
        return res, features

    def visualize_g(self, G):
        # Visualize the clustering of G using NetworkX (requires matplotlib)
        colors = [1. / G.node[node]['label'] for node in G.nodes()]
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        nx.draw_networkx(G, cmap=plt.get_cmap('jet'), node_color=colors, font_color='white', with_labels=False)
        plt.show()


def prepare_df(input_f):
    df = read_csv(input_f, sep='\t')

    def cut_context(text, spans):
        try:
            spans = tuple(map(int, re.split(r'[,-]', spans)))
            spans = [spans[i:i + 2] for i in range(0, len(spans), 2)]

            contex_buf = []
            prev_i = 0
            for l, r in spans:
                contex_buf.append(text[prev_i:l])
                contex_buf.append(' ')
                prev_i = r
            contex_buf.append(text[prev_i:])
            text = ''.join(contex_buf)
        except:
            pass
        return text.strip()

    df['context_wo'] = df.apply(lambda row: cut_context(row['context'], row['positions']), axis=1)

    df = prepare_PoS_morph(df, 'context_wo')
    df.to_csv(args.input_pos, sep='\t', index=None)


def process(args, sg):
    df = read_csv(args.input_pos, sep='\t')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)

        prediction_list = [sg.disambiguate(wordpos, contextpos)
                           for wordpos, contextpos in tqdm(df[['wordpos', 'contextmorph']].values)]
        df['predict_sense_id'] = [_[0] for _ in prediction_list]
        features = [_[1] for _ in prediction_list]
        df['feature_cos'] = [_[0] for _ in features]
        df['feature_prob'] = [_[1] for _ in features]

    if df['gold_sense_id'].any():
        per_word = df.groupby('word').aggregate(
            lambda f: adjusted_rand_score(
                f['gold_sense_id'], f['predict_sense_id']))
        per_word_ari = per_word['predict_sense_id']
        if args.ari_per_word:
            for word, ari in zip(per_word.index, per_word_ari):
                print('{:<20} {:+.4f}'.format(word, ari))
        print('Mean word ARI: {:.4f}'.format(np.mean(per_word_ari)))

    if args.output:
        print('Saving predictions to {}'.format(args.output))
        if 'solution' in args.output:
            cols_to_save = ['context_id', 'word', 'gold_sense_id', 'predict_sense_id', 'positions', 'context']
            df[cols_to_save].to_csv(args.output, sep='\t', index=None)
        else:
            df.to_csv(args.output, sep='\t', index=None)
    return np.mean(per_word_ari)


# ToDo test over ud-pipe
def prepare_PoS_udpipe(df):
    from ufal.udpipe import Model, Pipeline
    model_f = "/Users/ildarnurgaliev/projects/pretrained/udpipe/russian-syntagrus-ud-2.4-190531.udpipe"
    model = Model.load(model_f)
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

    def tag(text):
        text = text.translate({ord(_): '' for _ in "«»\"'"})
        processed = pipeline.process(text)
        output = [l for l in processed.split('\n') if not l.startswith('#')]
        tagged = ['_'.join(w.split('\t')[2:4]) for w in output if w and 'PUNCT' not in w]
        return ' '.join(tagged)

    df['contextudpipe'] = df['context'].apply(tag)
    return df


def prepare_PoS_morph(df, context_key):
    from pymystem3 import Mystem
    m = Mystem()

    tags = """
    A       ADJ                                                                                                                                                                                                                                                                    
    ADV     ADV                                                                                                                                                                                                                                                                    
    ADVPRO  ADV                                                                                                                                                                                                                                                                    
    ANUM    ADJ                                                                                                                                                                                                                                                                    
    APRO    DET                                                                                                                                                                                                                                                                    
    COM     ADJ                                                                                                                                                                                                                                                                    
    CONJ    SCONJ                                                                                                                                                                                                                                                                  
    INTJ    INTJ                                                                                                                                                                                                                                                                   
    NONLEX  X                                                                                                                                                                                                                                                                      
    NUM     NUM                                                                                                                                                                                                                                                                    
    PART    PART                                                                                                                                                                                                                                                                   
    PR      ADP                                                                                                                                                                                                                                                                    
    S       NOUN                                                                                                                                                                                                                                                                   
    SPRO    PRON                                                                                                                                                                                                                                                                   
    UNKN    X                                                                                                                                                                                                                                                                      
    V       VERB
    """.strip().split()
    tag_mapping = dict([tags[i:i + 2] for i in range(0, len(tags), 2)])
    pat = re.compile(r'\b(?:[а-я]+|\d+)', re.IGNORECASE | re.UNICODE)
    word_mapping = {
        'млн': 'миллион'
    }

    def tag(word):
        word = re.sub(r'[^\w\s]', '', word)
        word = word_mapping.get(word, word)
        if word.isdigit():
            return word + '_' + 'NUM'
        processed = m.analyze(word)[0]
        # print(word, processed)
        try:
            lemma = processed["analysis"][0]["lex"].lower().strip()
            pos = processed["analysis"][0]["gr"].split(',')[0]
            pos = pos.split('=')[0].strip()
            tagged = lemma + '_' + tag_mapping[pos]
        except:
            tagged = word + '_' + ' NOUN'
        return tagged

    df['contextmorph'] = df[context_key].apply(lambda text: ' '.join([tag(_) for _ in pat.findall(text)]))

    df['wordpos'] = df['word'].apply(tag)

    return df


def select_diff_th(args, sg):
    best_r = 0
    metric = 0
    # for th in it.chain(np.arange(0.0, 0.21, 0.01), np.arange(0.3, 0.91, 0.1)):
    for th in np.arange(0.0, 0.41, 0.01):
        print('TH', th)
        sg.diff_th = th
        res = process(args, sg)
        if res - best_r > 0.0001:
            best_r = res
            metric = th

    print('Best diff_th={} -> {}'.format(metric, best_r))


def select_G_th(args, sg):
    best_r = 0
    metric = 0
    # for th in np.arange(0.5, 0.7, 0.01):
    for th in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        print('TH', th)
        sg.word_clusters_map = {}
        sg.threshold_g = th
        res = process(args, sg)
        if res - best_r > 0.0001:
            best_r = res
            metric = th

    print('Best G_th={} -> {}'.format(metric, best_r))


def select_dist_dbscan(args, sg):
    best_r = 0
    metric = 0
    for th in np.arange(0.75, 0.8, 0.01):
        print('th_dist_dbscan', th)
        sg.word_clusters_map = {}
        sg.min_dist_dbscan = th
        res = process(args, sg)
        if res - best_r > 0.0001:
            best_r = res
            metric = th

    print('Best th_dist_dbscan={} -> {}'.format(metric, best_r))


def select_k(args, sg):
    best_r = 0
    metric = 0
    for k in range(4, 20):
        print('k', k)
        sg.word_clusters_map = {}
        sg.k = k
        res = process(args, sg)
        if res - best_r > 0.0001:
            best_r = res
            metric = k

    print('Best k={} -> {}'.format(metric, best_r))


def select_N(args, sg):
    best_r = 0
    metric = 0
    for N in range(160, 300, 20):
        print('N', N)
        sg.word_clusters_map = {}
        sg.N = N
        res = process(args, sg)
        if res - best_r > 0.0001:
            best_r = res
            metric = N

    print('Best k={} -> {}'.format(metric, best_r))


def select_clstype(args, sg):
    best_r = 0
    metric = 0
    for cluster_type in range(3, 6):
        print('cluster_type', cluster_type)
        sg.word_clusters_map = {}
        sg.cluster_type = cluster_type
        prev_th = sg.threshold_g
        res = process(args, sg)
        if res - best_r > 0.0001:
            best_r = res
            metric = cluster_type
        sg.threshold_g = prev_th

    print('Best cluster_type={} -> {}'.format(metric, best_r))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    arg = parser.add_argument
    arg('--input', help='Path to input file with contexts', default='data/main/active-dict/train.csv')
    arg('--output', help='Path to output file with predictions', default='data/main/active-dict/train.solution.csv')
    # arg('--input_pos', help='Path to input_pos file', default='data/main/active-dict/train_pos.csv')
    arg('--input_pos', help='Path to input_pos file', default='data/main/active-dict/train_pos_cut.csv')
    arg('--ari-per-word', help='show ARI per-word', default=True)

    # arg('--input', help='Path to input file with contexts', default='data/additional/active-rutenten/train.csv')
    # arg('--output', help='Path to output file with predictions', default='data/additional/active-rutenten/train.solution.csv')
    # arg('--input_pos', help='Path to input_pos file', default='data/additional/active-rutenten/train_pos.csv')
    # arg('--ari-per-word', help='show ARI per-word', default=True)

    args = parser.parse_args()
    print(args)

    prepare_df(args.input)  # generate _pos.csv
    sg = SenseGraph()
    process(args, sg)

    # select_diff_th(args, sg)
    # select_G_th(args, sg)
    # select_k(args, sg)
    # select_dist_dbscan(args, sg)
    # select_N(args, sg)
    # select_clstype(args, sg)
