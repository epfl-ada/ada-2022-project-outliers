import pickle
from pprint import pprint

from LDA import LDA, get_file, run_neural_embedding, normalize_output_topics_csv
from gensim.models.wrappers.ldamallet import LdaMallet
import csv
import time
import sys
import os
from multiprocessing import Pool
from NLP import nlp
import matplotlib.pyplot as plt


# path of mallet file, place in root directory
mallet_path = './mallet-2.0.8/bin/mallet'


class MalletLDA(LDA):
    def __init__(self, path, *args):
        if len(args) > 1:
            super().__init__(path, args)
        else:
            super().__init__(path)
        self.name = "mallet"
        self.__alpha = 50
        self.__threshold = 0.0
        self.__optimize_interval = 0

    def set_alpha(self, value):
        self.__alpha = value

    def set_threshold(self, value):
        self.__threshold = value

    def set_interval(self, value):
        self.__optimize_interval = value

    def create_model(self, topics=20, workers=1):
        return LdaMallet(mallet_path, corpus=self.corpus, num_topics=topics, id2word=self.words,
                         workers=workers, alpha=self.__alpha, topic_threshold=self.__threshold,
                         optimize_interval=self.__optimize_interval)

    def load_lda_model(self):
        return get_file("lda-data/mallet-model")

    def final_model(self, topic):
        best_model = None
        best_score = -1
        for i in range(4):
            model = self.create_model(topic)
            score = self.compute_coherence(model).get_coherence()
            if score > best_score:
                best_score = score
                best_model = model
        return (best_score, best_model)


query = "50_1"
docs_num = 30
path = "test/rapid/" + query + "/docs_{}/".format(docs_num)

if __name__ == '__main__':
    clusters_id = [x for x in range(19)]
    for cid in clusters_id:
        path = "title/cid_{}/".format(cid)
        run_neural_embedding(path, filename="labels-", n=6)

    sys.exit(0)
    num_topics = 2
    path = "test/final/" + query + "/"
    if not os.path.exists(path): os.makedirs(path)

    topics = [1, 4]
    # topics = [top for top in range(1, num_topics + 1)]
    nlp(query, docs_num)
    start = time.time()
    mallet = MalletLDA(path)
    with Pool(num_topics) as p:
        res = (p.map(mallet.final_model, topics))

    # print(res)
    best = max(res, key=lambda it: it[0])
    output_csv = normalize_output_topics_csv(best[1])
    for x in res:
        output_csv = normalize_output_topics_csv(x[1])
        pprint(output_csv)
    # sys.exit(0)
    with open(path + "labels-1.csv", "w") as fb:
        writer = csv.writer(fb)
        writer.writerows(output_csv)
    mallet_time = time.time() - start
    print("time for mallet: " + str(mallet_time))

    run_neural_embedding(mallet.path)
    print("time NETL: " + str((time.time() - start - mallet_time)))
    print("total time: " + str((time.time() - start)))
    sys.exit(0)
    model = mallet.create_model_and_save(3)
    tmp = model[mallet.corpus]
    for i, row in enumerate(tmp):
        print(i, row)  # i is doc_num and row is a list of tuples (topic_num, distribution) ...
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        print(row)
        for j, (topic_num, prop_topic) in enumerate(row):
            print(j)
            print(topic_num)
            print(prop_topic)
        break

    mallet.df_topic_sent_keywords_print()
    sys.exit(0)

    # n = 20
    orig_num_topics_start = 3
    orig_num_topics_end = 7
    alphas = [0, 1, 10, 20, 40, 60, 80, 100]
    threshold = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4]
    tmp = 0
    queries = ["32_1", "50_1", "59_2", "75_1"]
    stops = ["32_2", "50_2", "59_3", "75_2"]
    nlp(query, docs_num, True)
    mallet = MalletLDA(path)
    mallet.run_multiple_increasing_topics(6, limit=14)
    # with Pool(len(alphas)) as p:
    #     p.map(simulation.increasing_topics_set_alpha, alphas)
