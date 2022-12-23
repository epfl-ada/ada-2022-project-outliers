import multiprocessing

from NLP import nlp
from LDA import get_file, normalize_output_topics_csv, run_neural_embedding, plot_multiple_models
from malletLDA import MalletLDA
import pandas as pd
from multiprocessing import Pool
from statistics import mean
import pickle
import time
from math import ceil
import sys
import csv
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def my_func(data):
    # init the model and compute the score
    topic = data[0]
    path = data[1]
    # todo uncomment
    mallet = MalletLDA(path)
    model = mallet.create_model(topic)
    model_score = mallet.compute_coherence(model).get_coherence()
    # save the best score

    output_csv = normalize_output_topics_csv(model)
    with open(path + "labels-{}.csv".format(topic), "w") as fb:
        writer = csv.writer(fb)
        writer.writerows(output_csv)

    return (model_score, topic)


def generate_topics(path, num_topics):
    # create a folder for each cluster
    if not os.path.exists(path): os.makedirs(path)
    scores = []
    # parallel here
    pool = multiprocessing.Pool(processes=5)
    data = [(x, path) for x in range(1, num_topics)]
    for res in pool.map(my_func, data):
        scores.append(res)

    # sort by topic
    scores.sort(key=lambda x: x[1])
    # extract score of each model sorted by number of topics
    scores = [tup[0] for tup in scores]
    # plot the scores
    plot_multiple_models(1, num_topics, 1, scores, name=path + "plot")


# run neural embedding for 19 clusters for 4 continuous topics
def netl_clusters():
    clusters_id = [x for x in range(19)]
    for cid in clusters_id:
        path = "title/cid_{}/".format(cid)
        run_neural_embedding(path, filename="labels-", n=5)


if __name__ == "__main__":
    # run LDA for 19 clusters
    clusters_id = [x for x in range(19)]
    for c_id in clusters_id:
        path = "title/cid_{}/".format(c_id)
        nlp(c_id)
        generate_topics(path, 11)

    # path = "final/all/"
    # nlp()
    # generate_topics(path, 11)


    sys.exit(0)

    run_neural_embedding(path, filename="cid_{}_labels-", n=19)

    #############
    # set own_data variable to list of strings which represent list of documents
    own_data = None
    nlp(own_data=own_data)
    # set path variable to the local path you desire to store LDA results. slash (/) at the end is needed.
    path = "test/"
    mallet = MalletLDA(path)

    # run only one mallet model with 1 topic
    one_topic_model = mallet.create_model(1)
    one_topic_model_score = mallet.compute_coherence(one_topic_model).get_coherence()
    output_csv = normalize_output_topics_csv(one_topic_model)
    with open(path + "labels-1.csv", "w") as fb:
        writer = csv.writer(fb)
        writer.writerows(output_csv)

    # run NETL
    run_neural_embedding(path, filename="labels-")

    # run 4 LDA models with 5 topics
    mallet.run_multiple_fix_topic(4, 4)
    run_neural_embedding(path)

    mallet_values = get_file(path + "data-1")
    best_score = max(mallet_values['values'])
    print("one topic model score is {} and 4-topic model is {}".format(one_topic_model_score, best_score))
