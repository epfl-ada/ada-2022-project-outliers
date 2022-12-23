import pickle
import sys

from gensim.models import CoherenceModel
from NLP import nlp
from statistics import mean
from pprint import pprint
import pandas as pd
import os
import time

from abc import abstractmethod

import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
import csv
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel

os.environ.update({'MALLET_HOME': r'./mallet-2.0.8/bin/mallet'})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# path of mallet file, place in root directory
mallet_path = './mallet-2.0.8/bin/mallet'


def init_lemmatized():
    return get_file("lda-data/lemmatized")


def init_corpus():
    return get_file("lda-data/corpus")


def init_words():
    return get_file("lda-data/words")


def load_mallet_model():
    return get_file("lda-data/mallet-model")


def load_topics():
    return get_file("lda-data/topic-document")


def init_passages():
    return get_file("lda-data/passages")


def get_file(filename):
    infile = open(filename, 'rb')
    tmp = pickle.load(infile)
    infile.close()
    return tmp


def normalize_output_topics_csv(best):
    topics = best.show_topics(formatted=False, num_topics=best.get_topics().shape[0])  # all topics
    num_terms = len(topics[0][1])
    output = [["topic_id"]]
    for x in range(num_terms):
        if x == num_terms - 1:
            tmp = "term{}".format(x)
        else:
            tmp = "term{}".format(x)
        output[0].append(tmp)
    for i in range(len(topics)):
        # set topics in order, it is important for neural embedding
        line = [i]
        terms = topics[i][1]
        for k in range(num_terms):
            if k == num_terms - 1:
                line.append(str(terms[k][0]))
            else:
                line.append(str(terms[k][0]))
        output.append(line)
    return output


def plot_multiple_models(start, limit, step, coherence_values, name=None, show=False):
    # Show graph
    x = range(start, limit, step)
    plt.plot(x, coherence_values, linestyle='--', marker="o")
    plt.xlabel("Num Topics")
    # plt.xlabel("iteration")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    # max point
    max_value = max(coherence_values)
    xpos = coherence_values.index(max_value)
    x_max = x[xpos]
    # zip joins x and y coordinates in pairs
    for xs, ys in zip(x, coherence_values):
        label = "{}".format(xs)
        if xs == x_max:  # annotate the max value in red
            plt.annotate(x_max, (x_max, max_value), fontsize=9, color='red', textcoords="offset points",
                         xytext=(0, 5), ha='center')
            continue
        plt.annotate(label,  # this is the text
                     (xs, ys),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 8),  # distance from text to points (x,y)
                     ha='center',  # horizontal alignment can be left, right or center
                     fontsize=8)
    if name is not None:
        name += '.png'
        plt.savefig(name)
    if show: plt.show()
    plt.close('all')


# return list of list of terms
def get_formatted_topics(model):
    topics = model.show_topics(formatted=False, num_topics=model.get_topics().shape[0])  # all topics
    formatted_topics = []
    for topic in topics:
        terms = []
        for term in topic[1]:
            terms.append(term[0])
        formatted_topics.append(terms)
    return formatted_topics


#  n-1 should be equal to number of labels.csv files
def run_neural_embedding(path, n=2, filename="labels-", arr=None):
    os.chdir("NETL-Automatic-Topic-Labelling/model_run")
    if arr is not None:
        iter_var = arr
    else:
        iter_var = range(1, n)
    for x in iter_var:
        cand_out = "./../../" + path + "output_candidates_" + filename + str(x)
        unsup_out = "./../../" + path + "output_unsupervised_" + filename + str(x)
        sup_out = "./../../" + path + "output_supervised_" + filename + str(x)
        label_file_name = "./../../" + path + filename + str(x) + ".csv"
        os.system(
            "~/.conda/envs/AI2020BsC/bin/python get_labels.py -cg -us -s -d " + label_file_name + " -ocg " + cand_out + " -ouns " + unsup_out + " -osup " + sup_out)
    os.chdir("./../..")


def prepare_best_data_for_labelling(path):
    mallet_values = get_file(path)
    best_score_pos = mallet_values['values'].index(max(mallet_values['values']))
    best_model = mallet_values['model'][best_score_pos]
    # pprint(best_model.show_topics(formatted=False))
    output_csv = normalize_output_topics_csv(best_model)
    return output_csv


class LDA:
    def __init__(self, path, *args):
        if len(args) > 1:
            self.lemmatized = get_file(args[0])
            self.corpus = get_file(args[1])
            self.words = get_file(args[2])
            self.passages = init_passages()
        else:
            self.lemmatized = get_file("lda-data/lemmatized-cluster")
            self.corpus = get_file("lda-data/corpus-cluster")
            self.words = get_file("lda-data/words-cluster")
            # self.lemmatized = init_lemmatized()
            # self.corpus = init_corpus()
            # self.words = init_words()
            self.passages = init_passages()
        self.name = ""
        self.path = path

    def get_name(self):
        return self.name

    def set_path(self, value):
        self.path = value

    @abstractmethod
    def load_lda_model(self):
        pass

    @abstractmethod
    def create_model(self, topics=20, workers=1):
        pass

    def run_multiple_lda_and_print(self, name=None, topics=20, limit=10, start=1, step=1, fix=0):
        if fix == 0:
            model_list, coherence_values = self.compute_coherence_values_fix(limit, topics)
            fix = 1  # in order to avoid file named data-0, which creates error in future methods
        else:
            model_list, coherence_values = self.compute_coherence_values_increasing(limit, start, step)
        # save coherence score for each model
        x = range(start, limit, step)
        with open(self.path + "data-{}".format(fix), 'wb') as f:
            pickle.dump({'model': model_list, 'model-list': list(x), 'values': coherence_values}, f)

        plot_multiple_models(start, limit, step, coherence_values, name=name)

    # create LDA model for increasing number of topics, from start to limit
    # and compute their coherence score
    # return list of models and respective coherence score
    def compute_coherence_values_increasing(self, limit=20, start=1, step=1):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = self.create_model(num_topics)
            model_list.append(model)
            coherence_model = self.compute_coherence(model)
            coherence_values.append(coherence_model.get_coherence())

        return model_list, coherence_values

    # create LDA model for fix number of topics and compute their coherence score
    # return list of models and respective coherence score
    def compute_coherence_values_fix(self, limit, topics):
        scores = []
        models = []
        for x in range(1, limit):
            model = self.create_model(topics)
            models.append(model)
            scores.append(self.compute_coherence(model).get_coherence())

        return models, scores

    def run_multiple_fix_topic(self, topics, limit=20):
        if not os.path.exists(self.path): os.makedirs(self.path)

        self.run_multiple_lda_and_print(self.path + "img-1", limit=limit, topics=topics)
        output_csv = prepare_best_data_for_labelling(self.path + "data-1")
        with open(self.path + "labels-1.csv", "w") as fb:
            writer = csv.writer(fb)
            writer.writerows(output_csv)

    def run_multiple_increasing_topics(self, n, step=1, limit=16, start=1):
        if not os.path.exists(self.path): os.makedirs(self.path)

        for x in range(1, n):
            self.run_multiple_lda_and_print(self.path + "img-" + str(x), fix=x, limit=limit, start=start, step=step)
            output_csv = prepare_best_data_for_labelling(self.path + "data-" + str(x))
            # in order to find number of topics for best model compute number of rows in labels.csv
            # this file goes to Neural embedding
            with open(self.path + "labels-" + str(x) + ".csv", "w") as fb:
                writer = csv.writer(fb)
                writer.writerows(output_csv)

    def create_model_and_save(self, topics, workers=1):
        tmp = self.create_model(topics, workers)
        out = open("lda-data/" + self.get_name(), "wb")
        pickle.dump(tmp, out)
        out.close()
        return tmp

    def compute_coherence(self, model, workers=1, topics=None):
        return CoherenceModel(model, topics=topics, texts=self.lemmatized, dictionary=self.words, coherence='c_v',
                              topn=10, processes=workers)

    def format_topics_sentences(self):
        # Init output
        sent_topics_df = pd.DataFrame()

        # get the best lda model
        # with open("lda-data/mallet", "rb") as f:
        with open(self.path + "best", "rb") as f:
            ldamodel = pickle.load(f)

        # Get main topic in each document
        corpus = self.corpus  # len is a number of retrieved documents
        passages = self.passages
        # transform mallet model to classic
        if self.name == "mallet": ldamodel = malletmodel2ldamodel(ldamodel)
        for i, row in enumerate(ldamodel[corpus]):
            # row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)  # old line
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), passages[i][0], passages[i][1]]),
                        ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Passage_ID', 'Relevance']

        # Add original text to the end of the output
        texts = self.words
        contents = pd.Series(texts)
        # sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        df_dominant_topic = sent_topics_df.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Passage_ID', 'Relevance']
        df_dominant_topic.to_csv(self.path + "dist.csv")
        return df_dominant_topic

    def df_topic_sent_keywords_print(self, min_relevance=3):
        df_topic_sents_keywords = self.format_topics_sentences()
        rel_docs = df_topic_sents_keywords["Relevance"]
        n_rel_docs = len([el for el in rel_docs if int(el) >= min_relevance])
        # with open(self.path + "dist.csv", "r") as f:
        #     df_topic_sents_keywords = pd.read_csv(f)
        # could happen that no document belongs to certain topic, this provoke one parameter missing. This causes no
        # problem, just an empty space in that case will mean a 0
        grouped_dominant_topic = df_topic_sents_keywords.groupby('Dominant_Topic')

        n_docs = df_topic_sents_keywords.shape[0]
        # tmp.apply(print)
        print("Consider the relevance score >= " + str(min_relevance))
        # first is number of total rel documents, second is number of docs,  third is min_relevance, after the topics relevance score
        values = [n_rel_docs, n_docs, min_relevance]
        for topic, group in grouped_dominant_topic:
            tmp = [el for el in group['Relevance'] if int(el) >= min_relevance]
            print("Relevance of topic {} is {}/{} with total of {} retrieved documents".format(topic, len(tmp),
                                                                                               n_rel_docs, n_docs))
            values.append(len(tmp))

        return values

    def visualize(self, model):
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(model, self.corpus, self.words)
        vis


if __name__ == "__main__":
    etas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    optimizations = [1, 10, 20, 50, 100, 200]
    alphas = [0, 1, 10, 20, 40, 60, 80, 100]
    test_arr = [19, 24, 9, 26]
    path = "test/mallet-test/31_6/fix_4-docs_30/"
    with open(path + "data-1", "rb") as f:
        data = pickle.load(f)

    for v in test_arr:
        output_csv = normalize_output_topics_csv(data["model"][v])
        with open(path + "labels-{}.csv".format(v), "w") as fb:
            writer = csv.writer(fb)
            writer.writerows(output_csv)

    run_neural_embedding(path, arr=test_arr)
    sys.exit(0)

    for i in range(1, 9):
        run_neural_embedding("test/mallet-test/61_1/fix_{}-docs_30/".format(i), 2)
    # for x in range(1, 9):
    #     for opt in optimizations:
    #         run_neural_embedding("test/fix/mallet/31_6/fix_{}-inter_{}-docs_40/".format(x, opt), 2)
    # for x in range(1, 7):
    #     for a in alphas:
    #         run_neural_embedding("test/fix/mallet/68_11/fix_{}-alpha_{}-docs_40/".format(x, a), 2)
