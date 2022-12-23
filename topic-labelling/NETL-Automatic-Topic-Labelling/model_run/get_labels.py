"""
Author:         Shraey Bhatia
Date:           October 2016
File:           get_labels.py

Updated by:     Sihwa Park
Date:           January 7, 2019
Fix:            Updated to work with Python 3.6.5

It is the script to generate candidate labels, unsupervised best labels and labels from SVM ranker supervised model.
Update the parameters in this script.
Also after you download the files mentioned in readme and if you keep them in different path change it over here.

Parameters:
-cg To generate candidate labels.
-us To get unsupervised labels.
-s To get supervised labels.
Ideally should first use -cg to get candidate label file before going for unsupervised or supervised model. But can be used directly if
you already have your candidate label file for the topics.
Eamample for topics given in toy_data/toytopics.csv
"""

import os
import argparse

parser = argparse.ArgumentParser()

# Common Parameters
# data = "toy_data/toytopics.csv"  # The file in csv format which contains the topic terms that needs a label.
# data = "data/run.csv"  # The file in csv format which contains the topic terms that needs a label.

# Parameters for candidate Generation of Labels
doc2vecmodel = "pre_trained_models/doc2vec/docvecmodel.d2v"  # Path for Doc2vec Model.
word2vecmodel = "pre_trained_models/word2vec/word2vec"  # Path for Word2vec Model.
num_candidates = 19  # Number of candidates labels that need to generated for a topic.
output_filename = "output_candidates"  # Name of the output file that will store the candidate labels.
doc2vec_indices_file = "support_files/doc2vec_indices"  # The filtered doc2vec indices file.
word2vec_indices_file = "support_files/word2vec_indices"  # The filtered word2vec indices file
parser.add_argument("-cg", "--candidates", help="get candidate labels", action="store_true")

# Unsupevised
num_unsup_labels = 3
cand_gen_output = "output_candidates"  # The file which contains candiate generation output.Also used to get supervised output
out_unsup = "output_unsupervised"  # The Output File name for unsupervised labels
parser.add_argument("-us", "--unsupervised", help="get unsupervised labels", action="store_true")

# supervised parameters
num_sup_labels = 3  # Number of supervised labels needed. Should be less than the candidate labels.
pagerank_model = "./support_files/pagerank-titles-sorted.txt"  # This is precomputed pagerank model needed to genrate pagerank features.
svm_classify = "./support_files/svm_rank_classify"  # SVM rank classify. After you download SVM Ranker classify gibve the path of svm_rank_classify here
pretrained_svm_model = "./support_files/svm_model"  # This is trained supervised model on the whole our dataset. Run train train_svm_model.py if you want a new model on different dataset.
out_sup = "./output_supervised"  # The output file for supervised labels.
parser.add_argument("-s", "--supervised", help="get supervised labels", action="store_true")

parser.add_argument("-ocg", dest="output_cand", action="store", default="output_candidates")
parser.add_argument("-ouns", dest="output_uns", action="store", default="output_unsupervised")
parser.add_argument("-osup", dest="output_sup", action="store", default="output_supervised")
parser.add_argument("-d", dest="data", action="store", default="data/run.csv")

args = parser.parse_args()
if args.candidates:  # It calls cand_generation python file to get labels in unsupervised way
    query1 = "~/.conda/envs/AI2020BsC/bin/python cand_generation.py " + str(
        num_candidates) + " " + doc2vecmodel + " " + word2vecmodel + " " + args.data + " " + args.output_cand + " " + doc2vec_indices_file + " " + word2vec_indices_file
    print("Extracting candidate labels")
    os.system(query1)

if args.unsupervised:  # It calls unsupervised_labels python file to get labels in unsupervised way
    query2 = "~/.conda/envs/AI2020BsC/bin/python unsupervised_labels.py " + str(
        num_unsup_labels) + " " + args.data + " " + args.output_cand + " " + args.output_uns
    print("Executing Unsupervised model")
    os.system(query2)

if args.supervised:  # It calls supervised_labels python file to get labels in supervised way.
    query3 = "~/.conda/envs/AI2020BsC/bin/python supervised_labels.py " + str(
        num_sup_labels) + " " + pagerank_model + " " + args.data + " " + args.output_cand + " " + svm_classify + " " + pretrained_svm_model + " " + args.output_sup
    print("Executing Supervised Model")
    os.system(query3)
