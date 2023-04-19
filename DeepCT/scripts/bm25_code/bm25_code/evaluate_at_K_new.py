import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

def get_micro_scores_at_K(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    
    number_of_correctly_retrieved = len(act_set & pred_set) 
    number_of_relevant_cases = len(act_set)
    number_of_retrieved_cases = k

    return number_of_correctly_retrieved, number_of_relevant_cases, number_of_retrieved_cases


# def precision(actual, predicted, k):
#     act_set = set(actual)
#     pred_set = set(predicted[:k])
#     result = len(act_set & pred_set) / float(k)
#     return result


# def recall(actual, predicted, k):
#     act_set = set(actual)
#     pred_set = set(predicted[:k])
#     if float(len(act_set)) != 0:
#         result = len(act_set & pred_set) / float(len(act_set))
#     else:
#         result = 0
#     return result


# def get_scores_at_K(actual, predicted, k):
#     precision_at_K = precision(actual, predicted, k)
#     recall_at_K = recall(actual, predicted, k)
#     F1_at_K = 0
#     if precision_at_K > 0 or recall_at_K > 0:
#         F1_at_K = (2 * precision_at_K * recall_at_K) / (precision_at_K + recall_at_K)

#     return precision_at_K, recall_at_K, F1_at_K


# def get_f1_vs_K(gold_labels_csv, predicted_similarity_scores_csv):
#     gold_labels = pd.read_csv(gold_labels_csv)

#     similarity_df = pd.read_csv(predicted_similarity_scores_csv)
#     precision_vs_K = []
#     recall_vs_K = []
#     f1_vs_K = []
#     for k in tqdm(range(1, 21)):
#         precision_scores = []
#         recall_scores = []
#         f1_scores = []
#         # for query_case_id in tqdm(similarity_df.query_case_id.values):
#         for query_case_id in tqdm(list(gold_labels["query_case_id"].values)):
#             # print(f"\nquery_case_id is {query_case_id}")

#             if query_case_id not in [
#                 1864396,
#                 1508893,
#             ]:  # remove queries with no citations @kiran
#                 gold = gold_labels[
#                     gold_labels["query_case_id"].values == query_case_id
#                 ].values[0][1:]
#                 actual = np.asarray(list(gold_labels.columns)[1:])[
#                     np.logical_or(gold == 1, gold == -2)
#                 ]
#                 actual = [str(i) for i in actual]

#                 # candidate_docs = list(gold_labels.columns.values)
#                 candidate_docs = [int(i) for i in gold_labels.columns.values[1:]]
#                 column_name = 'query_case_id' if 'query_case_id' in similarity_df.columns else 'Unnamed: 0'
#                 similarity_scores = similarity_df[
#                     similarity_df[column_name].values == query_case_id
#                 ].values[0][1:]
#                 similarity_scores = similarity_scores[1:] # hacky cleanup for missing columns

#                 assert(len(similarity_scores) == len(candidate_docs))

#                 sorted_candidates = [
#                     x
#                     for _, x in sorted(
#                         zip(similarity_scores, candidate_docs),
#                         key=lambda pair: float(pair[0]),
#                         reverse=True,
#                     )
#                 ]

#                 sorted_candidates.remove((query_case_id))
#                 sorted_candidates = [str(i) for i in sorted_candidates]

#                 # print(actual[:10])
#                 # print([str(i) for i in sorted_candidates[:10]])
#                 # print(similarity_scores)
#                 # print(candidate_docs[:10])
#                 # print(sorted_candidates[:10])

#                 precision_at_K, recall_at_K, f1_at_K = get_scores_at_K(
#                     actual=actual, predicted=sorted_candidates, k=k
#                 )
#                 precision_scores.append(precision_at_K)
#                 recall_scores.append(recall_at_K)
#                 f1_scores.append(f1_at_K)
#         recall_vs_K.append(np.average(recall_scores))
#         precision_vs_K.append(np.average(precision_scores))
#         f1_vs_K.append(np.average(f1_scores))

#     return {
#         "recall_vs_K" : recall_vs_K, 
#         "precision_vs_K" : precision_vs_K, 
#         "f1_vs_K" : f1_vs_K
#     }

def get_f1_vs_K(gold_labels_csv, predicted_similarity_scores_csv):
    gold_labels = pd.read_csv(gold_labels_csv)
    gold_labels = gold_labels.drop(gold_labels.columns[0], axis=1)
    similarity_df = pd.read_csv(predicted_similarity_scores_csv)
    precision_vs_K = []
    recall_vs_K = []
    f1_vs_K = []
    for k in tqdm(range(1, 20)):
        precision_scores = []
        recall_scores = []
        f1_scores = []
        number_of_correctly_retrieved_all = []
        number_of_relevant_cases_all = []
        number_of_retrieved_cases_all = []
        for query_case_id in similarity_df.query_case_id.values:
            if query_case_id not in [
                1864396,
                1508893,
            ] :
                gold = gold_labels[
                    gold_labels["query_case_id"].values == query_case_id
                ].values[0][1:]
                actual = np.asarray(list(gold_labels.columns)[1:])[
                    np.logical_or(gold == 1, gold == -2)
                ]
                actual = [str(i) for i in actual]

                # candidate_docs = list(similarity_df.columns)[1:]
                candidate_docs = [int(i) for i in gold_labels.columns.values[1:]]
                column_name = 'query_case_id' if 'query_case_id' in similarity_df.columns else 'Unnamed: 0'
                similarity_scores = similarity_df[
                    similarity_df[column_name].values == query_case_id
                ].values[0][1:]
                similarity_scores = similarity_scores[1:]
                assert(len(similarity_scores) == len(candidate_docs))

                sorted_candidates = [
                    x
                    for _, x in sorted(
                        zip(similarity_scores, candidate_docs),
                        key=lambda pair: float(pair[0]),
                        reverse=True,
                    )
                ]
                sorted_candidates.remove((query_case_id))
                sorted_candidates = [str(i) for i in sorted_candidates]

                number_of_correctly_retrieved, number_of_relevant_cases, number_of_retrieved_cases = get_micro_scores_at_K(actual=actual, predicted=sorted_candidates, k=k)
                number_of_correctly_retrieved_all.append(number_of_correctly_retrieved)
                number_of_relevant_cases_all.append(number_of_relevant_cases)
                number_of_retrieved_cases_all.append(number_of_retrieved_cases) 

        recall_scores = np.sum(number_of_correctly_retrieved_all)/np.sum(number_of_retrieved_cases_all)
        precision_scores = np.sum(number_of_correctly_retrieved_all)/np.sum(number_of_relevant_cases_all)
        if recall_scores == 0 or precision_scores == 0:
            f1_scores = 0
        else :    
            f1_scores = (2*precision_scores*recall_scores)/(precision_scores+recall_scores)

        recall_vs_K.append(recall_scores)
        precision_vs_K.append(precision_scores)
        f1_vs_K.append(f1_scores)
    
    return {
        "recall_vs_K" : recall_vs_K, 
        "precision_vs_K" : precision_vs_K, 
        "f1_vs_K" : f1_vs_K
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="get_f1_vs_K.py")
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="path for gold label csv file",
    )
    parser.add_argument(
        "--sim",
        type=str,
        required=True,
        help="path for predicted similarity scores csv file",
    )
    # '/workspace/tejas/PCR/DeepCT/scripts/bm25_code/bm25_code/exp_results/EXP_Mon_Mar_20_15:39:32_2023_901/filled_similarity_matrix.csv'
    args = parser.parse_args()

    output_numbers = get_f1_vs_K(
        gold_labels_csv=args.gold,
        predicted_similarity_scores_csv=args.sim,
    )
    with open(f'output6_uni.json', 'w') as f:
        json.dump(output_numbers, f, indent = 4)
