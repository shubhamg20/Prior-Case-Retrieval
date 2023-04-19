import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result


def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if float(len(act_set)) != 0:
        result = len(act_set & pred_set) / float(len(act_set))
    else:
        result = 1
    return result


def get_scores_at_K(actual, predicted, k):
    precision_at_K = precision(actual, predicted, k)
    recall_at_K = recall(actual, predicted, k)
    F1_at_K = 0
    if precision_at_K > 0 or recall_at_K > 0:
        F1_at_K = (2 * precision_at_K * recall_at_K) / (precision_at_K + recall_at_K)

    return precision_at_K, recall_at_K, F1_at_K


def get_f1_vs_K(gold_labels_csv, predicted_similarity_scores_csv):
    gold_labels = pd.read_csv(gold_labels_csv)
    gold_labels = gold_labels.drop(gold_labels.columns[0], axis=1)
    similarity_df = pd.read_csv(predicted_similarity_scores_csv)
    precision_vs_K = []
    recall_vs_K = []
    f1_vs_K = []
    for k in (range(1, 21)):
        precision_scores = []
        recall_scores = []
        f1_scores = []
        # for query_case_id in tqdm(similarity_df.query_case_id.values):
        for query_case_id in (list(gold_labels["query_case_id"].values)):
            # print(f"\nquery_case_id is {query_case_id}")

            if query_case_id not in [
                1864396,
                1508893,
            ]:  # remove queries with no citations @kiran
                gold = gold_labels[
                    gold_labels["query_case_id"].values == query_case_id
                ].values[0][1:]
                actual = np.asarray(list(gold_labels.columns)[1:])[
                    np.logical_or(gold == 1, gold == -2)
                ]
                actual = [str(i) for i in actual]

                # candidate_docs = list(gold_labels.columns.values)
                candidate_docs = [int(i) for i in gold_labels.columns.values[1:]]
                column_name = 'query_case_id' if 'query_case_id' in similarity_df.columns else 'Unnamed: 0'
                similarity_scores = similarity_df[
                    similarity_df[column_name].values == query_case_id
                ].values[0][1:]
                similarity_scores = similarity_scores[1:] # hacky cleanup for missing columns

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

                # print(actual[:10])
                # print([str(i) for i in sorted_candidates[:10]])
                # print(similarity_scores)
                # print(candidate_docs[:10])
                # print(sorted_candidates[:10])

                precision_at_K, recall_at_K, f1_at_K = get_scores_at_K(
                    actual=actual, predicted=sorted_candidates, k=k
                )
                precision_scores.append(precision_at_K)
                recall_scores.append(recall_at_K)
                f1_scores.append(f1_at_K)
        recall_vs_K.append(np.average(recall_scores))
        precision_vs_K.append(np.average(precision_scores))
        f1_vs_K.append(np.average(f1_scores))

    return {
        "recall_vs_K" : recall_vs_K, 
        "precision_vs_K" : precision_vs_K, 
        "f1_vs_K" : f1_vs_K
    }

 # '/workspace/tejas/PCR/DeepCT/scripts/bm25_code/bm25_code/gold_data.csv'
# '/workspace/tejas/PCR/DeepCT/scripts/bm25_code/bm25_code/exp_results/EXP_Sat_Mar_18_17:49:30_2023_251/filled_similarity_matrix.csv'
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

    args = parser.parse_args()

    output_numbers = get_f1_vs_K(
        gold_labels_csv=args.gold,
        predicted_similarity_scores_csv=args.sim,
    )
    with open(f'output.json', 'w') as f:
        json.dump(output_numbers, f, indent = 4)

