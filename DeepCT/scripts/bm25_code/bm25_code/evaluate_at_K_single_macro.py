import os
import argparse

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
#         for query_case_id in tqdm(similarity_df.query_case_id.values):

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

#                 candidate_docs = list(similarity_df.columns)[1:]
#                 similarity_scores = similarity_df[
#                     similarity_df["query_case_id"].values == query_case_id
#                 ].values[0][1:]

#                 sorted_candidates = [
#                     x
#                     for _, x in sorted(
#                         zip(similarity_scores, candidate_docs),
#                         key=lambda pair: pair[0],
#                         reverse=True,
#                     )
#                 ]
#                 sorted_candidates.remove(str(query_case_id))
#                 # print(query_case_id, sorted_candidates[:10])

#                 precision_at_K, recall_at_K, f1_at_K = get_scores_at_K(
#                     actual=actual, predicted=sorted_candidates, k=k
#                 )
#                 precision_scores.append(precision_at_K)
#                 recall_scores.append(recall_at_K)
#                 f1_scores.append(f1_at_K)
#         recall_vs_K.append(np.average(recall_scores))
#         precision_vs_K.append(np.average(precision_scores))
#         f1_vs_K.append(np.average(f1_scores))

#     print("Precision @ K:")
#     print(precision_vs_K)

#     print("Recall @ K:")
#     print(recall_vs_K)

#     print("F1 @ K:")
#     print(f1_vs_K)


def get_micro_scores_at_K(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])

    number_of_correctly_retrieved = len(act_set & pred_set)
    number_of_relevant_cases = len(act_set)
    number_of_retrieved_cases = k

    return (
        number_of_correctly_retrieved,
        number_of_relevant_cases,
        number_of_retrieved_cases,
    )


def get_f1_vs_K(gold_labels_csv, predicted_similarity_scores_csv):
    gold_labels = pd.read_csv(gold_labels_csv)

    similarity_df = pd.read_csv(predicted_similarity_scores_csv)
    del_columns = [i for i in similarity_df.columns if i.startswith('Unnamed')]
    similarity_df = similarity_df.drop(columns=del_columns, axis = 1)
    precision_vs_K = []
    recall_vs_K = []
    f1_vs_K = []
    for k in (range(1, 21)):
        precision_scores = []
        recall_scores = []
        f1_scores = []
        number_of_correctly_retrieved_all = []
        number_of_relevant_cases_all = []
        number_of_retrieved_cases_all = []
        for query_case_id in (similarity_df.query_case_id.values):
            if (
                query_case_id
                not in [
                    1864396,
                    1508893,
                ]
                # and len("AILA_Q10") == len(query_case_id)
                # and query_case_id != "AILA_Q10"
            ):  # remove queries with no citations @kiran

                gold = gold_labels[
                    gold_labels["query_case_id"].values == query_case_id
                ].values[0][1:]
                actual = np.asarray(list(gold_labels.columns)[1:])[
                    np.logical_or(gold == 1, gold == -2)
                ]

                candidate_docs = list(similarity_df.columns)[1:]

                similarity_scores = similarity_df[
                    similarity_df["query_case_id"].values == query_case_id
                ].values[0][1:]

                # print(similarity_df)
                # print(len(similarity_scores), len(candidate_docs))
                assert(len(similarity_scores) == len(candidate_docs))

                sorted_candidates = [
                    x
                    for _, x in sorted(
                        zip(similarity_scores, candidate_docs),
                        key=lambda pair: float(pair[0]),
                        reverse=True,
                    )
                ]
                try:
                    sorted_candidates.remove(str(query_case_id))
                except:
                    print("processing AILA", query_case_id)
                # print(query_case_id, sorted_candidates[:10])
                # print(len(sorted_candidates), sorted_candidates)

                (
                    number_of_correctly_retrieved,
                    number_of_relevant_cases,
                    number_of_retrieved_cases,
                ) = get_micro_scores_at_K(
                    actual=actual, predicted=sorted_candidates, k=k
                )
                number_of_correctly_retrieved_all.append(number_of_correctly_retrieved)
                number_of_relevant_cases_all.append(number_of_relevant_cases)
                number_of_retrieved_cases_all.append(number_of_retrieved_cases)

                precision_at_K, recall_at_K, f1_at_K = get_scores_at_K(
                    actual=actual, predicted=sorted_candidates, k=k
                )
                precision_scores.append(precision_at_K)
                recall_scores.append(recall_at_K)
                f1_scores.append(f1_at_K)
        recall_vs_K.append(np.average(recall_scores))
        precision_vs_K.append(np.average(precision_scores))
        f1_vs_K.append(np.average(f1_scores))

        # recall_scores = np.sum(number_of_correctly_retrieved_all) / np.sum(
        #     number_of_retrieved_cases_all
        # )
        # precision_scores = np.sum(number_of_correctly_retrieved_all) / np.sum(
        #     number_of_relevant_cases_all
        # )
        # f1_scores = (2 * precision_scores * recall_scores) / (
        #     precision_scores + recall_scores
        # )

        # recall_vs_K.append(recall_scores)
        # precision_vs_K.append(precision_scores)
        # f1_vs_K.append(f1_scores)

    print("Precision @ K:")
    print(precision_vs_K)

    print("Recall @ K:")
    print(recall_vs_K)

    print("F1 @ K:")
    print(f1_vs_K)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="get_f1_vs_K.py")
    parser.add_argument(
        "--gold_labels_csv",
        type=str,
        # required=True,
        help="path for gold label csv file",
    )
    parser.add_argument(
        "--predicted_similarity_scores_csv",
        type=str,
        # required=True,
        help="path for predicted similarity scores csv file",
    )

    args = parser.parse_args()
    # split = "test"
    # args.gold_labels_csv = "./data/gold_citation_labels/" + split + "_gold_labels.csv"
    # args.predicted_similarity_scores_csv = (
    #     "./data/gold_citation_labels/" + split + "_gold_labels.csv"
    # )
    # args.gold_labels_csv = (
    #     "/home/abhinav/dev/contrastivePCR/data/COLIEE/COLIEE_train_gold_labels.csv"
    # )
    # args.predicted_similarity_scores_csv = "/home/abhinav/dev/contrastivePCR/similarity_predictions/COLIEE/COLIEE_train_IOU_sim.csv"
    # # args.predicted_similarity_scores_csv = "/home/abhinav/dev/contrastivePCR/similarity_predictions/distil.csv"

    # args.gold_labels_csv = (
    #     "/home/abhinav/dev/contrastivePCR/data/COLIEE-22/COLIEE22_test_gold_labels.csv"
    # )
    # args.predicted_similarity_scores_csv = "/home/abhinav/dev/contrastivePCR/similarity_predictions/COLIEE-22/COLIEE22_test_IOU_sim.csv"

    # args.gold_labels_csv = "./AILA_gold.csv"
    # args.predicted_similarity_scores_csv = "./AILA_prediction.csv"
    # args.predicted_similarity_scores_csv = "./AILA_prediction_test.csv"

    get_micro_f1_vs_K(
        gold_labels_csv=args.gold_labels_csv,
        predicted_similarity_scores_csv=args.predicted_similarity_scores_csv,
    )
