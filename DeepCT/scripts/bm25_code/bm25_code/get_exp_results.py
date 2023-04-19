import os, sys, glob
import json

def get_train_counterpart(exp_name):
    assert(('test' in exp_name) and ('train' not in exp_name))
    if 'ik_test_iou_filtered' in exp_name:
        return exp_name.replace('ik_test_iou_filtered', 'ik_train_iouf')
    else :
        return exp_name.replace("test", "train")

def get_exp_result(exp_name):
    path = f'./data/corpus/{exp_name}/'
    output_files = {}
    for config_file_path in glob.glob('./exp_results/*/config_file.json'):
        with open(config_file_path, 'r') as f:
            config = json.load(f)
            if config['path_prior_cases'].startswith(path):
                output_dict_path = config_file_path[:config_file_path.rfind('/')] + f'/output.json'
                with open(output_dict_path, 'r') as f2:
                    output_dict = json.load(f2)
                output_files[config['n_gram']] = output_dict

    if 'train' in exp_name:
        for n_gram, res in output_files.items():
            best_k = res['f1_vs_K'].index(max(res['f1_vs_K']))
            # print(best_k, res)
            return_dict = {
                'exp_name' : exp_name,
                'n_gram' : n_gram,
                'best_k' : best_k+1,
                'recall_vs_K' : res['recall_vs_K'][best_k],
                'precision_vs_K' : res['precision_vs_K'][best_k],
                'f1_v_k' : res['f1_vs_K'][best_k],
            }
            EXP_RESULTS.append(return_dict)
    
    
    elif 'test' in exp_name:
        counter_trainname = get_train_counterpart(exp_name)
        for n_gram, res in output_files.items():
            # print(output_files)
            # print(EXP_RESULTS)

            train_entry = [i for i in EXP_RESULTS if ((i['exp_name'] == counter_trainname) and (i['n_gram'] == n_gram))]
            if(len(train_entry) == 0):
                print(f'Train counterpart experiment for {exp_name} : {counter_trainname}, n_gram : {n_gram} not found!\nRerunning with different order of experiments should do the trick.')
            assert(len(train_entry) == 1)
            best_k_train = train_entry[0]['best_k']

            return_dict = {
                'exp_name' : exp_name,
                'n_gram' : n_gram,
                'best_k_train' : best_k_train,
                'recall_vs_K' : res['recall_vs_K'][best_k_train-1], # as it is pushed as + 1
                'precision_vs_K' : res['precision_vs_K'][best_k_train-1],
                'f1_v_k' : res['f1_vs_K'][best_k_train-1],
            }
            EXP_RESULTS.append(return_dict)

def show(i):
    if 'best_k_train' in i:
        print(i)
    
if __name__ == '__main__':
    exp_sentence_removed = ['ik_train', 'ik_test', 
    'ik_train_events', 'ik_train_atomic', 
    'ik_test_events', 'ik_test_atomic',
    'ik_train_iouf', 'ik_test_iou_filtered',
    'RR/ik_train', 'RR/ik_test', ]
    temp = []
    for i in exp_sentence_removed:
        temp.append(f'sentence_removed/{i}')
    exp_sentence_removed = temp

    ############ don't touch above ############

    exp_not_sentence_removed = ['ik_train', 'ik_test', 
    'ik_train_events', 'ik_train_atomic', 
    'ik_test_events', 'ik_test_atomic', 
    'ik_train_iouf', 'ik_test_iou_filtered',
    'RR/ik_train', 'RR/ik_test', ]
    
    exp_coliee = [
        'COLIEE2021_events/train', 
        'COLIEE2021_atomic_events/train', 
        'COLIEE2021_iou_filtered/train',
        'COLIEE2021/train', 
        'COLIEE2021_RR/train',
        'COLIEE2021_events/test', 
        'COLIEE2021_atomic_events/test', 
        'COLIEE2021_iou_filtered/test',
        'COLIEE2021/test', 
        'COLIEE2021_RR/test',
    ]

    exp_all = exp_sentence_removed + exp_not_sentence_removed
    # exp_all = exp_coliee
    EXP_RESULTS = []

    for i in exp_all:
        _ = get_exp_result(i)
        
    for i in EXP_RESULTS:
        show(i)
