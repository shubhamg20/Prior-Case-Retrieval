with open('/workspace/tejas/PCR/DeepCT/data/PCR_UCREAT/test_new.tsv') as dataset_file, open('/workspace/tejas/PCR/DeepCT/collections_new/test_results.tsv') as prediction_file:
    for l1, l2 in zip(dataset_file, prediction_file):
        print("vsdf")
        n, e, a = 0, 0, 0
        counter = 0
        print(dataset_file)
        print('here')
        print(l1.split('\t')[0])
        counter=counter+1
        if(counter == 10):
            break