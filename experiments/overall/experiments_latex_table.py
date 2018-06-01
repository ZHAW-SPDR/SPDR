import yaml
import csv

EXPERIMENTS = 'experiments.yml'
RESULTS = './experiments/overall/results.csv'
IN_PATH = './data/in/'

with open(EXPERIMENTS) as exp:
    experiments_ = list(yaml.load_all(exp))
with open(RESULTS) as f:
    csv_ = list(csv.reader(f,delimiter=';')) 

for idx, (experiment_, row) in enumerate(zip(experiments_, csv_)):

    ds = experiment_['data']['in']
    ds_subset = experiment_['data']['datasubset_to_use']
    network = experiment_['embeddings']['network']
    ds = ds.replace(IN_PATH,'').replace('/','')
    network = network.replace('.h5','')
    experiment_name = "DATASET_%s.SUBSET_%s.MODEL_%s" % (ds.lower(), ds_subset.lower(), network.lower())
    
    i = idx+1
    text = (experiment_name.replace('_','\_').replace('.','.\\newline '))

    # print(i, ' & \\multirow{4}{*}{\\parbox{6.6cm}{\\vspace{-0.4cm} ', text,'}} & ','DER',' & ',row[1], ' & ',row[5],' & ',row[9],' \\\\* ')
    # print('\\cline{3-6}')
    # print('', ' & ', '',' & ','Purity',' & ',row[2],' & ',row[6],' & ',row[10], ' \\\\* ')
    # print('\\cline{3-6}')
    # print('', ' & ', '',' & ','Coverage',' & ',row[3],' & ',row[7],' & ',row[11], ' \\\\* ')
    # print('\\cline{3-6}')
    # print('', ' & ', '',' & ','F-Score',' & ',row[4],' & ',row[8],' & ',row[12], ' \\\\ \hline')
    print(i, ' & ', ds.replace('_','\_'), '/', ds_subset.replace('_','\_'), ' & ', network.replace('_','\_'), ' \\\\ \hline')