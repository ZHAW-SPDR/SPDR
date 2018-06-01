import yaml
import os
import shutil
import pickle
from keras import backend as K
from spdr.metrics import SPDR_Metrics

IN_PATH = './data/in/'
CONFIG = 'config.yml'
#EXPERIMENTS = 'single_experiment.yml'
EXPERIMENTS = 'experiments.yml'
RUN_PATH = './data/runs/'
PICKLE_PATH = './data/pickles/'
OUT_PATH = './data/out/'

with open(EXPERIMENTS) as exp:
    experiments_ = list(yaml.load_all(exp))

for idx, experiment_ in enumerate(experiments_):

    ds = experiment_['data']['in']
    ds_subset = experiment_['data']['datasubset_to_use']
    network = experiment_['embeddings']['network']
    ds = ds.replace(IN_PATH,'').replace('/','')
    network = network.replace('.h5','')
    experiment_name = "DATASET_%s.SUBSET_%s.MODEL_%s" % (ds.lower(), ds_subset.lower(), network.lower())
    run_folder = os.path.join(RUN_PATH, experiment_name)
    print('='*20, 'Plotting experiment',idx+1,'of',len(experiments_), '='*20)
    print('-'*10, 'Name:', experiment_name, '-'*10)

    # Load
    RESULTS_PICKLE = os.path.join(run_folder, 'result.pkl')
    with open(RESULTS_PICKLE, 'rb') as pFile:
        results = pickle.load(pFile)
    try:
        spdr_metrics = SPDR_Metrics(results["reference"], results["hypothesis"], results["uem"])
        dataset = results['dataset']
        metrics = spdr_metrics.get_DiarizationErrorRate(detailed=True)
        
        der = spdr_metrics.get_DiarizationErrorRate(detailed=False)
        purity = spdr_metrics.get_DiarizationPurity(detailed=False)
        coverage = spdr_metrics.get_DiarizationCoverage(detailed=False)
        cpfscore = spdr_metrics.get_DiarizationCoveragePurityFScore(detailed=False)
        confusion = metrics['confusion']
        falsealarm = metrics['false alarm']
        correct = metrics['correct']
        misseddetection = metrics['missed detection']
        total = metrics['total']
        ptext = "DER: %.4f | Purity: %.4f | Coverage: %.4f | PC-F-Score: %.4f \nConfusion: %.2f | False Alarm: %.2f | Correct: %.2f | Missed Detection %.2f" % (\
            der, \
            purity, \
            coverage, \
            cpfscore, \
            confusion, \
            falsealarm, \
            correct, \
            misseddetection
            )
        print('='*15,'Results','='*15)
        print('DER:\t\t\t%.4f'%der)
        print('Purity:\t\t\t%.4f'%purity)
        print('Coverage:\t\t%.4f'%coverage)
        print('PC-F-Score:\t\t%.4f'%cpfscore)
        print('Confusion:\t\t%.4f'%confusion)
        print('False Alarm:\t\t%.4f'%falsealarm)
        print('Correct:\t\t%.4f'%correct)
        print('Missed Detection:\t%.4f'%misseddetection)
        print('Total:\t\t\t%.4f'%total)
        print('='*39)
    except:
        print("Could not determine metric")
    # try:
    #     plot = spdr_metrics.get_Plot(title=dataset['id'], text=ptext, get_plt=True)
    #     plot.savefig(os.path.join(run_folder, 'result.png'))
    #     plot.close()
    # except Exception as error:
    #     continue
    # finally:
    #     del metrics # forcefully destruct object