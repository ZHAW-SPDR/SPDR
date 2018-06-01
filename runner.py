from controller import Controller
import yaml
import os
import shutil
import pickle
from keras import backend as K

IN_PATH = './data/in/'
CONFIG = 'config.yml'
#EXPERIMENTS = 'single_experiment.yml'
EXPERIMENTS = 'experiments.yml'
RUN_PATH = './data/runs/'
PICKLE_PATH = './data/pickles/'
OUT_PATH = './data/out/'

# functions 
def update_config(old, new):
    '''
        Updating a yaml config with a new one.
        Requires max depth of two.
        Requires first level to not have values.
    '''
    config = dict(old)
    for k1, v1 in new.items():
        for k2, v2 in v1.items():
            config[k1][k2] = v2
    return config

# define run
with open(CONFIG) as config:
    original_yaml = yaml.safe_load(config)
old = dict(original_yaml)
with open(EXPERIMENTS) as exp:
    experiments_ = list(yaml.load_all(exp))

for idx, experiment_ in enumerate(experiments_):
    experiment_yaml = update_config(old, experiment_)
    
    ds = experiment_['data']['in']
    ds_subset = experiment_['data']['datasubset_to_use']
    network = experiment_['embeddings']['network']
    ds = ds.replace(IN_PATH,'').replace('/','')
    network = network.replace('.h5','')
    experiment_name = "DATASET_%s.SUBSET_%s.MODEL_%s" % (ds.lower(), ds_subset.lower(), network.lower())
    print('='*20, 'Running experiment',idx+1,'of',len(experiments_), '='*20)
    print('-'*10, 'Name:', experiment_name, '-'*10)

    # remove previous experiment data
    for path, dirs, files in os.walk(OUT_PATH):
        for elem in dirs:
            if ds_subset in elem:
                shutil.rmtree(os.path.join(OUT_PATH, elem))
    
    for path, dirs, files in os.walk(PICKLE_PATH):
        for f in files:
            if ds_subset in f:
                os.unlink(os.path.join(PICKLE_PATH, f))
    
    # setup directory
    run_folder = os.path.join(RUN_PATH, experiment_name)
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
    else:
        shutil.rmtree(run_folder)
        os.mkdir(run_folder)
    
    # update config
    with open(CONFIG,'w') as config:
        yaml.dump(experiment_yaml, config, default_flow_style=False)

    # run
    try:
        c = Controller()
        res = c.run(ret=True)

        # store results
        #res['plot'].suptitle('Performance on \n%s' % experiment_name)
        #res['plot'].savefig(os.path.join(run_folder, 'result.png'))
        #res['plot'].close()
        RESULTS_PICKLE = os.path.join(run_folder, 'result.pkl')
        with open(RESULTS_PICKLE, 'wb') as pFile:
            pickle.dump(res, pFile)
        print("Results stored for this experiment!")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        print('An exception occurred: {}'.format(error))
    finally:
        # cleanup
        with open(CONFIG,'w') as config:
            yaml.dump(original_yaml, config, default_flow_style=False)
    
# Explicitly reap session to avoid an AttributeError sometimes thrown by
# TensorFlow on shutdown. See:
# https://github.com/tensorflow/tensorflow/issues/3388
K.clear_session()