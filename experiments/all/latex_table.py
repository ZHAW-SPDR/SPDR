import csv
RESULTS = './experiments/all/random.csv'
MEETINGS = 7
RUNS_PER_MEETING = 10
ATTRIBUTE_COUNT = 4

with open(RESULTS) as f:
    csv_ = list(csv.reader(f,delimiter=',')) 


COUNTER = 0
MEETING = ''
for i, row in enumerate(csv_):
    if i == 0:
        continue # skipping header
    
    if MEETING != row[0]:
        COUNTER = 0
        MEETING = row[0]

    COUNTER += 1
    
    attributes = row[1:]
    attributes_labels = ['DER', 'Purity', 'Coverage', 'F-Score']
    
    if COUNTER == 1:
        m_val = MEETING.replace('_','\_')
    else:
        m_val = ''
    if COUNTER == 10:
        t_val = ' \\\\ \hline'
    else:
        t_val = ' \\\\* \cline{2-6}'    
    print (m_val, ' & ', COUNTER, ' & ', row[1], ' & ', row[2], ' & ', row[3], ' & ', row[4], t_val)
    
