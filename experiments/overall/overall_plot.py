import csv
import matplotlib.pyplot as plt
import numpy as np
RESULTS = './experiments/overall/results.csv'

with open(RESULTS) as f:
    #csv_ = list(csv.reader(f, dialect='excel', delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)) 
    csv_ = list(csv.reader(f, dialect='excel', delimiter=';', quoting=csv.QUOTE_NONNUMERIC)) 

x = [row[0] for row in csv_]
yt = np.linspace(0,1,21)
plt.subplot(111)
plt.title('Diarization Error Rate')
plt.xticks(x)
plt.yticks(yt, ["%.2f" %  ytl for ytl in yt])
plt.plot(x, [row[1] for row in csv_], label='Cutoff 0.0')
plt.plot(x, [row[5] for row in csv_], label='Cutoff 0.05')
plt.plot(x, [row[9] for row in csv_], label='Cutoff 0.1')
plt.xlabel("Run")
plt.ylabel("Metric Value")
plt.ylim((-0.05,1.05))
plt.gca().xaxis.grid(True)
plt.legend()
plt.savefig('./data/plots/experiment_overall/exp_overall_der.png')
plt.close()
plt.subplot(111)
plt.title('Purity')
plt.xticks(x)
plt.yticks(yt, ["%.2f" %  ytl for ytl in yt])
plt.plot(x, [row[2] for row in csv_], label='Cutoff 0.0')
plt.plot(x, [row[6] for row in csv_], label='Cutoff 0.05')
plt.plot(x, [row[10] for row in csv_], label='Cutoff 0.1')
plt.xlabel("Run")
plt.ylabel("Metric Value")
plt.ylim((-0.05,1.05))
plt.gca().xaxis.grid(True)
plt.legend()
plt.savefig('./data/plots/experiment_overall/exp_overall_purity.png')
plt.close()
plt.subplot(111)
plt.title('Coverage')
plt.xticks(x)
plt.yticks(yt, ["%.2f" %  ytl for ytl in yt])
plt.plot(x, [row[3] for row in csv_], label='Cutoff 0.0')
plt.plot(x, [row[7] for row in csv_], label='Cutoff 0.05')
plt.plot(x, [row[11] for row in csv_], label='Cutoff 0.1')
plt.xlabel("Run")
plt.ylabel("Metric Value")
plt.ylim((-0.05,1.05))
plt.gca().xaxis.grid(True)
plt.legend()
plt.savefig('./data/plots/experiment_overall/exp_overall_coverage.png')
plt.close()
plt.subplot(111)
plt.title('F-Score')
plt.xticks(x)
plt.yticks(yt, ["%.2f" %  ytl for ytl in yt])
plt.plot(x, [row[4] for row in csv_], label='Cutoff 0.0')
plt.plot(x, [row[8] for row in csv_], label='Cutoff 0.05')
plt.plot(x, [row[12] for row in csv_], label='Cutoff 0.1')
plt.xlabel("Run")
plt.ylabel("Metric Value")
plt.ylim((-0.05,1.05))
plt.gca().xaxis.grid(True)
plt.legend()
plt.savefig('./data/plots/experiment_overall/exp_overall_fscore.png')
plt.close()