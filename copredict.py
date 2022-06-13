import numpy as np
import scipy.io as sio
import csv
import os


# ########### load the probs of the model trained on EO images ####################
probs_EO = sio.loadmat('EO_iteration_num2_fortest_bestacc.mat')
probs_EO = probs_EO['preds']
# print(probs_EO.shape)
probs_EO_max = np.squeeze(np.max(probs_EO, 2)).tolist()
# print(len(probs_EO_max))
probs_EO_argmax = np.squeeze(np.argmax(probs_EO, 2)).tolist()

# ######### load the probs of the model trained on EO-SAR images ###################
probs_EOSAR = sio.loadmat('EOSAR_iteration_num1_fortest_bestacc.mat')
probs_EOSAR = probs_EOSAR['preds']
probs_EOSAR_max = np.squeeze(np.max(probs_EOSAR, 2)).tolist()
probs_EOSAR_argmax = np.squeeze(np.argmax(probs_EOSAR, 2)).tolist()

all_names = []
for file in sorted(os.listdir('./test_images_EO')):
    name = file.split('_')[-1].split('.')[0]
    all_names.append(name)


csv_file = open('results_final.csv', 'w', newline='')
fwriter = csv.writer(csv_file)
row_now = ['image_id', 'class_id']
fwriter.writerow(row_now)
for i in range(826):
    if probs_EO_max[i] > probs_EOSAR_max[i]:
        predict = probs_EO_argmax[i]
    else:
        predict = probs_EOSAR_argmax[i]
    row_now = [all_names[i], predict]
    fwriter.writerow(row_now)
csv_file.close()

