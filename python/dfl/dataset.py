import torch
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import os, csv


class LossTrainingDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, root_dir, dataset_name):
      """
      Args:
          root_dir (string): Directory of asc / dat.
      """
      self.root_dir = root_dir
      self.dataset_name = dataset_name
      self.data_dir = root_dir + dataset_name

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + 'wav')
        y = torch.load('data/' + ID + 'wav')

        return X, y

def load_asc_data(ase_folder):

    sets = ['train', 'val']
    folders = {}
    for setname in sets:
        folders[setname] = ase_folder + "/" + setname + "set"
    labels = {}
    names = {}
    datasets = {}

    for setname in sets:
        foldername = folders[setname]

        labels[setname] = []
        names[setname] = []
        datasets[setname] = []

        n = []
        l = []

        with open('%s/meta.txt' % foldername, 'r') as csvfile:
            metareader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in metareader:
                n.append(row[0][6:])
                l.append(row[1])

        for i in tqdm(range(len(n))):
            filename = n[i]
            fs, inputAudio = wavfile.read(foldername + '/' + filename)
            if not (fs == 16000):
                raise ValueError('Sample frequency is not 16kHz')
            shape = np.shape(inputAudio)
            # print filename, fs, np.max(inputAudio), shape
            if len(shape) > 1 and shape[1] > 1:
                for j in range(shape[1]):
                    inputData = np.reshape(inputAudio[:, j], [1, 1, shape[0], 1])
                    datasets[setname].append(inputData)
                    labels[setname].append(l[i])
                    names[setname].append(n[i])
            else:
                inputData = np.reshape(inputAudio, [1, 1, shape[0], 1])
                datasets[setname].append(inputData)
                labels[setname].append(l[i])
                names[setname].append(n[i])

    label_list = list(set(labels[sets[0]]))

    return datasets, labels, names, label_list


# DOMESTIC AUDIO TAGGING - LOAD DATA
def load_dat_data(dat_folder):

    sets = ['train', 'val']
    csv_files = {}
    csv_files[sets[0]] = dat_folder + "/development_chunks_refined.csv"
    csv_files[sets[1]] = dat_folder + "/evaluation_chunks_refined.csv"
    labels = {}
    names = {}
    datasets = {}

    for setname in sets:

        labels[setname] = []
        names[setname] = []
        datasets[setname] = []

        n = []
        l = []

        with open(csv_files[setname], 'r') as csvfile:
            metareader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in metareader:
                n.append(row[1] + ".wav")
                with open('%s/%s.csv' % (dat_folder, row[1]), 'r') as csvfile2:
                    metareader2 = csv.reader(csvfile2, delimiter=',', quotechar='|')
                    for row in metareader2:
                        if row[0] == 'majorityvote':
                            l.append(row[1])

        for i in tqdm(range(len(n))):
            filename = n[i]
            fs, inputAudio = wavfile.read(dat_folder + '/' + filename)
            if not (fs == 16000):
                raise ValueError('Sample frequency is not 16kHz')
            shape = np.shape(inputAudio)
            if len(shape) > 1 and shape[1] > 1:
                for j in range(shape[1]):
                    inputData = np.reshape(inputAudio[:, j], [1, 1, shape[0], 1])
                    datasets[setname].append(inputData)
                    labels[setname].append(l[i])
                    names[setname].append(n[i])
            else:
                inputData = np.reshape(inputAudio, [1, 1, shape[0], 1])
                datasets[setname].append(inputData)
                labels[setname].append(l[i])
                names[setname].append(n[i])

    label_list = []
    for label in labels[sets[0]]:
        for ch in list(label):
            if not (label == 'S'):
                label_list.append(ch)
    label_list = list(set(label_list))

    return datasets, labels, names, label_list

if __name__ =='__main__':
    datapath = '/home/shuyu/Projects/intel/dataset/'
    ds = LossTrainingDataset(datapath, 'asc')
    datasets, labels, names, label_list = load_asc_data(datapath + 'asc')
    print(datasets)
    print(names)
    print(label_list)
    datasets, labels, names, label_list = load_dat_data(datapath + 'dat')

    print(datasets)
    print(names)
    print(label_list)