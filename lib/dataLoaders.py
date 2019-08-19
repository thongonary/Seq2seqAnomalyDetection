import sys
import numpy as np
from torch.utils import data
from prettytable import PrettyTable

class ParticleDataset(data.Dataset):
    def __init__(self,
                 template='../data/20190702_20part_PtOrder_v1/{}.npy',
                 N_part = 20,
                 N_features = 5
                ):
        self.template = template
        self.SM_names = ['Wlnu', 'qcd', 'Zll', 'ttbar']
        self.SM_fraction = np.array([0.592, 0.338, 0.067, 0.003])
        self.BSM_names = ['Ato4l', 'leptoquark', 'hToTauTau', 'hChToTauNu']

        self.process_labels = {'Ato4l':r'$A\to 4\ell$',
                               'leptoquark':r'$LQ$',
                               'hToTauTau':r'$h^{0}\to \tau\tau$',
                               'hChToTauNu':r'$h^{\pm}\to \tau\nu$',
                               'SMMix': 'SM Mix'
                               }
        self.process_colors = {'Ato4l':'k',
                               'leptoquark':'g',
                               'hToTauTau':'r',
                               'hChToTauNu':'b',
                               'SMMix':'darkorange'
                               }

        self.N_part = N_part

        self.N_features = N_features
        self.feature_names = ['Pt', 'eta', 'phi', 'charge', 'pId'][:N_features]

        self.loss = {}

    def loadTrainSM(self, training_split_fraction=0.5, N_train_max = 1e10):
        raw_sample = {}
        l = np.zeros(4)
        for i,n in enumerate(self.SM_names):
            print('Fetching', n)
            raw_sample[n] = np.load(self.template.format(n))
            l[i] = raw_sample[n].shape[0]

        i_min = np.argmin(l/self.SM_fraction)
        if N_train_max > training_split_fraction*l[i_min]/self.SM_fraction[i_min]:
            self.limiting = self.SM_names[i_min]
        else:
            self.limiting = 'stat'
        if self.limiting=='qcd':
            print('QCD is limiting, using it for both val and split')
            i_min = np.argsort(l/self.SM_fraction)[1]
            N_train = min(N_train_max, training_split_fraction*l[i_min]/self.SM_fraction[i_min])
        else:
            N_train = min(N_train_max, training_split_fraction*l[i_min]/self.SM_fraction[i_min])

        N_val = N_train*(1-training_split_fraction)/training_split_fraction - 1
        print('Expected {:.2f}M train'.format(N_train/1.0e6))
        print('Expected {:.2f}M val\n'.format(N_val/1.0e6))


        x_train_s = {}
        x_val_s = {}

        table = PrettyTable(['Sample', 'Evts tot', 'Train', 'Val'])
        self.N_train_SM = {}

        for i,n in enumerate(self.SM_names):
            N_train_aux = int(N_train * self.SM_fraction[i])
            self.N_train_SM[n] = N_train_aux
            x_train_s[n] = raw_sample[n].astype(np.float32)[:N_train_aux,:self.N_part, :self.N_features]
            N_val_aux = int(N_val * self.SM_fraction[i])
            if self.limiting=='qcd' and n == 'qcd':
                print('Reloading QCD for validation')
                N_reload = N_val_aux - (raw_sample[n].shape[0] - N_train_aux)
                idx_start = N_train_aux - N_reload
                self.N_train_SM[n] = idx_start
                x_val_s[n] = raw_sample[n].astype(np.float32)[idx_start:, :self.N_part, :self.N_features]
            elif N_train_aux+N_val_aux < raw_sample[n].shape[0]:
                print('Loading', n)
                x_val_s[n] = raw_sample[n].astype(np.float32)[N_train_aux:N_train_aux+N_val_aux, :self.N_part, :self.N_features]
            else:
                print(N_train_aux, N_val_aux, raw_sample[n].shape[0])
                print('Error', n)
                raise
                continue
            table.add_row([n, '{:.0f}k'.format(raw_sample[n].shape[0]/1e3), '{:.0f}k'.format(x_train_s[n].shape[0]/1e3), '{:.0f}k'.format(x_val_s[n].shape[0]/1e3)])
        print(table)

        self.SMMix_train = np.concatenate((x_train_s['Wlnu'], x_train_s['qcd'], x_train_s['Zll'], x_train_s['ttbar']))
        self.SMMix_val = np.concatenate((x_val_s['Wlnu'], x_val_s['qcd'], x_val_s['Zll'], x_val_s['ttbar']))

        print('Tot training {:.2f} M'.format(self.SMMix_train.shape[0]/1.0e6))
        print('Tot val {:.2f} M'.format(self.SMMix_val.shape[0]/1.0e6))

    def loadValidationSamples(self, samples='BSM+SM'):
        if not hasattr(self, 'valSamples'):
            self.valSamples = {}

        if 'BSM' in samples:
            for n in self.BSM_names:
                sys.stdout.write('Loading '+n)
                sys.stdout.flush()
                self.valSamples[n] = np.load(self.template.format(n)).astype(np.float32)[:, :self.N_part, :self.N_features]
                sys.stdout.write(' ({:.1f}k)\n'.format(1e-3*self.valSamples[n].shape[0]))

        if 'SM' in samples.replace('BSM', ''):
            for n in self.SM_names:
                sys.stdout.write('Loading '+n)
                sys.stdout.flush()
                idx_start = self.N_train_SM[n]
                self.valSamples[n] = np.load(self.template.format(n)).astype(np.float32)[idx_start:, :self.N_part, :self.N_features]
                sys.stdout.write(' ({:.1f}k)\n'.format(1e-3*self.valSamples[n].shape[0]))

            l = np.zeros(4)
            for i,n in enumerate(self.SM_names):
                l[i] = self.valSamples[n].shape[0]

            i_min = np.argmin(l/self.SM_fraction)
            print('SM Mix limiting stat. sample: {} ({:.2f}M)'.format(self.SM_names[i_min], l[i_min]*1e-6))

            self.SM_val_weights = []
            for i,n in enumerate(self.SM_names):
                w = np.float128(self.SM_fraction[i]/ self.SM_fraction[i_min]) * np.float128(l[i_min]/l[i])
                self.SM_val_weights.append(w)
            print('SM validation weights')
            print(list(zip(self.SM_names, self.SM_val_weights)))

    def charge(self, target):
        self.inputs = target

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # Fix phi
        target = self.inputs[idx]
        #new_phi = np.mod(target[:,2] - target[0,2], 2*np.pi)
        #new_phi = np.where(new_phi < -np.pi, new_phi+2*np.pi, np.where(new_phi > np.pi, new_phi - 2*np.pi, new_phi))
        #target[:,2] = new_phi
        return target, target
