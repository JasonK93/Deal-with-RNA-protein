import numpy as np
import DCA
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


pseudocount_weight = 0.5  # relative weight of pseudo count
theta = 0.2  # threshold for sequence id in re weighting
inputfile = 'RF00167.afa.txt'

def Compute_Results(Pij, Pi, Pij_true, Pi_true, invC, N, q):
    list_row = []

    for i in xrange(0, N - 1):
        for j in xrange(i + 1, N):
            # mutual information
            MI_true, si_true, sj_true = DCA.calculate_mi(i, j, Pij_true, Pi_true, q)

            # direct information from mean-field
            W_mf = DCA.ReturnW(invC, i, j, q)
            DI_mf_pc = DCA.bp_link(i, j, W_mf, Pi, q)
            list_row.append([i+1,j+1,MI_true,DI_mf_pc])

    return list_row



N, M, q, align = DCA.return_alignment(inputfile)
Pij_true, Pi_true, Meff = DCA.Compute_True_Frequencies(align, M, N, q, theta)
print('N = {0}, M = {1}, Meff = {2}, q = {3}'.format(N, M, Meff, q))
Pij, Pi = DCA.with_pc(Pij_true, Pi_true, pseudocount_weight, N, q)
C = DCA.Compute_C(Pij, Pi, N, q)
invC = np.linalg.inv(C)


data = Compute_Results(Pij, Pi, Pij_true, Pi_true, invC, N, q)
data = pd.DataFrame(data)
data.columns = ['i','j','MI','DI']


plt.figure(1)
data_MI = data.pivot('i', 'j', 'MI')
ax = sns.heatmap(data_MI)

plt.figure(2)
data_DI = data.pivot('i', 'j', 'DI')
ax = sns.heatmap(data_DI)

plt.show()

MI = data['MI']
DI = data['DI']

plt.hist(MI, bins=100)
plt.hist(DI, bins=100)


sort_MI = data.sort_values(by='MI' ,ascending= False)
sort_DI = data.sort_values(by='DI', ascending= False)

sort_MI.head(20)
sort_DI.head(20)



