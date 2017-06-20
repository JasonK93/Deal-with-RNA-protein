import numpy as np
import plmDCA as plm
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial.distance import squareform,pdist
import re



theta = 0.3
pseudocount_weight = 0.5
msa_fasta_filename = 'RF00167.afa.txt'
seqid_of_interest = '1Y26'


Pij_true, Pi_true, alignment_width, q, encoded_seq_of_interest, focus_to_uniprot_offset_map, header_alignment \
    = plm.read_alignment(msa_fasta_filename, seqid_of_interest, theta)

Pij, Pi = plm.with_pc(Pij_true, Pi_true, pseudocount_weight, alignment_width, q)

C = plm.Compute_C(Pij, Pi, alignment_width, q)

invC = np.linalg.inv(C)


list_row = []
for i in xrange(0, alignment_width - 1):
    for j in xrange(i + 1, alignment_width):
        MI_true, _, _ = plm.calculate_mi(i, j, Pij_true, Pi_true, q)
        W_mf = plm.ReturnW(invC, i, j, q)
        DI_mf_pc = plm.bp_link(i, j, W_mf, Pi, q)
        list_row.append([focus_to_uniprot_offset_map[i], plm.number2letter(encoded_seq_of_interest[i]),
                    focus_to_uniprot_offset_map[j], plm.number2letter(encoded_seq_of_interest[j]),
                    MI_true, DI_mf_pc])

data = pd.DataFrame(list_row)
data.columns = ['i', 'i_letter', 'j', 'j_letter', 'MI', 'DI']

plt.figure(1)
data_MI = data.pivot('i', 'j', 'MI')
ax = sns.heatmap(data_MI)

plt.figure(2)
data_DI = data.pivot('i', 'j', 'DI')
ax = sns.heatmap(data_DI)



MI = data['MI']
DI = data['DI']
plt.figure(3)
plt.hist(MI, bins=100)
plt.figure(4)
plt.hist(DI, bins=100)

plt.show()
sort_MI = data.sort_values(by='MI' ,ascending= False)
sort_DI = data.sort_values(by='DI', ascending= False)

sort_MI.head(20)
sort_DI.head(20)




