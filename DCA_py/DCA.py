import numpy as np
import math
from scipy.spatial.distance import squareform,pdist

'''
Direct Coupling Analysis (DCA)

INPUTS:
    inputfile - file containing the FASTA alignment
    outputfile - file for dca results.The file is composed by N(N-1)/2
                (N = length of the sequences) rows and 4 columns:
                residue i (column 1), residue j (column 2),
                MI(i,j) (Mutual Information between i and j), and
                DI(i,j) (Direct Information between i and j).
                Note: all insert columns are removed from the alignment.
SOME RELEVANT VARIABLES:
   N        number of residues in each sequence (no insert)
   M        number of sequences in the alignment
   Meff     effective number of sequences after re weighting
   q        equal to 21 (20 amino acids + 1 gap)
   align    M x N matrix containing the alignment
   Pij_true N x N x q x q matrix containing the re weighted frequency
            counts.
   Pij      N x N x q x q matrix containing the re weighted frequency
            counts with pseudo counts.
   C        N(q-1) x N(q-1) matrix containing the covariance matrix.

'''
def dca(inputfile, outputfile):
    pseudocount_weight = 0.5  # relative weight of pseudo count
    theta = 0.2  # threshold for sequence id in re weighting

    N,M,q,align = return_alignment(inputfile)
    Pij_true,Pi_true, Meff = Compute_True_Frequencies(align, M, N, q, theta)
    print('N = {0}, M = {1}, Meff = {2}, q = {3}'.format(N, M, Meff, q))
    Pij,Pi = with_pc(Pij_true,Pi_true,pseudocount_weight,N,q)
    C = Compute_C(Pij,Pi,N,q)
    invC = np.linalg.inv(C)
    with open (outputfile, 'w') as fp:
        Compute_Results(Pij, Pi, Pij_true, Pi_true, invC, N, q, fp)


def return_alignment(inputfile):
#  reads alignment from inputfile, removes inserts and converts into numbers

    align_full = []
    
    with open (inputfile, 'r') as fp:
        file = fp.readlines()
        for i in xrange(0,len(file)/2):
            align_full.append(file[2*i+1][:-1])
        M = len(align_full)
        N = len(align_full[0])
        
        Z = np.zeros((M,N))
    
        for i in xrange(0,M):
            counter = 0
            for j in xrange(0, N):
                Z[i,counter] = letter2number(align_full[i][j])
                counter += 1
        q = int(np.max(np.max(Z)))
        return N ,M, q, Z

# computes and prints the mutual and direct informations
def Compute_Results(Pij, Pi, Pij_true,Pi_true, invC, N, q, fp):
        for i in xrange(0,N-1):
            for j in xrange(i+1,N):
                # mutual information
                MI_true,si_true,sj_true = calculate_mi(i,j,Pij_true,Pi_true,q)
                
                # direct information from mean-field
                W_mf = ReturnW(invC,i,j,q) 
                DI_mf_pc = (bp_link(i,j,W_mf,Pi,q))
                line = str([i+1,j+1,MI_true, DI_mf_pc])+"\n"
                fp.write(line)

                
def Compute_True_Frequencies(align,M,N,q,theta):
    W = np.ones((1,M))
    if (theta > 0.0):
        W = (1./(1+sum(squareform(pdist(align,'hamm')<theta))))
    Meff = sum(W)
    
    Pij_true = np.zeros((N,N,q,q))
    Pi_true = np.zeros((N,q))
    
    for j in xrange(0, M):
        for i in xrange(0, N):
            Pi_true[i, int(align[j,i]-1)] = Pi_true[i, int(align[j,i]-1)] + W[j]
    Pi_true = Pi_true/Meff
    
    for l in xrange(0, M):
        for i in xrange(0, N-1):
            for j in xrange(i+1,N):
                Pij_true[i, j, int(align[l,i]-1), int(align[l,j]-1)] = \
                    Pij_true[i,j,int(align[l,i]-1),int(align[l,j]-1)] + W[l]
                Pij_true[j,i,int(align[l,j]-1),int(align[l,i]-1)] = \
                    Pij_true[i,j,int(align[l,i]-1),int(align[l,j]-1)]
    Pij_true = Pij_true/Meff
    
    scra = np.eye(q)
    
    for i in xrange(0,N):
        for alpha in xrange(0,q):
            for beta in xrange(0,q):
                Pij_true[i,i,alpha,beta] = Pi_true[i,alpha] * scra[alpha,beta]
    
    return Pij_true, Pi_true, Meff


def letter2number(a):
    if a =='A':
        x = 1
    elif a == 'U':
        x = 2
    elif a == 'C':
        x = 3
    elif a == 'G':
        x = 4
    elif a == '-':
        x = 5
    else:
        x = 5
            
    return x


def with_pc(Pij_true, Pi_true, pseudocount_weight,N,q):
    # adds pseudocount
    Pij = (1.-pseudocount_weight)*Pij_true + pseudocount_weight/q/q*np.ones((N,N,q,q))
    Pi = (1.-pseudocount_weight)*Pi_true + pseudocount_weight/q*np.ones((N,q))
    scra = np.eye(q);
    
    for i in xrange(0, N):
        for alpha in xrange(0, q):
            for beta in xrange(0, q):
                Pij[i,i,alpha,beta] =  (1.-pseudocount_weight)*Pij_true[i,i,alpha,beta]\
                                       + pseudocount_weight/q*scra[alpha,beta]
    return Pij, Pi

def Compute_C(Pij,Pi,N,q):
    # compute correlation matrix
    C = np.zeros((N*(q-1),N*(q-1)))
    
    for i in xrange(0, N):
        for j in xrange(0, N):
            for alpha in xrange(0, q-1):
                for beta in xrange(0, q-1):
                    C[mapkey(i+1,alpha+1,q)-1,mapkey(j+1,beta+1,q)-1] \
                        = Pij[i,j,alpha,beta] - Pi[i,alpha]*Pi[j,beta]
    return C

    
def mapkey(i,alpha,q):
    A = (q-1)*(i-1)+(alpha)
    return A
    

def calculate_mi(i,j,P2,P1,q):
    M = 0
    for alpha in xrange(0,q):
        for beta in xrange(0, q):
            if P2[i, j, alpha, beta] > 0:
                M = M + P2[i,j,alpha, beta]*np.log(P2[i,j, alpha, beta] / P1[i,alpha]/P1[j,beta])
    
    s1 = 0
    s2 = 0
    
    for alpha in xrange(0, q):
        if P1[i,alpha] > 0:
            s1 = s1- P1[i,alpha] * np.log( P1[i, alpha])
        if P1[j,alpha] > 0:

            s2 = s2 - P1[j,alpha] * np.log(P1[j,alpha])
    return M, s1, s2
    

def ReturnW(C, i, j, q):
    W = np.ones((q,q))


    for a in xrange(0,q-1):
        for b in xrange(0, q-1):
            W[a,b] = np.exp(-C[mapkey(i+1,a+1,q)-1,mapkey(j+1,b+1,q)-1])
    return W

def bp_link(i,j,W,P1,q):
    mu1, mu2 = compute_mu(i,j,W,P1,q)
    DI = compute_di(i,j,W, mu1,mu2,P1)
    return DI
    
def compute_mu(i,j,W,P1,q):
    epsilon=1e-4
    diff =1.0
    mu1 = (np.ones((1,q))/q)[0]
    mu2 = (np.ones((1,q))/q)[0]
    pi = P1[i,:]
    pj = P1[j,:]
    
    while diff > epsilon:
        
        scra1 = np.dot(mu2 , np.transpose(W))
        scra2 = np.dot(mu1 , W)

        new1 = pi/scra1
        new1 = new1/sum(new1)
        # print '----',pi,scra1,new1,sum(new1)
        new2 = pj/scra2
        new2 = new2/sum(new2)
        diff = max(np.amax( np.abs(new1-mu1)), np.amax(np.abs(new2-mu2)))


        mu1 = new1
        mu2 = new2
    return mu1, mu2

def compute_di(i,j,W, mu1,mu2, Pia):
    # compute direct information
    tiny = 1.0e-100
    mu1 = np.mat(mu1)
    mu2 = np.mat(mu2)
    Pdir = np.multiply(W,(np.dot(np.transpose(mu1),mu2)))
    Pdir = Pdir / (sum(sum(Pdir)).sum())

    Pfac = np.dot(np.transpose(np.mat(Pia[i,:])) , np.mat(Pia[j,:]))


    temp = np.log((Pdir + tiny) / (Pfac + tiny))
    DI = np.dot(np.transpose(np.mat(Pdir)),np.mat(temp)).trace()
    return float(DI)

if __name__ == '__main__':
    dca('RF00167.afa.txt','DCA_results.txt')





