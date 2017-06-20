import numpy as np
import math
from scipy.spatial.distance import squareform,pdist
import re

def calculate_evolutionary_constraints_plmDCA_RNAversion(msa_fasta_filename, seqid_of_interest, outputfile):
    theta = 0.3
    pseudocount_weight = 0.5

    Pij_true, Pi_true, alignment_width, q, encoded_seq_of_interest, focus_to_uniprot_offset_map ,header_alignment\
        = read_alignment(msa_fasta_filename, seqid_of_interest, theta)
    
    Pij, Pi = with_pc(Pij_true, Pi_true, pseudocount_weight, alignment_width, q)


    C = Compute_C(Pij, Pi, alignment_width, q)

    invC = np.linalg.inv(C)

    with open(outputfile, 'w') as fp:
        for i in xrange(0,alignment_width-1):
            for j in xrange(i+1,alignment_width):
                MI_true, _, _ = calculate_mi(i, j, Pij_true, Pi_true, q)
                W_mf = ReturnW(invC, i, j, q)
                DI_mf_pc = bp_link(i, j, W_mf, Pi, q)
                line = str([focus_to_uniprot_offset_map[i], number2letter(encoded_seq_of_interest[i]),
                           focus_to_uniprot_offset_map[j], number2letter(encoded_seq_of_interest[j]),
                           MI_true, DI_mf_pc])+"\n"
                fp.write(line)


def read_alignment(msa_fasta_filename, seqid_of_interest, theta):
    encoded_focus_alignment, focus_index_of_interest, focus_to_uniprot_offset_map,header_alignment = \
        read_alignment_fasta(msa_fasta_filename, seqid_of_interest)

    encoded_seq_of_interest = encoded_focus_alignment[focus_index_of_interest-1]

    alignment_height, alignment_width = len(encoded_focus_alignment), len(encoded_focus_alignment[0])

    W = np.ones((1,alignment_height))
    if theta >0.0:
        W = (1. / (1 + sum(squareform(pdist(encoded_focus_alignment, 'hamm') < theta))))

    Meff = sum(W)
    q = max(max(encoded_focus_alignment))
    Pij_true = np.zeros((alignment_width, alignment_width, q, q))
    Pi_true = np.zeros((alignment_width, q))

    for j in xrange(0, alignment_height):
        for i in xrange(0, alignment_width):
            Pi_true[i, encoded_focus_alignment[j][i]-1] = Pi_true[i, encoded_focus_alignment[j][i]-1] + W[j]

    Pi_true = Pi_true/Meff
    
    for l in xrange(0, alignment_height):
        for i in xrange(0, alignment_width-1):
            for j in xrange(i+1,alignment_width):
                Pij_true[i, j, encoded_focus_alignment[l][i]-1, encoded_focus_alignment[l][j]-1] = \
                    Pij_true[i, j,encoded_focus_alignment[l][i]-1,encoded_focus_alignment[l][j]-1] + W[l]
                Pij_true[j, i, encoded_focus_alignment[l][j]-1, encoded_focus_alignment[l][i]-1] = \
                    Pij_true[i, j,encoded_focus_alignment[l][i]-1,encoded_focus_alignment[l][j]-1]
    Pij_true = Pij_true / Meff

    scra = np.eye(q)

    for i in xrange(0, alignment_width):
        for alpha in xrange(0, q):
            for beta in xrange(0, q):
                Pij_true[i, i, alpha, beta] = np.dot(Pi_true[i, alpha] , scra[alpha, beta])




    return Pij_true, Pi_true, alignment_width, q, encoded_seq_of_interest, focus_to_uniprot_offset_map, header_alignment


def read_alignment_fasta(msa_fasta_filename, seqid_of_interest):
    METHOD_TO_RESOLVE_AMBIGUOUS_RESIDUES = 2

    full_alignment = []
    header_alignment = []

    with open(msa_fasta_filename, 'r') as fp:
        file = fp.readlines()
        for i in xrange(0, len(file) / 2):
            full_alignment.append(file[2 * i + 1][:-1])
            header_alignment.append(file[2*i][1:-1])
        alignment_height = len(full_alignment)
        alignment_width = len(full_alignment[0])


        full_index_of_interest, range_of_interest_start, range_of_interest_end = \
            find_seq_of_interest(full_alignment,seqid_of_interest,header_alignment)

    encoded_focus_alignment = []
    skipped_sequence_counter = 0

    focuscolumnlist, focus_to_uniprot_offset_map = scan_sequence_of_interest_for_focus_columns(
        full_alignment[full_index_of_interest], range_of_interest_start, letter2number)

    for full_alignment_index in xrange(0, alignment_height):
        focus_alignment_row = [full_alignment[full_alignment_index][x] for x in focuscolumnlist]
        encoded_focus_alignment_row = [letter2number(x) for x in focus_alignment_row]         
        encoded_focus_alignment.append(encoded_focus_alignment_row)
        
        if full_alignment_index == full_index_of_interest:
            focus_index_of_interest = len(encoded_focus_alignment)

    return encoded_focus_alignment, focus_index_of_interest, focus_to_uniprot_offset_map,header_alignment


def with_pc(Pij_true, Pi_true, pseudocount_weight, alignment_width, q):
    Pij = (1. - pseudocount_weight) * Pij_true + pseudocount_weight / q / q * np.ones((alignment_width, alignment_width, q, q))
    Pi = (1. - pseudocount_weight) * Pi_true + pseudocount_weight / q * np.ones((alignment_width, q))
    scra = np.eye(q)

    for i in xrange(0, alignment_width):
        for alpha in xrange(0,q):
            for beta in xrange(0,q):
                Pij[i, i, alpha, beta] = (1. - pseudocount_weight) * Pij_true[i, i, alpha,beta] \
                                         + pseudocount_weight / q * scra[alpha, beta]

    return Pij, Pi


def Compute_C(Pij, Pi, alignment_width, q):
    C = np.zeros((alignment_width * (q - 1), alignment_width * (q - 1)))

    for i in xrange(0, alignment_width):
        for j in xrange(0, alignment_width):
            for alpha in xrange(0, q - 1):
                for beta in xrange(0, q - 1):
                    C[mapkey(i + 1, alpha + 1, q) - 1, mapkey(j + 1, beta + 1, q) - 1] \
                        = Pij[i, j, alpha, beta] - Pi[i, alpha] * Pi[j, beta]

    return C


def mapkey(i,alpha,q):
    A = (q-1)*(i-1)+(alpha)
    return A


def calculate_mi(i, j, P2, P1, q):
    M = 0
    for alpha in xrange(0, q):
        for beta in xrange(0, q):
            if P2[i, j, alpha, beta] > 0:
                M = M + P2[i, j, alpha, beta] * np.log(P2[i, j, alpha, beta] / P1[i, alpha] / P1[j, beta])

    s1 = 0
    s2 = 0

    for alpha in xrange(0, q):
        if P1[i, alpha] > 0:
            s1 = s1 - P1[i, alpha] * np.log(P1[i, alpha])
        if P1[j, alpha] > 0:
            s2 = s2 - P1[j, alpha] * np.log(P1[j, alpha])
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


def compute_mu(i, j, W, P1, q):
    epsilon = 1e-4
    diff = 1.0
    mu1 = (np.ones((1, q)) / q)[0]
    mu2 = (np.ones((1, q)) / q)[0]
    pi = P1[i, :]
    pj = P1[j, :]

    while diff > epsilon:
        scra1 = np.dot(mu2, np.transpose(W))
        scra2 = np.dot(mu1, W)

        new1 = pi / scra1
        new1 = new1 / sum(new1)
        
        new2 = pj / scra2
        new2 = new2 / sum(new2)
        diff = max(np.amax(np.abs(new1 - mu1)), np.amax(np.abs(new2 - mu2)))

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

    # DI = ( np.transpose(Pdir) * np.log((Pdir+tiny)/(Pfac+tiny))).trace

    temp = np.log((Pdir + tiny) / (Pfac + tiny))
    DI = np.dot(np.transpose(np.mat(Pdir)),np.mat(temp)).trace()
    return float(DI)


def find_seq_of_interest(full_alignment, seqid_of_interest,header_alignment):
    index_of_interest = -1
    for scan_index in xrange(0,len(full_alignment)):
        # ----------------------------------------------------------------
        seqid, range_start, range_end = split_uniprot_id(header_alignment[scan_index])
        if seqid == seqid_of_interest:
            index_of_interest = scan_index
            break
    return index_of_interest, int(range_start), int(range_end)


def split_uniprot_id(pfam_uniprot_range_line):
    slashposition =  pfam_uniprot_range_line.find('/')
    seqid = pfam_uniprot_range_line[0:slashposition]
    rangestring = pfam_uniprot_range_line[slashposition+1:len(pfam_uniprot_range_line)]
                                   
    hyphenposition = rangestring.find('-')
    range_start = rangestring[0:hyphenposition]
    range_end = rangestring[hyphenposition +1 : len(rangestring)]
    return seqid, range_start, range_end


def scan_sequence_of_interest_for_focus_columns(sequence_of_interest, range_of_interest_start, letter2number):
    focuscolumnlist = []
    uniprotoffsetlist = []
    next_uniprotoffset = range_of_interest_start

    for pos in xrange(0,len(sequence_of_interest)):
        residuecode = letter2number(sequence_of_interest[pos])
        if residuecode > 1 :
            focuscolumnlist = [focuscolumnlist, pos]
            uniprotoffsetlist = [uniprotoffsetlist, next_uniprotoffset]
        if residuecode == -2 or residuecode >1 :
            
            next_uniprotoffset = next_uniprotoffset + 1
            
    focuscolumnlist = re.findall(r'(\w*[0-9]+)\w*',str(focuscolumnlist))
    focuscolumnlist = [int(x) for x in focuscolumnlist]

    uniprotoffsetlist = re.findall(r'(\w*[0-9]+)\w*',str(uniprotoffsetlist))
    uniprotoffsetlist = [int(x) for x in uniprotoffsetlist]
                       
    return focuscolumnlist, uniprotoffsetlist


def letter2number(a):
    if a == '-':
        x = 1
    elif a == 'A':
        x = 2
    elif a == 'U':
        x = 3
    elif a == 'C':
        x = 4
    elif a == 'G':
        x = 5
    else:
        x = 5

    return x


def number2letter(a):
    if a == 1:
        x = '-'
    elif a == 2:
        x = 'A'
    elif a == 3:
        x = 'U'
    elif a == 4:
        x = 'C'
    elif a == 5:
        x = 'G'
    else:
        x = 'G'

    return x


if __name__ == '__main__':
    calculate_evolutionary_constraints_plmDCA_RNAversion('RF00167.afa.txt','1Y26', 'plmDCA_results.txt')