import os
import re

import numpy as np
import string

################################# ??????????????????????????????????
################################# ??????????????????????????????????
################################# ??????????????????????????????????
################################# ??????????????????????????????????
################################# ??????????????????????????????????
################################# ??????????????????????????????????
#### ??????????????????????? ##### ^^^^^^ *********  CHANGE THIS PLS
# np.random.seed(77)

# dataDir = '/u/cs401/A3/data/'
dataDir = '/Users/maryamebrahimi/Desktop/CSC2511_A3/data/'


def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> Levenshtein("who is there".split(), "is there".split())
    0.333 0 0 1                                                                           
    >>> Levenshtein("who is there".split(), "".split())
    1.0 0 0 3                                                                           
    >>> Levenshtein("".split(), "who is there".split())
    Inf 0 3 0                                                                           
    """

    WER = 0.0
    nS = 0
    nI = 0
    nD = 0
    m = len(h)
    n = len(r)
    ### edge cases
    if m == 0:
        return 1.0, 0, 0, n
    elif n == 0:
        return float("inf"), 0, m, 0
    mat = np.matrix(np.ones((n, m)) * np.inf)
    mat[:, 0] = np.array(range(0, n)).reshape((n, 1))
    mat[0, :] = np.array(range(0, m)).reshape((1, m))

    for i in range(1, n):
        for j in range(1, m):
            ### if ref i and hyp j match
            if r[i] == h[j]:
                S = mat[i-1, j-1]
            ### substitution
            else:
                S = mat[i-1, j-1] + 1
            ### deletion
            D = mat[i-1, j] + 1

            ### insertion
            I = mat[i, j-1] + 1

            mat[i, j] = min([S, D, I])

    ### backtrack in the matrix to count the number of different errors
    i = n-1
    j = m-1
    while i>0 and j>0:
        del_case = mat[i-1, j]
        subs_case = mat[i-1, j-1]
        ins_case = mat[i, j-1]

        min_index = np.argmin([subs_case, ins_case, del_case])

        ### subs error
        if min_index == 0 and r[i] != h[j]:
            nS += 1
            i -= 1
            j -= 1
        ### match case
        elif r[i] == h[j]:
            i -= 1
            j -= 1
        ### insert error
        elif min_index == 1:
            nI += 1
            j -= 1
        ### deletion error
        elif min_index == 2:
            nD += 1
            i -= 1


    # print(mat)
    return (nS + nI + nD) / n, nS, nI, nD


def cleanText(text):
    # remove punctuations
    text = re.sub(r"[^a-zA-Z0-9\s\[\]]", r"", text)
    # put all text to lowercase
    text = text.lower().strip().split()
    return text


if __name__ == "__main__":
    google_leven = []
    kaldi_leven = []
    for root, dirs, files in os.walk(dataDir):
        for speaker in dirs:

            root = os.path.join(dataDir, speaker)
            referenceDir = os.path.join(root, 'transcripts.txt')
            GoogleDir = os.path.join(root, 'transcripts.Google.txt')
            KaldiDir = os.path.join(root, 'transcripts.Kaldi.txt')

            ### open files
            referenceFile = open(referenceDir, 'r')
            GoogleFile = open(GoogleDir, 'r')
            KaldiFile = open(KaldiDir, 'r')


            for i, (ref, google, kaldi) in enumerate(zip(referenceFile, GoogleFile, KaldiFile)):
                ref = cleanText(ref)
                google = cleanText(google)
                kaldi = cleanText(kaldi)

                g = Levenshtein(ref, google)
                k = Levenshtein(ref, kaldi)

                google_leven.append(list(g))
                kaldi_leven.append(list(k))

                print(speaker, "Google", i, g[0], "S:", g[1], "I:", g[2], "D:", g[3])
                print(speaker, "Kaldi", i, k[0], "S:", k[1], "I:", k[2], "D:", k[3])



    print("WER")
    print("Google mean: ", np.mean(np.array(google_leven)[:,0]), " std: ", np.std(np.array(google_leven)[:,0]))
    print("Kaldi mean: ", np.mean(np.array(kaldi_leven)[:,0]), " std: ", np.std(np.array(kaldi_leven)[:,0]))

    print("Substitution")
    print("Google mean: ", np.mean(np.array(google_leven)[:,1]), " std: ", np.std(np.array(google_leven)[:,1]))
    print("Kaldi mean: ", np.mean(np.array(kaldi_leven)[:,1]), " std: ", np.std(np.array(kaldi_leven)[:,1]))

    print("Insertion")
    print("Google mean: ", np.mean(np.array(google_leven)[:,2]), " std: ", np.std(np.array(google_leven)[:,2]))
    print("Kaldi mean: ", np.mean(np.array(kaldi_leven)[:,2]), " std: ", np.std(np.array(kaldi_leven)[:,2]))

    print("Deletion")
    print("Google mean: ", np.mean(np.array(google_leven)[:,3]), " std: ", np.std(np.array(google_leven)[:,3]))
    print("Kaldi mean: ", np.mean(np.array(kaldi_leven)[:,3]), " std: ", np.std(np.array(kaldi_leven)[:,3]))
