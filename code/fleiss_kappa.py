""" Computes the Fleiss' Kappa value as described in (Fleiss, 1971) """
import pickle

DEBUG = True

def computeKappa(mat):
    """ Computes the Kappa value
        @param n Number of rating per subjects (number of human raters)
        @param mat Matrix[subjects][categories]
        @return The Kappa value """
    n = checkEachLineCount(mat)   # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])
    
    if DEBUG:
        print n, "raters."
        print N, "subjects."
        print k, "categories."
    
    # Computing p[]
    p = [0.0] * k
    for j in xrange(k):
        p[j] = 0.0
        for i in xrange(N):
            p[j] += mat[i][j]
        p[j] /= N*n
    if DEBUG: print "p =", p
    
    # Computing P[]    
    P = [0.0] * N
    for i in xrange(N):
        P[i] = 0.0
        for j in xrange(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))
    #if DEBUG: print "P =", P
    
    # Computing Pbar
    Pbar = sum(P) / N
    if DEBUG: print "Pbar =", Pbar
    
    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj
    if DEBUG: print "PbarE =", PbarE
    
    kappa = (Pbar - PbarE) / (1 - PbarE)
    if DEBUG: print "kappa =", kappa
    
    return kappa

def checkEachLineCount(mat):
    """ Assert that each line has a constant number of ratings
        @param mat The matrix checked
        @return The number of ratings
        @throws AssertionError If lines contain different number of ratings """
    n = sum(mat[0])
    i = 0
    j =0
    for line in mat[1:]:
        j += 1
        if sum(line) != n:
            print "LINE", line
            i += 1
    print "i=", i
    print "j=", j
    
    assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
    return n

if __name__ == "__main__":
    """ Example on this Wikipedia article data set """
    """
    mat = \
    [
        [0,0,0,0,14],
        [0,2,6,4,2],
        [0,0,3,5,6],
        [0,3,9,2,0],
        [2,2,8,1,1],
        [7,7,0,0,0],
        [3,2,6,3,0],
        [2,5,3,2,2],
        [6,5,2,1,0],
        [0,2,2,3,7]
    ]
    """

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_ALL.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_REAL.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_FAKE.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_WordRNN10.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_WordRNN07.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_WordRNN05.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_GoogleLM.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)
    
    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_AttentionAC.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_NoAttentionAC.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_SkipConnections.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_MLESeqGAN.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_SS.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_SeqGAN.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    #with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_RankGAN.pickle", "rb") as handle:
    #    fleis_kappa_matrix = pickle.load(handle)

    with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_LeakGAN.pickle", "rb") as handle:
        fleis_kappa_matrix = pickle.load(handle)

    kappa = computeKappa(fleis_kappa_matrix)




