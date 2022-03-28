from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

################################# ??????????????????????????????????
################################# ??????????????????????????????????
################################# ??????????????????????????????????
################################# ??????????????????????????????????
################################# ??????????????????????????????????
################################# ??????????????????????????????????
#### ??????????????????????? ##### ^^^^^^ *********  CHANGE THIS PLS

random.seed(3)
np.random.seed(3)

# dataDir = '/u/cs401/A3/data/'
dataDir = '/Users/maryamebrahimi/Desktop/CSC2511_A3/data/'


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        # based on slide 32 of tutorial A3
        part1 = np.sum((self.mu[m] ** 2) / (self.Sigma[m] * 2))
        part2 = (self._d / 2) * np.log(2 * np.pi)
        part3 = np.sum(np.log(self.Sigma[m])) / 2

        return part1 + part2 + part3

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    # print("*** in log_b_m_x")
    # print("x shape ", x.shape)

    # based on slide 32 of tutorial A3
    part1 = (x ** 2) / (myTheta.Sigma[m] * 2)
    part2 = myTheta.mu[m] * x / myTheta.Sigma[m]

    # for both single row and vectorized cases
    a = 1
    if len(x.shape) == 1:
        a = 0

    # print("%%%  cont part  ", myTheta.precomputedForM(m))
    l_b_m_x = -np.sum((part1 - part2), axis=a) - myTheta.precomputedForM(m)
    # print("log_b_m_x shape ", l_b_m_x.shape)
    return l_b_m_x



def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    # print("******************************* in log_p_m_x")
    # print("log_Bs shape ", log_Bs.shape)
    # print("^^^^^^ log_BS  ", log_Bs)

    w_and_b = np.log(myTheta.omega) + log_Bs
    # print("w_and_b shape ", w_and_b.shape)

    denominator = LogsumExpTrick(w_and_b)
    # print("denominator shape ", denominator.shape)

    l_p_m_x = w_and_b - denominator
    # print("l_p_m_x shape ", l_p_m_x.shape)
    #
    # print("w_and_b  ", w_and_b)
    # print("denominator   ", denominator)
    # print("sunstract  ", l_p_m_x)

    return l_p_m_x


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m,
        which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    # print("*** in logLin")
    w_and_b = np.log(myTheta.omega) + log_Bs
    l_lik = np.sum(LogsumExpTrick(w_and_b))
    # print("l_lik  ", l_lik)
    return l_lik




def LogsumExpTrick(x):
    # to solve overflow issues, used The Log-Sum-Exp Trick based on this link:
    # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/ and
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html

    c = np.max(x, axis=0, keepdims=True)
    return c + logsumexp(x - c, axis=0, keepdims=True)


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""

    #### ????? should we make train also work for X: d and X: Txd ?????

    T = X.shape[0]
    # print("---- NOW in train")
    # print("X shape ", X.shape)
    myTheta = theta(speaker, M, X.shape[1])
    # print("my Theta name ", myTheta.name)
    # print("my Theta.mu ", myTheta.mu.shape)
    # print("my Theta.omega ", myTheta.omega.shape)
    # print("my Theta.sigma ", myTheta.Sigma.shape)

    # perform initialization (Slide 32)
    ### ???? should it be k-means ?????? for now just random thingy

    # initializing data based on slide 28 of tutorial A3
    # for omega, it should be 0<= omega <=1 and the sum should be 1 => 1/m for each
    init_omega = np.ones((M, 1)) / M
    myTheta.reset_omega(init_omega)

    # for mu, it is initialized to a random vector from data
    init_mu = X[np.random.randint(0, T, M)]
    myTheta.reset_mu(init_mu)

    # for sigma, it is initialized to identity matrix
    init_sigma = np.ones((M, myTheta._d))
    myTheta.reset_Sigma(init_sigma)

    # training based on Algorithm 1 in handout, page 9
    i = 0
    prev_L = -np.inf
    improvement = np.inf
    while i <= maxIter and improvement >= epsilon:
        ### compute intermediate results
        print("^^^^^ in train main loop i->", i)
        log_Bs = []
        for m in range(M):
            log_Bs.append(log_b_m_x(m, X, myTheta))

        log_Probs = log_p_m_x(np.array(log_Bs), myTheta)

        ### compute likelihood
        L = logLik(np.array(log_Bs), myTheta)

        ### Update parameters
        # print("&&& log_probs   ", log_Probs)
        lik = np.exp(log_Probs)
        denominator_lik = np.sum(lik, axis = 1, keepdims = True)

        # update omega
        new_omega = np.sum(lik, axis=1) / T
        # print("new omega ", new_omega, " sum  ", np.sum(new_omega))
        myTheta.reset_omega(new_omega)
        # update mu
        new_mu = (lik @ X) / denominator_lik
        # print("new mu ", new_mu)
        myTheta.reset_mu(new_mu)
        # update sigma
        new_sigma = ((lik @ (X ** 2)) / denominator_lik) - (myTheta.mu ** 2)
        # print("new sigma  ", new_sigma)
        myTheta.reset_Sigma(new_sigma)

        improvement = L - prev_L
        prev_L = L
        i += 1

        print(f"Iteration {i} likelihood of {round(prev_L, 4)} and dif of {round(improvement, 4)}")

    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    print(" ^^^^ in test ")
    print(mfcc)
    print(mfcc.shape)
    print("correctID  ", correctID)
    print("models ", models)
    print("models shapr ", len(models))

    

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print("TODO: you will need to modify this main block for Sec 2.3")
    print("ME: still don't know what 2.3 wants. skipped")
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            # X is Txd or d ???
            trainThetas.append(train(speaker, X, M, epsilon, maxIter))
            break
        break
    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
