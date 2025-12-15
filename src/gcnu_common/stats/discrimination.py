
import numpy as np

def discriminativeLDA(Xs):
    """Computes discriminant Fisher LDA for C classes. Data is a list of C
    matrices. Each matrix contains the samples of a given class. Samples appear
    as columns in data matrices. Algorithm take from Fukunaga et al., 1990,
    Section 10.2.

    :param Xs: data for LDA
    :type Xs: list of numpy matrices

    :return: C-1 discriminative direction and their corresponding eigenvalues
    :rtype: tuple (numpy matrix with C-1 columns of discriminative directions,
    numpy array of corresponding C-1 eigenvalues)
    """

    L = len(Xs)
    n = Xs[0].shape[0]

    cLis = [Xs[i].shape[1] for i in range(L)]
    Ms = np.array([Xs[i].mean(axis=1) for i in range(L)]).T
    M0 = Ms.mean(axis=1)
    Sw = np.zeros(shape=(n, n), dtype=float)
    Sb = np.zeros(shape=(n, n), dtype=float)
    for i in range(L):
        aux = Ms[:, i] - M0
        Sb += cLis[i] * np.outer(aux, aux)
        for j in range(cLis[i]):
            aux = Xs[i][:, j] - Ms[:, i]
            Sw += np.outer(aux, aux)
    SwInvSb = np.linalg.solve(Sw, Sb)
    eigvals, eigvecs = np.linalg.eig(SwInvSb)
    indices = np.argsort(-eigvals)
    eigvecs, eigvals = eigvecs[:,indices], eigvals[indices]
    return eigvecs[:, :L-1], eigvals[:L-1]
