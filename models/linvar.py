from statsmodels.tsa.api import VAR
import statsmodels.stats.multitest as multitest

import numpy as np


class LinVAR:
    def __init__(self, X: np.ndarray, K=1):
        """
        Linear VAR model.

        @param X: numpy array with data of shape T x p.
        @param K: order of the VAR model (maximum lag).
        """
        # X.shape: T x p
        super(LinVAR, self).__init__()

        self.model = VAR(X)
        self.p = X.shape[1]
        self.K = K

        # Fit the model
        self.model_results = self.model.fit(maxlags=self.K)

    def infer_causal_structure(self, kind="f", adjust=False, signed=False):
        """
        Infer GC based on the fitted VAR model.

        @param kind: type of the statistical test for GC (as implemented within statsmodels). Default: F-test.
        @param adjust: whether to adjust p-values? If True, p-values are adjusted using the Benjamini-Hochberg procedure
        for controlling the FDR.
        @param signed: whether to return coeffcient signs?
        @return: p x p array with p-values, p x p array with hypothesis test results, and, if signed == True,
        p x p array with coefficient signs.
        """
        pvals = np.zeros((self.p, self.p))
        reject = None
        for i in range(self.p):
            for j in range(self.p):
                pvals[i, j] = self.model_results.test_causality(caused=i, causing=j, kind=kind).pvalue
        reject = pvals <= 0.05
        if adjust:
            reject, pvals, alpha_Sidak, alpha_Bonf = multitest.multipletests(pvals.ravel(), method="fdr_bh")
            pvals = np.reshape(pvals, (self.p, self.p))
            reject = np.reshape(reject, (self.p, self.p))
        if signed:
            return pvals, reject, np.sign(self.model_results.params[1:, :].T * reject)
        else:
            return pvals, reject
