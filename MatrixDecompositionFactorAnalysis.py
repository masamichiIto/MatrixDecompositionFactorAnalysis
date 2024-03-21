import numpy as np
import pandas as pd

class MatrixDecompositionFactorAnalysis:
    def __init__(self, ndim=2, max_iter = 300, tol=1e-4, trace=True, random_state=None):
        self.ndim = ndim
        self.max_iter = max_iter
        self.tol = 1e-4
        self.trace = trace
        self.random_state = random_state
        self.losses = [np.Inf]

    def _calc_loss(self, Sxx, Lam, Psi):
        loss = np.trace(Sxx - Lam@Lam.T - Psi**2)/np.trace(Sxx)
        return loss

    def _eigh_my(self, symM):
        evals, evecs = np.linalg.eigh(symM)
        sort_idx = np.argsort(evals)[::-1] # idx for reordering in decending order
        return evals[sort_idx], evecs[:,sort_idx]

    def _Sxz_step(self, Sxx, Lam, Psi):
        B = np.concatenate((Lam, Psi), axis=1)
        BtSxxB = B.T@Sxx@B
        eval_bsb, evec_bsb = self._eigh_my(BtSxxB)
        eval_bsb[eval_bsb < 0] = 0 # BtSxxBはランク落ちしてるため，数値計算上ごく小さい負の数が発生する．その後の計算のためにそれを0に置き換える
        Sxz = (B.T@np.linalg.inv(B@B.T)).T@evec_bsb@np.diag(np.sqrt(eval_bsb))@evec_bsb.T # Sxz = B'^+L\Delta L'
        return Sxz

    def _B_step(self, Sxz):
        Lam = Sxz[:,:self.ndim]
        Psi = np.diag(np.diag(Sxz[:,self.ndim:]))
        return Lam, Psi

    def _initialize(self, Sxx):
        vals, vec = self._eigh_my(Sxx)
        sqrt_vals = np.sqrt(vals)
        Lam = vec[:,:self.ndim]@np.diag(sqrt_vals[:self.ndim])
        Psi = np.diag(np.sqrt(np.diag(Sxx - Lam@Lam.T)))
        return Lam, Psi

    def fit(self, data, is_corr_given):
        if not is_corr_given:
            Sxx = np.corrcoef(data.T)
        else:
            Sxx = data
        #print(self._eigh_my(Sxx))

        # initialize parameters
        Lam, Psi = self._initialize(Sxx)
        #print(Lam)
        #print(Psi)
        # iterative update of Sxz and B = [Lam, Psi]
        for iter in range(self.max_iter):
            # update Sxz
            Sxz = self._Sxz_step(Sxx, Lam, Psi)

            # update B = [Lam, Psi]
            Lam, Psi = self._B_step(Sxz)

            # calculate loss function value
            loss = self._calc_loss(Sxx, Lam, Psi)

            if self.trace:
              print("{0}: {1:.4f}".format(iter+1, loss))

            self.losses.append(loss)
            if 0 <= self.losses[iter] - self.losses[iter+1] <= self.tol:
                self.est_Lam = Lam
                self.est_Psi = Psi
                self.min_loss = loss
                break


"""
# synthesize data
Lam = np.array([[0.8,0],
                [0.9,0],
                [0.7,0],
                [0.6,0],
                [0,-0.9],
                [0,0.9],
                [0,-0.7],
                [0,0.8]])
Psi = np.sqrt(np.diag(1-np.diag(Lam@Lam.T)))

cov = Lam@Lam.T + Psi**2
mean = np.zeros(8)
X = np.random.multivariate_normal(mean, cov, size=1000) # X ~ MVN((0,...,0).T, Lam Lam' + Psi^2)

# fit data to MDFA model
mdfa = MatrixDecompositionFactorAnalysis(ndim=2, max_iter=300, tol = 1e-6, trace=True)
mdfa.fit(X, is_corr_given=False)

# look results
mdfa.est_Lam, mdfa.est_Psi
"""