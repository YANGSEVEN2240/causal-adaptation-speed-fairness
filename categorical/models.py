import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn, optim

from categorical.utils import kullback_leibler, logit2proba, logsumexp, proba2logit


def joint2conditional(joint):
    marginal = np.sum(joint, axis=-1)
    conditional = joint / np.expand_dims(marginal, axis=-1)

    return CategoricalStatic(marginal, conditional)


def jointlogit2conditional(joint, is_btoa):
    sa = logsumexp(joint)
    sa -= sa.mean(axis=1, keepdims=True)
    sba = joint - sa[:, :, np.newaxis]
    sba -= sba.mean(axis=2, keepdims=True)

    return CategoricalStatic(sa, sba, from_probas=False, is_btoa=is_btoa)


def sample_joint(k, n, concentration=1, dense=False, logits=True):
    """Sample n causal mechanisms of categorical variables of dimension K.

    The concentration argument specifies the concentration of the resulting cause marginal.
    """
    if logits:
        sa = stats.loggamma.rvs(concentration, size=(n, k))
        sa -= sa.mean(axis=1, keepdims=True)

        conditional_concentration = concentration if dense else concentration / k
        if conditional_concentration > 0.1:
            sba = stats.loggamma.rvs(conditional_concentration, size=(n, k, k))
        else:
            # A loggamma with small shape parameter is well approximated
            # by a negative exponential with parameter scale = 1/ shape
            sba = - stats.expon.rvs(scale=1 / conditional_concentration, size=(n, k, k))
        sba -= sba.mean(axis=2, keepdims=True)
        return CategoricalStatic(sa, sba, from_probas=False)
    else:
        pa = np.random.dirichlet(concentration * np.ones(k), size=n)
        condconcentration = concentration if dense else concentration / k
        pba = np.random.dirichlet(condconcentration * np.ones(k), size=[n, k])
        return CategoricalStatic(pa, pba, from_probas=True)


def sample_fairness_joint(k_a, k_x, k_y, n, concentration=1, dense=False, logits=True):
    """Sample n causal mechanisms for three variables (A, X, Y)."""
    if logits:
        s_a = stats.loggamma.rvs(concentration, size=(n, k_a))
        s_a -= s_a.mean(axis=1, keepdims=True)

        cond_conc_x = concentration if dense else concentration / k_x
        if cond_conc_x > 0.1:
            s_x_given_a = stats.loggamma.rvs(cond_conc_x, size=(n, k_a, k_x))
        else:
            s_x_given_a = -stats.expon.rvs(scale=1 / cond_conc_x, size=(n, k_a, k_x))
        s_x_given_a -= s_x_given_a.mean(axis=2, keepdims=True)

        cond_conc_y = concentration if dense else concentration / k_y
        if cond_conc_y > 0.1:
            s_y_given_ax = stats.loggamma.rvs(cond_conc_y, size=(n, k_a, k_x, k_y))
        else:
            s_y_given_ax = -stats.expon.rvs(scale=1 / cond_conc_y, size=(n, k_a, k_x, k_y))
        s_y_given_ax -= s_y_given_ax.mean(axis=3, keepdims=True)

        return FairnessCategoricalStatic(s_a, s_x_given_a, s_y_given_ax, from_probas=False)
    else:
        p_a = np.random.dirichlet(concentration * np.ones(k_a), size=n)
        cond_conc_x = concentration if dense else concentration / k_x
        p_x_given_a = np.random.dirichlet(cond_conc_x * np.ones(k_x), size=[n, k_a])
        cond_conc_y = concentration if dense else concentration / k_y
        p_y_given_ax = np.random.dirichlet(cond_conc_y * np.ones(k_y), size=[n, k_a, k_x])
        return FairnessCategoricalStatic(p_a, p_x_given_a, p_y_given_ax, from_probas=True)


class CategoricalStatic:
    """Represent n categorical distributions of variables (a,b) of dimension k each."""

    def __init__(self, marginal, conditional, from_probas=True, is_btoa=False):
        """The distribution is represented by a marginal p(a) and a conditional p(b|a)

        marginal is n*k array.
        conditional is n*k*k array. Each element conditional[i,j,k] is p_i(b=k |a=j)
        """
        self.n, self.k = marginal.shape
        self.BtoA = is_btoa

        if not conditional.shape == (self.n, self.k, self.k):
            raise ValueError(
                f'Marginal shape {marginal.shape} and conditional '
                f'shape {conditional.shape} do not match.')

        if from_probas:
            self.marginal = marginal
            self.conditional = conditional
            self.sa = proba2logit(marginal)
            self.sba = proba2logit(conditional)
        else:
            self.marginal = logit2proba(marginal)
            self.conditional = logit2proba(conditional)
            self.sa = marginal
            self.sba = conditional

    def to_joint(self, return_probas=True):
        if return_probas:
            return self.conditional * self.marginal[:, :, np.newaxis]
        else:  # return logits
            joint = self.sba \
                    + (self.sa - logsumexp(self.sba))[:, :, np.newaxis]
            return joint - np.mean(joint, axis=(1, 2), keepdims=True)

    def reverse(self):
        """Return conditional from b to a.
        Compute marginal pb and conditional pab such that pab*pb = pba*pa.
        """
        joint = self.to_joint(return_probas=False)
        joint = np.swapaxes(joint, 1, 2)  # invert variables
        return jointlogit2conditional(joint, not self.BtoA)

    def probadist(self, other):
        pd = np.sum((self.marginal - other.marginal) ** 2, axis=1)
        pd += np.sum((self.conditional - other.conditional) ** 2, axis=(1, 2))
        return pd

    def scoredist(self, other):
        sd = np.sum((self.sa - other.sa) ** 2, axis=1)
        sd += np.sum((self.sba - other.sba) ** 2, axis=(1, 2))
        return sd

    def sqdistance(self, other):
        """Return the squared euclidean distance between self and other"""
        return self.probadist(other), self.scoredist(other)

    def kullback_leibler(self, other):
        p0 = self.to_joint().reshape(self.n, self.k ** 2)
        p1 = other.to_joint().reshape(self.n, self.k ** 2)
        return kullback_leibler(p0, p1)

    def intervention(self, on, concentration=1, dense=True):
        # sample new marginal
        if on == 'independent':
            # make cause and effect independent,
            # but without changing the effect marginal.
            newmarginal = self.reverse().marginal
        elif on == 'geometric':
            newmarginal = logit2proba(self.sba.mean(axis=1))
        elif on == 'weightedgeo':
            newmarginal = logit2proba(np.sum(self.sba * self.marginal[:, :, None], axis=1))
        else:
            newmarginal = np.random.dirichlet(concentration * np.ones(self.k), size=self.n)

        # TODO use logits of the marginal for stability certainty
        # replace the cause or the effect by this marginal
        if on == 'cause':
            return CategoricalStatic(newmarginal, self.conditional)
        elif on in ['effect', 'independent', 'geometric', 'weightedgeo']:
            # intervention on effect
            newconditional = np.repeat(newmarginal[:, None, :], self.k, axis=1)
            return CategoricalStatic(self.marginal, newconditional)
        elif on == 'mechanism':
            # sample a new mechanism from the same prior
            sba = sample_joint(self.k, self.n, concentration, dense, logits=True).sba
            return CategoricalStatic(self.sa, sba, from_probas=False)
        elif on == 'gmechanism':
            # sample from a gaussian centered on each conditional
            sba = np.random.normal(self.sba, self.sba.std())
            sba -= sba.mean(axis=2, keepdims=True)
            return CategoricalStatic(self.sa, sba, from_probas=False)
        elif on == 'singlecond':
            newscores = stats.loggamma.rvs(concentration, size=(self.n, self.k))
            newscores -= newscores.mean(1, keepdims=True)
            # if 'simple':
            #     a0 = 0
            # elif 'max':
            a0 = np.argmax(self.sa, axis=1)
            sba = self.sba.copy()
            sba[np.arange(self.n), a0] = newscores
            return CategoricalStatic(self.sa, sba, from_probas=False)
        else:
            raise ValueError(f'Intervention on {on} is not supported.')

    def sample(self, m, return_tensor=False):
        """For each of the n distributions, return m samples. (n*m*2 array) """
        flatjoints = self.to_joint().reshape((self.n, self.k ** 2))
        samples = np.array(
            [np.random.choice(self.k ** 2, size=m, p=p) for p in flatjoints])
        a = samples // self.k
        b = samples % self.k
        if not return_tensor:
            return a, b
        else:
            return torch.from_numpy(a), torch.from_numpy(b)

    def to_module(self):
        return CategoricalModule(self.sa, self.sba, is_btoa=self.BtoA)

    def __repr__(self):
        return (f"n={self.n} categorical of dimension k={self.k}\n"
                f"{self.marginal}\n"
                f"{self.conditional}")


def test_ConditionalStatic():
    print('test categorical static')

    # test the reversion formula on a known example
    pa = np.array([[.5, .5]])
    pba = np.array([[[.5, .5], [1 / 3, 2 / 3]]])
    anspb = np.array([[5 / 12, 7 / 12]])
    anspab = np.array([[[3 / 5, 2 / 5], [3 / 7, 4 / 7]]])

    test = CategoricalStatic(pa, pba).reverse()
    answer = CategoricalStatic(anspb, anspab)

    probadist, scoredist = test.sqdistance(answer)
    assert probadist < 1e-4, probadist
    assert scoredist < 1e-4, scoredist

    # ensure that reverse is reversible
    distrib = sample_joint(3, 17, 1, True)
    assert np.allclose(0, distrib.reverse().reverse().sqdistance(distrib))

    distrib.kullback_leibler(distrib.reverse())
    n = 10000
    a, b = distrib.sample(n)
    c = a * distrib.k + b
    val, approx = np.unique(c[0], return_counts=True)
    approx = approx.astype(float) / n
    joint = distrib.to_joint()[0].flatten()
    assert np.allclose(joint, approx, atol=1e-2, rtol=1e-1), print(joint, approx)


class CategoricalModule(nn.Module):
    """Represent n categorical conditionals as a pytorch module"""

    def __init__(self, sa, sba, is_btoa=False):
        super(CategoricalModule, self).__init__()
        self.n, self.k = tuple(sa.shape)

        sa = sa.clone().detach() if torch.is_tensor(sa) else torch.tensor(sa)
        sba = sba.clone().detach() if torch.is_tensor(sba) else torch.tensor(sba)
        self.sa = nn.Parameter(sa.to(torch.float32))
        self.sba = nn.Parameter(sba.to(torch.float32))
        self.BtoA = is_btoa

    def forward(self, a, b):
        """
        :param a: n*m collection of m class in {1,..., k} observed
        for each of the n models
        :param b: n*m like a
        :return: the log-probability of observing a,b,
        where model 1 explains first row of a,b,
        model 2 explains row 2 and so forth.
        """
        batch_size = a.shape[1]
        if self.BtoA:
            a, b = b, a
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, batch_size)
        return self.to_joint()[rows.view(-1), a.view(-1).long(), b.view(-1).long()].view(self.n, batch_size)

    def to_joint(self):
        return F.log_softmax(self.sba, dim=2) \
            + F.log_softmax(self.sa, dim=1).unsqueeze(dim=2)

    def to_static(self):
        return CategoricalStatic(
            logit2proba(self.sa.detach().numpy()),
            logit2proba(self.sba.detach().numpy())
        )

    def kullback_leibler(self, other):
        joint = self.to_joint()
        return torch.sum((joint - other.to_joint()) * torch.exp(joint),
                         dim=(1, 2))

    def scoredist(self, other):
        return torch.sum((self.sa - other.sa) ** 2, dim=1) \
            + torch.sum((self.sba - other.sba) ** 2, dim=(1, 2))

    def __repr__(self):
        return f"CategoricalModule(joint={self.to_joint().detach()})"


def test_CategoricalModule(n=7, k=5):
    print('test categorical module')
    references = sample_joint(k, n, 1)
    intervened = references.intervention(on='cause', concentration=1)

    modules = references.to_module()

    # test that reverse is numerically stable
    kls = references.reverse().reverse().to_module().kullback_leibler(modules)
    assert torch.allclose(torch.zeros(n), kls), kls

    # test optimization
    optimizer = optim.SGD(modules.parameters(), lr=1)
    aa, bb = intervened.sample(13, return_tensor=True)
    negativeloglikelihoods = -modules(aa, bb).mean()
    optimizer.zero_grad()
    negativeloglikelihoods.backward()
    optimizer.step()

    imodules = intervened.to_module()
    imodules.kullback_leibler(modules)
    imodules.scoredist(modules)


class JointModule(nn.Module):

    def __init__(self, logits):
        super(JointModule, self).__init__()
        self.n, k2 = logits.shape  # logits is flat

        self.k = int(np.sqrt(k2))
        # if self.k ** 2 != k2:
        #     raise ValueError('Logits matrix can not be reshaped to square.')

        # normalize to sum to 0
        logits = logits - logits.mean(dim=1, keepdim=True)
        self.logits = nn.Parameter(logits)

    @property
    def logpartition(self):
        return torch.logsumexp(self.logits, dim=1)

    def forward(self, a, b):
        batch_size = a.shape[1]
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, batch_size).view(-1)
        index = (a * self.k + b).view(-1)
        return F.log_softmax(self.logits, dim=1)[rows.long(), index.long()].view(self.n, batch_size)

    def kullback_leibler(self, other):
        a = self.logpartition
        kl = torch.sum((self.logits - other.logits) * torch.exp(self.logits - a[:, None]), dim=1)
        return kl - a + other.logpartition

    def scoredist(self, other):
        return torch.sum((self.logits - other.logits) ** 2, dim=1)

    def __repr__(self):
        return f"CategoricalJoint(logits={self.logits.detach()})"


class Counter:

    def __init__(self, counts):
        self.counts = counts
        self.n, self.k, self.k2 = counts.shape

    @property
    def total(self):
        return self.counts.sum(axis=(1, 2), keepdims=True)

    # @jit
    def update(self, a: np.ndarray, b: np.ndarray):
        for aaa, bbb in zip(a.T, b.T):
            self.counts[np.arange(self.n), aaa, bbb] += 1


def test_Counter():
    c = Counter(np.zeros([1, 2, 2]))
    c.update(np.array([[0, 0, 0, 1]]), np.array([[0, 0, 1, 1]]))
    assert c.total == 4
    assert np.allclose(c.counts / c.total, [[.5, .25], [0, .25]])


class JointMAP:

    def __init__(self, prior, counter):
        self.prior = prior
        self.n0 = self.prior.sum(axis=(1, 2), keepdims=True)
        self.counter = counter

    @property
    def frequencies(self):
        return ((self.prior + self.counter.counts) /
                (self.n0 + self.counter.total))

    def to_joint(self):
        return np.log(self.frequencies)



class FairnessCategoricalStatic:
    """Three-variable distribution: P(A), P(X|A), P(Y|A,X)"""

    def __init__(self, p_a, p_x_given_a, p_y_given_ax, from_probas=True, is_anticausal=False):
        self.n = p_a.shape[0]
        self.k_a = p_a.shape[1]
        self.k_x = p_x_given_a.shape[2]
        self.k_y = p_y_given_ax.shape[3]
        self.is_anticausal = is_anticausal

        if from_probas:
            self.p_a, self.p_x_given_a, self.p_y_given_ax = p_a, p_x_given_a, p_y_given_ax
            self.s_a = proba2logit(p_a)
            self.s_x_given_a = proba2logit(p_x_given_a)
            self.s_y_given_ax = proba2logit(p_y_given_ax)
        else:
            self.s_a, self.s_x_given_a, self.s_y_given_ax = p_a, p_x_given_a, p_y_given_ax
            self.p_a = logit2proba(p_a)
            self.p_x_given_a = logit2proba(p_x_given_a)
            self.p_y_given_ax = logit2proba(p_y_given_ax)

    def to_joint(self, return_probas=True):
        if return_probas:
            return (self.p_a[:, :, None, None] *
                    self.p_x_given_a[:, :, :, None] *
                    self.p_y_given_ax)
        else:
            joint = (self.s_a[:, :, None, None] +
                     self.s_x_given_a[:, :, :, None] +
                     self.s_y_given_ax)
            return joint - joint.mean(axis=(1, 2, 3), keepdims=True)

    def reverse(self):
        # 基于概率计算反方向条件分布，避免 logits 均值减法导致的偏差
        joint = self.to_joint(return_probas=True)  # [n, k_a, k_x, k_y]

        # p(a,y) = sum_x p(a,x,y)
        p_ay = np.sum(joint, axis=2)  # [n, k_a, k_y]
        p_y_given_a = p_ay / np.sum(p_ay, axis=2, keepdims=True)  # [n, k_a, k_y]

        # p(x|a,y) = p(a,x,y) / p(a,y)
        p_x_given_ay = joint / p_ay[:, :, None, :]  # [n, k_a, k_x, k_y]
        # 转置为 [n, k_a, k_y, k_x] 以匹配 FairnessAntiCausalModule
        p_x_given_ay = p_x_given_ay.transpose(0, 1, 3, 2)  # [n, k_a, k_y, k_x]

        # 转回 logits
        s_y_given_a = proba2logit(p_y_given_a)
        s_x_given_ay = proba2logit(p_x_given_ay)

        return FairnessCategoricalStatic(
            self.s_a, s_y_given_a, s_x_given_ay,
            from_probas=False, is_anticausal=True
        )

    def scoredist(self, other):
        dist = np.sum((self.s_a - other.s_a) ** 2, axis=1)
        dist += np.sum((self.s_x_given_a - other.s_x_given_a) ** 2, axis=(1, 2))
        dist += np.sum((self.s_y_given_ax - other.s_y_given_ax) ** 2, axis=(1, 2, 3))
        return dist

    def kullback_leibler(self, other):
        p0 = self.to_joint().reshape(self.n, -1)
        p1 = other.to_joint().reshape(other.n, -1)
        return kullback_leibler(p0, p1)

    def intervention(self, on, concentration=1, dense=True):
        if on == 'A':
            new_p_a = np.random.dirichlet(concentration * np.ones(self.k_a), size=self.n)
            return FairnessCategoricalStatic(new_p_a, self.p_x_given_a, self.p_y_given_ax,
                                             is_anticausal=self.is_anticausal)
        elif on == 'X':
            new_p_x = np.random.dirichlet(concentration * np.ones(self.k_x), size=self.n)
            new_p_x_given_a = np.repeat(new_p_x[:, None, :], self.k_a, axis=1)
            return FairnessCategoricalStatic(self.p_a, new_p_x_given_a, self.p_y_given_ax,
                                             is_anticausal=self.is_anticausal)
        elif on == 'AX':
            new_p_a = np.random.dirichlet(concentration * np.ones(self.k_a), size=self.n)
            new_p_x = np.random.dirichlet(concentration * np.ones(self.k_x), size=self.n)
            new_p_x_given_a = np.repeat(new_p_x[:, None, :], self.k_a, axis=1)
            return FairnessCategoricalStatic(new_p_a, new_p_x_given_a, self.p_y_given_ax,
                                             is_anticausal=self.is_anticausal)
        elif on == 'Y':
            # 采样新的边际分布 p*(y)，独立于 A 和 X
            new_s_y = stats.loggamma.rvs(concentration, size=(self.n, self.k_y))
            new_s_y -= new_s_y.mean(axis=1, keepdims=True)
            # 扩展为 [n, k_a, k_x, k_y]，对所有 a,x 相同
            new_s_y_given_ax = np.repeat(
                np.repeat(new_s_y[:, None, None, :], self.k_a, axis=1),
                self.k_x, axis=2
            )
            return FairnessCategoricalStatic(
                self.s_a, self.s_x_given_a, new_s_y_given_ax,
                from_probas=False, is_anticausal=self.is_anticausal
            )
        else:
            raise ValueError(f'Intervention on {on} not supported.')

    def sample(self, m, return_tensor=False):
        flat_joint = self.to_joint().reshape((self.n, -1))
        total = self.k_a * self.k_x * self.k_y
        samples = np.array([np.random.choice(total, size=m, p=p) for p in flat_joint])
        a = samples // (self.k_x * self.k_y)
        remainder = samples % (self.k_x * self.k_y)
        x = remainder // self.k_y
        y = remainder % self.k_y
        if not return_tensor:
            return a, x, y
        else:
            return torch.from_numpy(a), torch.from_numpy(x), torch.from_numpy(y)

    def to_module(self):
        if self.is_anticausal:
            return FairnessAntiCausalModule(self.s_a, self.s_x_given_a, self.s_y_given_ax, is_btoa=True)
        else:
            return FairnessCausalModule(self.s_a, self.s_x_given_a, self.s_y_given_ax, is_btoa=False)

class FairnessCausalModule(nn.Module):
    """Causal: P(A) * P(X|A) * P(Y|A,X)"""

    def __init__(self, s_a, s_x_given_a, s_y_given_ax, is_btoa=False):
        super(FairnessCausalModule, self).__init__()
        self.n, self.k_a = s_a.shape
        self.k_x = s_x_given_a.shape[2]
        self.k_y = s_y_given_ax.shape[3]
        self.BtoA = is_btoa
        self.s_a = nn.Parameter(torch.FloatTensor(s_a))
        self.s_x_given_a = nn.Parameter(torch.FloatTensor(s_x_given_a))
        self.s_y_given_ax = nn.Parameter(torch.FloatTensor(s_y_given_ax))

    def to_joint(self):
        return (self.s_a[:, :, None, None] +
                self.s_x_given_a[:, :, :, None] +
                self.s_y_given_ax)

    def forward(self, a, x, y):
        batch_size = a.shape[1]
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, batch_size)
        log_p_a = F.log_softmax(self.s_a, dim=1)[rows.view(-1), a.view(-1).long()].view(self.n, batch_size)
        log_p_x_a = F.log_softmax(self.s_x_given_a, dim=2)[
            rows.view(-1), a.view(-1).long(), x.view(-1).long()
        ].view(self.n, batch_size)
        log_p_y_ax = F.log_softmax(self.s_y_given_ax, dim=3)[
            rows.view(-1), a.view(-1).long(), x.view(-1).long(), y.view(-1).long()
        ].view(self.n, batch_size)
        return log_p_a + log_p_x_a + log_p_y_ax

    def kullback_leibler(self, other):
        joint = self.to_joint()
        other_joint = other.to_joint()
        log_joint = F.log_softmax(joint.reshape(self.n, -1), dim=1)
        log_other = F.log_softmax(other_joint.reshape(self.n, -1), dim=1)
        return torch.sum(torch.exp(log_joint) * (log_joint - log_other), dim=1)

    # ... existing code ...

    def scoredist(self, other):
        dist = torch.sum((self.s_a - other.s_a) ** 2, dim=1)
        dist += torch.sum((self.s_x_given_a - other.s_x_given_a) ** 2, dim=(1, 2))
        dist += torch.sum((self.s_y_given_ax - other.s_y_given_ax) ** 2, dim=(1, 2, 3))
        return dist


class FairnessAntiCausalModule(nn.Module):
    """Anticausal: P(A) * P(Y|A) * P(X|A,Y)"""

    def __init__(self, s_a, s_y_given_a, s_x_given_ay, is_btoa=True):
        super(FairnessAntiCausalModule, self).__init__()
        self.n, self.k_a = s_a.shape
        self.k_y = s_y_given_a.shape[2]
        self.k_x = s_x_given_ay.shape[3]
        self.BtoA = is_btoa
        self.s_a = nn.Parameter(torch.FloatTensor(s_a))
        self.s_y_given_a = nn.Parameter(torch.FloatTensor(s_y_given_a))
        self.s_x_given_ay = nn.Parameter(torch.FloatTensor(s_x_given_ay))

    def to_joint(self):
        s_a_exp = self.s_a[:, :, None, None]
        s_y_a_exp = self.s_y_given_a[:, :, None, :]
        s_x_ay_trans = self.s_x_given_ay.permute(0, 1, 3, 2)
        return s_a_exp + s_y_a_exp + s_x_ay_trans

    def forward(self, a, x, y):
        batch_size = a.shape[1]
        rows = torch.arange(0, self.n).unsqueeze(1).repeat(1, batch_size)
        log_p_a = F.log_softmax(self.s_a, dim=1)[rows.view(-1), a.view(-1).long()].view(self.n, batch_size)
        log_p_y_a = F.log_softmax(self.s_y_given_a, dim=2)[
            rows.view(-1), a.view(-1).long(), y.view(-1).long()
        ].view(self.n, batch_size)
        log_p_x_ay = F.log_softmax(self.s_x_given_ay, dim=3)[
            rows.view(-1), a.view(-1).long(), y.view(-1).long(), x.view(-1).long()
        ].view(self.n, batch_size)
        return log_p_a + log_p_y_a + log_p_x_ay

    def kullback_leibler(self, other):
        joint = self.to_joint()
        other_joint = other.to_joint()
        log_joint = F.log_softmax(joint.reshape(self.n, -1), dim=1)
        log_other = F.log_softmax(other_joint.reshape(self.n, -1), dim=1)
        return torch.sum(torch.exp(log_joint) * (log_joint - log_other), dim=1)

    # ... existing code ...

    def scoredist(self, other):
        dist = torch.sum((self.s_a - other.s_a) ** 2, dim=1)
        dist += torch.sum((self.s_y_given_a - other.s_y_given_a) ** 2, dim=(1, 2))
        dist += torch.sum((self.s_x_given_ay - other.s_x_given_ay) ** 2, dim=(1, 2, 3))
        return dist


if __name__ == "__main__":
    print("hi")
    test_ConditionalStatic()
    test_CategoricalModule()
    test_Counter()
