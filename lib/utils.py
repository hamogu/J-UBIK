import numpy as np
import nifty7 as ift
import scipy


def get_normed_exposure_operator(exposure_field, data_array):
    norm = (data_array[exposure_field.val !=0] / exposure_field.val[exposure_field.val!=0]).mean()
    normed_exp_field = exposure_field * norm
    normed_exp_field = ift.DiagonalOperator(normed_exp_field)
    return normed_exp_field


def prior_sample_plotter(opchain, n):
    pl = ift.Plot()
    for ii in range(n):
        f = ift.from_random(opchain.domain)
        tmp = opchain(f)
        pl.add(tmp)
    return pl.output()


def get_mask_operator(exp_field):
    mask = np.zeros(exp_field.shape)
    mask[exp_field.val==0] = 1
    mask_field = ift.Field.from_raw(exp_field.domain, mask)
    mask_operator = ift.MaskOperator(mask_field)
    return mask_operator
#FIXME actually here are pixels (Bad Pixels?) in the middle of the data which are kind of dead which are NOT included in the expfield
#this should be fixed, otherwise we could run into problems with the reconstruction


def convolve_operators(a, b):
    FFT = ift.FFTOperator(a.target)
    convolved = FFT.inverse(FFT(a.real)*FFT(b.real))
    return convolved.real


def convolve_field_operator(field, operator):
    FFT = ift.FFTOperator(operator.target)

    harmonic_field = FFT(field.real)
    fieldOp = ift.DiagonalOperator(harmonic_field.real)

    harmonic_operator = FFT @ operator.real
    convolved = FFT.inverse @ fieldOp @ harmonic_operator
    return convolved.real


class PositiveSumPriorOperator(ift.LinearOperator):
    """
    Operator performing a coordinate transformation, requiring MultiToTuple and Trafo.
    """
    def __init__(self, domain, target=None):
        self._domain = domain
        if not isinstance(self._domain, ift.MultiDomain):
            raise TypeError('domain must be a MultiDomain')
        if target == None:
            self._target = self._domain
        else:
            self._target = target
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._multi = MultiToTuple(self._domain)
        self._trafo = Trafo(self._multi.target)

    def apply(self, x, mode):
        self._check_input(x, mode)
        op = self._multi.adjoint @ self._trafo @ self._multi
        if mode == self.TIMES:
            res = op(x)
        else:
            res = op.adjoint(x)
        return res


class MultiToTuple(ift.LinearOperator):
    """
    Puts several Fields of a Multifield of the same domains, into a Domaintuple
    along a UnstructuredDomain.

    """
    def __init__(self, domain):
        self._domain = domain
        if not isinstance(self._domain, ift.MultiDomain):
            raise TypeError('domain has to be a ift.MultiDomain')
        self._first_dom = domain[domain.keys()[0]][0]
        for key in self._domain.keys():
            if not self._first_dom == domain[key][0]:
                raise TypeError('All sub domains must be equal ')
        n_doms = ift.UnstructuredDomain(len(domain.keys()))
        self._target = ift.makeDomain((n_doms, self._first_dom))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            lst = []
            for key in x.keys():
                lst.append(x[key].val)
            x = np.array(lst)
            res = ift.Field.from_raw(self._target, x)
        else:
            dct = {}
            ii = 0
            for key in self._domain.keys():
                tmp_field = ift.Field.from_raw(self._first_dom, x.val[ii,:,:])
                dct.update({key : tmp_field})
                ii += 1
            res = ift.MultiField.from_dict(dct)
        return res


class Trafo(ift.EndomorphicOperator):
    """
    #NOTE RENAME TRAFO
    This Operator performs a coordinate transformation into a coordinate system,
    in which the Oth component is the sum of all components of the former basis.
    """
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._n = self.domain.shape[0]
        self._build_mat()
        self._capability = self.TIMES | self.ADJOINT_TIMES
        lamb, s = self._build_mat()
        self._lamb = lamb
        if not np.isclose(lamb[0], 0):
            raise ValueError('Transformation does not work, check eigenvalues self._lamb')
        self._s = s
        if s[0, 0] < 0:
            s[:, 0] = -1 * s[:, 0]
        self._s_inv = scipy.linalg.inv(s)

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            y = np.einsum('ij, jmn->imn', self._s, x)
        else:
            y = np.einsum('ij, jmn-> imn', self._s_inv, x)
        return ift.Field.from_raw(self._tgt(mode), y)

    def _build_mat(self):
        l = self._n
        one = np.zeros([l]*2)
        np.fill_diagonal(one, 1)

        norm_d = np.ones([l]*2) / l
        proj = one - norm_d
        eigv, s = np.linalg.eigh(proj)
        return eigv, s


def get_distributions_for_positive_sum_prior(domain, number):
    for i in range(number):
        field_adapter = ift.FieldAdapter(domain, f'amp_{i}')
        tmp_operator = field_adapter.adjoint @ field_adapter
        if i == 0:
            operator = tmp_operator.exp()
        else:
            operator = operator + tmp_operator
    return operator


def makePositiveSumPrior(domain, number):
    distributions = get_distributions_for_positive_sum_prior(domain, number)
    positive_sum = PositiveSumPriorOperator(distributions.target)
    op = positive_sum @ distributions
    return op