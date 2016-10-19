import numpy as np
import scipy.stats as st


#
# def nextX(x, gradient, step_size):
#     return x + gradient + np.sqrt(2 * step_size) * np.random.normal(0, 1, gradient.shape)


def addNoise(d, step_size):
    return d + np.sqrt(2 * step_size) * np.random.normal(0, 1, d.shape)


"q(a|b)"


def proposal(a, b):
    q = st.multivariate_normal(np.abs(np.mean(b)), cov=1).pdf(a.reshape(-1))
    return q.reshape((1, -1))


def MH(input, prob, step_size):
    bestxx, grad_norm, upcode, pinput = prob(input)
    input1 = addNoise(upcode, step_size)

    #TODO this is not efficient will backprop, it must be fixed if it works.
    _, _, _, pinput1 = prob(input1)

    alpha = np.divide(pinput1 * proposal(input, input1), pinput * proposal(input1, input))
    r = np.minimum(1, alpha)
    u = np.random.uniform(0, 1, alpha.shape)

    r = np.where(u < r, input1, input)
    return bestxx, grad_norm, pinput, r
