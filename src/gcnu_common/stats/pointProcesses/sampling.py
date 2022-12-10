
import torch


def sampleInhomogeneousPP_thinning(CIF_times, CIF_values):
    """ Thining algorithm to sample from an inhomogeneous point process. Algorithm 2 from Yuanda Chen (2016). Thinning algorithms for simulating Point Prcesses.

    :param: CIF_times: times at which the CIF is evaluated
    :type: CIF_times: list like

    :param: CIF_values: values of the CIF evaluated at CIF_times
    :type: CIF_values: list like

    :return: (inhomogeneous, homogeneous): samples of the inhomogeneous and homogenous point process with CIF values CIF_values evaluated at CIF_times
    :rtype: tuple containing two lists
    """
    m = 0
    t = [CIF_times.min().item()]
    s = [CIF_times.min().item()]
    T = CIF_times.max().item()
    lambda_max = CIF_values.max().item()
    while s[m] < T:
        u = torch.rand(1)
        w = -torch.log(u).item()/lambda_max   # w~exponential(lambda_max)
        s.append(s[m]+w)        # {sm} homogeneous Poisson process
        D = torch.rand(1)
        CIF_index = (CIF_times-s[m+1]).abs().argmin()
        approxCIF_atSpike = CIF_values[CIF_index]
        if D <= approxCIF_atSpike/lambda_max:  # accepting with probability
            t.append(s[m+1])            # cifF(s[m+1])/lambda_max
                                               # {tn} inhomogeneous Poisson
                                               # process
        m += 1
    if t[-1] <= T:
        answer = {"inhomogeneous": t[1:], "homogeneous": s[1:]}
    else:
        answer = {"inhomogeneous": t[1:-1], "homogeneous": s[1:-1]}
    return answer


def sampleInhomogeneousPP_timeRescaling(CIF_times, CIF_values, dt=0.001):
    """ Time rescaling algorithm to sample from an inhomogeneous point
    process. Chapter 2 from Uri Eden's Point Process Notes.

    :param: CIF_times: times at which the CIF is evaluated
    :type: CIF_times: list like

    :param: CIF_values: values of the CIF evaluated at CIF_times
    :type: CIF_values: list like

    :param: dt: spike-time resolution.
    :type: dt: float

    :return: samples of the inhomogeneous point process with CIF values CIF_values evaluated at CIF_times
    :rtype: list
    """
    s = [CIF_times.min().item()]
    T = CIF_times.max().item()
    while s[-1] < T:
        u = torch.rand(1)
        z = -torch.log(u)   # z~exponential(1.0)
        t = s[-1]
        CIF_index = (CIF_times-t).abs().argmin()
        approxCIF_t = CIF_values[CIF_index]
        anInt = approxCIF_t*dt
        while anInt < z and t <= T:
            t += dt
            CIF_index = (CIF_times-t).abs().argmin()
            approxCIF_t = CIF_values[CIF_index].item()
            anInt += approxCIF_t*dt
        s.append(t)
    if s[-1] <= T:
        answer = s[1:]
    else:
        answer = s[1:-1]
    return answer
