import math


def consensus(solutions, ignore_zeros=True):
    #
    # Summarize the average value of solution values
    #
    # This currently assumes all solutions have the same variables
    #
    nsolutions = len(solutions)
    assert nsolutions > 1, "Need more than one solution to form a consensus pattern"
    keys = list(sorted(solutions[0]['variables'].keys()))

    total = {key:solutions[0]['variables'][key] for key in keys}
    for i in range(1, nsolutions):
        total = {key:(total[key] + solutions[i]['variables'][key]) for key in keys}

    mean = {key:total[key]/nsolutions for key in keys}

    if ignore_zeros:
        return {key:mean[key] for key in keys if math.fabs(mean[key]) > 1e-7}
    else:
        return mean
