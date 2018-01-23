def Z_init(model, i):
    if i > 10:
        return Set.End
    return 2*i+1