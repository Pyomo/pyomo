def doubleA_init(model):
    return (i*2 for i in model.A)
model.C = Set(initialize=DoubleA_init)
