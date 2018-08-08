model.A = Set()
model.w = Param(model.A)

data.load(filename='tab/XW.tab', select=('A','W'), param=model.w, index=model.A)
