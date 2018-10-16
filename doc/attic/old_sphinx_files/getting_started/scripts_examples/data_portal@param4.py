model.A = Set()
model.x = Param(model.A)
model.w = Param(model.A)

data.load(filename='tab/XW.tab', param=(model.x,model.w), index=model.A)
