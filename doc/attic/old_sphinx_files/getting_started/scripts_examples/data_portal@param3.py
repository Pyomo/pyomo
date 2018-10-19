model.A = Set(initialize=['A1','A2','A3','A4'])
model.x = Param(model.A)
model.w = Param(model.A)

data.load(filename='tab/XW.tab', param=(model.x,model.w))
