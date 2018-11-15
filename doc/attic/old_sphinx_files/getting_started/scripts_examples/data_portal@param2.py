model.A = Set(initialize=['A1','A2','A3','A4'])
model.y = Param(model.A, default=0.0)

data.load(filename='tab/Y.tab', param=model.y)
