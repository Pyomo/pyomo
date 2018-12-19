model = AbstractModel()
model.A = Set()

data = DataPortal()
data.load(filename='tab/A.tab', set=model.A)
instance = model.create(data)
