import pandas as pd

from pyomo.core.base import (Constraint, Param, Set, RangeSet, Var, value,
                             SortComponents)


def write_excel(model):
    objects = list(model.component_objects(sort=SortComponents.alphaOrder,
                                           active=True))

    names = []
    for obj in objects:
        if ((obj.type() in (Var, Constraint, Param) and obj.dim() > 0) or
            (obj.type() is Set and obj.dim() == 0)):
            names.append('=HYPERLINK("#%s!A1", "%s")' % (obj.name, obj.name))
        else:
            names.append(obj.name)
    toc = {'Name':names,
           'Type':[obj.type().__name__ for obj in objects],
           'Dim':[obj.dim() for obj in objects],
           'Count':[len(obj) if obj.type() is Set else len(obj.index_set())
                    for obj in objects]}
    TOC = pd.DataFrame(toc)
    TOC = TOC.set_index('Name')
    TOC = TOC[['Type', 'Dim', 'Count']]

    obj_frames = list()

    for obj in objects:
        data = dict()

        if obj.type() in (Var, Constraint, Param) and obj.dim() > 0:
            if obj.dim() == 1:
                indices = [obj.index_set().name]
            else:
                indices = [ind.name for ind in obj._implicit_subsets]
            for ind in indices:
                data[ind] = list()

            if obj.type() is Var:
                data['Value'] = list()
                data['Lower'] = list()
                data['Upper'] = list()
            elif obj.type() is Constraint:
                data['Body'] = list()
                data['Lower'] = list()
                data['Upper'] = list()
            elif obj.type() is Param:
                data['Value'] = list()

            if obj.type() in (Var, Constraint):
                data_objects = sorted(obj.values(),
                                      key=lambda dobj: dobj.index())
                for dobj in data_objects:
                    for i in range(len(indices)):
                        data[indices[i]].append(dobj.index() if obj.dim() < 2
                                                else dobj.index()[i])
                    if obj.type() is Var:
                        data['Lower'].append(dobj.lb)
                        data['Value'].append(dobj.value)
                        data['Upper'].append(dobj.ub)
                    else:
                        data['Lower'].append(value(dobj.lower))
                        data['Body'].append(value(dobj.body))
                        data['Upper'].append(value(dobj.upper))

            elif obj.type() is Param:
                for ind, val in obj.iteritems():
                    for i in range(len(indices)):
                        data[indices[i]].append(ind if obj.dim() < 2
                                                else ind[i])
                    data['Value'].append(val)

            df = pd.DataFrame(data)
            df = df.set_index(indices)
            if obj.type() is Var:
                df = df[['Lower', 'Value', 'Upper']]
            elif obj.type is Constraint:
                df = df[['Lower', 'Body', 'Upper']]

        elif obj.type() is Set and obj.dim() == 0:
            data['Items'] = sorted(obj.data())
            df = pd.DataFrame(data)

        obj_frames.append((obj, df))

    with pd.ExcelWriter('ExcelFile.xlsx') as writer:
        TOC.to_excel(writer, sheet_name='TOC')
        for obj, df in obj_frames:
            df.to_excel(writer, sheet_name=obj.name)
