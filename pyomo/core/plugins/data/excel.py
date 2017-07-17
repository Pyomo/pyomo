import pandas as pd

from pyomo.core.base import (Constraint, Param, Set, RangeSet, Var, Objective,
                             Block, Suffix, Expression, value, SortComponents)


def write_excel(model, filename=None, engine=None):
    valid_ctypes = set([Var, Constraint, Param, Set, RangeSet,
                        Expression, Objective, Block])
    objects = []
    scalars = dict()
    scalars['Param'] = []
    scalars['Var'] = []
    scalars['Constraint'] = []
    scalars['Objective'] = []
    for obj in model.component_objects(sort=SortComponents.alphaOrder,
                                       active=True):
        if obj.type() in valid_ctypes:
            objects.append(obj)
        if obj.type().__name__ in scalars and obj.dim() == 0:
            scalars[obj.type().__name__].append(obj)

    toc = {'Name'  : [obj.name for obj in objects],
           'Type'  : [obj.type().__name__ for obj in objects],
           'Dim'   : [obj.dim() for obj in objects],
           'Count' : [len(obj) for obj in objects],
           'Doc'   : [obj.doc for obj in objects]}
    TOC = pd.DataFrame(toc)
    TOC = TOC.set_index('Name')
    TOC = TOC[['Type', 'Dim', 'Count', 'Doc']]

    has_dual = has_rc = False
    for suf in model.component_data_objects(Suffix, active=True):
        if (suf.name == 'dual' and suf.import_enabled()):
            has_dual = True
        elif (suf.name == 'rc' and suf.import_enabled()):
            has_rc = True

    obj_frames = []

    for obj in objects:
        data = dict()

        if obj.type() in (Var, Constraint, Param) and obj.dim() > 0:
            if obj.dim() == 1:
                indices = [obj.index_set().name]
            else:
                indices = [ind.name for ind in obj._implicit_subsets]
            for ind in indices:
                data[ind] = []

            if obj.type() is Var:
                data['Value'] = []
                data['Lower'] = []
                data['Upper'] = []
                if has_rc:
                    data['RC'] = []
            elif obj.type() is Constraint:
                data['Body'] = []
                data['Lower'] = []
                data['Upper'] = []
                if has_dual:
                    data['Dual'] = []
            elif obj.type() is Param:
                data['Value'] = []

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
                        if has_rc:
                            if dobj in model.rc:
                                data['RC'].append(model.rc[dobj])
                            else:
                                data['RC'].append(None)
                    else:
                        data['Lower'].append(value(dobj.lower))
                        data['Body'].append(value(dobj.body))
                        data['Upper'].append(value(dobj.upper))
                        if has_dual:
                            if dobj in model.dual:
                                data['Dual'].append(model.dual[dobj])
                            else:
                                data['Dual'].append(None)

            elif obj.type() is Param:
                for ind, val in obj.iteritems():
                    for i in range(len(indices)):
                        data[indices[i]].append(ind if obj.dim() < 2 else
                                                ind[i])
                    data['Value'].append(val)

            df = pd.DataFrame(data)
            df = df.set_index(indices)
            if obj.type() is Var:
                if has_rc:
                    df = df[['Lower', 'Value', 'Upper', 'RC']]
                else:
                    df = df[['Lower', 'Value', 'Upper']]
            elif obj.type() is Constraint:
                if has_dual:
                    df = df[['Lower', 'Body', 'Upper', 'Dual']]
                else:
                    df = df[['Lower', 'Body', 'Upper']]

            obj_frames.append((obj, df))

        elif obj.type() in (Set, RangeSet) and obj.dim() == 0:
            data['Values'] = sorted(obj.data())
            df = pd.DataFrame(data)
            obj_frames.append((obj, df))

    # Set up dataframes for Scalar Sheet
    scalar_frames = []

    param_data = {'Param' : [obj.name for obj in scalars['Param']],
                  'Value' : [obj.value for obj in scalars['Param']],
                  'Doc'   : [obj.doc for obj in scalars['Param']]}
    scalar_frames.append(pd.DataFrame(param_data)
        .set_index('Param')[['Value','Doc']])

    var_data = {'Var'   : [obj.name for obj in scalars['Var']],
                'Lower' : [obj.lb for obj in scalars['Var']],
                'Value' : [obj.value for obj in scalars['Var']],
                'Upper' : [obj.ub for obj in scalars['Var']],
                'Doc'   : [obj.doc for obj in scalars['Var']]}
    if has_rc:
        var_data['RC'] = []
        for obj in scalars['Var']:
            var_data['RC'].append(model.rc[obj] if obj in model.rc else None)
        scalar_frames.append(pd.DataFrame(var_data)
            .set_index('Var')[['Lower','Value','Upper','RC','Doc']])
    else:
        scalar_frames.append(pd.DataFrame(var_data)
            .set_index('Var')[['Lower','Value','Upper','Doc']])

    con_data = {'Constraint' : [obj.name for obj in scalars['Constraint']],
                'Lower' : [value(obj.lower) for obj in scalars['Constraint']],
                'Body'  : [value(obj.body) for obj in scalars['Constraint']],
                'Upper' : [value(obj.upper) for obj in scalars['Constraint']],
                'Doc'   : [obj.doc for obj in scalars['Constraint']]}
    if has_dual:
        con_data['Dual'] = []
        for obj in scalars['Constraint']:
            con_data['Dual'].append(model.dual[obj] if obj in model.dual else
                                    None)
        scalar_frames.append(pd.DataFrame(con_data)
            .set_index('Constraint')[['Lower','Body','Upper','Dual','Doc']])
    else:
        scalar_frames.append(pd.DataFrame(con_data)
            .set_index('Constraint')[['Lower','Body','Upper','Doc']])

    objective_data = {'Objective':[obj.name for obj in scalars['Objective']],
                      'Value':[value(obj.expr) for obj in scalars['Objective']],
                      'Doc':[obj.doc for obj in scalars['Objective']]}
    scalar_frames.append(pd.DataFrame(objective_data)
        .set_index('Objective')[['Value','Doc']])

    if filename is None:
        filename = model.name + '.xlsx'

    with pd.ExcelWriter(filename, engine=engine) as writer:
        # Table of Contents
        TOC.to_excel(writer, sheet_name='TOC')

        # Scalar Sheet
        scalar_startrow = 1
        for df in scalar_frames:
            if df.empty:
                continue
            df.to_excel(writer, sheet_name='Scalar', startrow=scalar_startrow,
                        merge_cells=False)
            scalar_startrow += len(df.index) + 2

        if writer.engine == 'xlsxwriter':
            # Create link for each TOC entry to their respective sheet
            ws = writer.book.get_worksheet_by_name('TOC')
            link_format = writer.book.add_format({'color'     : 'blue',
                                                  'underline' : 1,
                                                  'bold'      : 1,
                                                  'align'     : 'center'})
            for i in range(len(objects)):
                obj = objects[i]
                if obj.type().__name__ in scalars and obj.dim() == 0:
                    sheet = 'Scalar'
                else:
                    sheet = obj.name
                ws.write_url(row=i + 1, col=0, url='internal:%s!A1' % sheet,
                             cell_format=link_format, string=obj.name)

            # Place 'TOC' link on the Scalar sheet
            if scalar_startrow > 1:
                scalar_sheet = writer.book.get_worksheet_by_name('Scalar')
                scalar_sheet.write_url('A1', 'internal:TOC!A1', string='TOC')

        elif writer.engine[:8] == 'openpyxl':
            # Engine is actually 'openpyxl22', only check that it starts
            # with 'openpyxl' in case there might be other versions
            from openpyxl.styles import Font, Color, colors
            from copy import copy

            # Create link for each TOC entry to their respective sheet
            ws = writer.book.get_sheet_by_name('TOC')
            # Use theme=10 to get color to change when link is clicked
            link_color = Color(theme=10)
            name_font = Font(bold=True, underline='single', color=link_color)
            for i in range(len(objects)):
                obj = objects[i]
                if obj.type().__name__ in scalars and obj.dim() == 0:
                    sheet = 'Scalar'
                else:
                    sheet = obj.name
                ws.cell(row=i + 2, column=1).hyperlink = '#%s!A1' % sheet
                ws.cell(row=i + 2, column=1).font = name_font

            # Place 'TOC' link on the Scalar sheet
            toc_font = copy(name_font)
            toc_font.bold = False
            if scalar_startrow > 1:
                scalar_sheet = writer.book.get_sheet_by_name('Scalar')
                scalar_sheet['A1'].value = 'TOC'
                scalar_sheet['A1'].hyperlink = '#TOC!A1'
                scalar_sheet['A1'].font = toc_font

        for obj, df in obj_frames:
            # When cells are merged, things like the autofilter do not
            # work. For example the autofilter only recognizes the first row
            # of the merged block when filtering.
            df.to_excel(writer, sheet_name=obj.name, startrow=2,
                        merge_cells=False)

            cols = df.index.nlevels + len(df.columns)

            if writer.engine == 'xlsxwriter':
                ws = writer.book.get_worksheet_by_name(obj.name)

                ws.write_url('A1', 'internal:TOC!A1', string='TOC')
                ws.write('A2', obj.name)
                ws.write('B2', obj.type().__name__)
                ws.write('C2', obj.doc)

                ws.autofilter(2, 0, 2, cols - 1)

            elif writer.engine[:8] == 'openpyxl':
                from openpyxl.utils import get_column_letter

                ws = writer.book.get_sheet_by_name(obj.name)

                ws['A1'].value = 'TOC'
                ws['A1'].hyperlink = '#TOC!A1'
                ws['A1'].font = toc_font

                ws['A2'] = obj.name
                ws['B2'] = obj.type().__name__
                ws['C2'] = obj.doc

                ws.auto_filter.ref = "A3:%s3" % get_column_letter(cols)




"""
TODO:
Indexed Objective?
Expressions
Blocks
Indexed Sets, RangeSets
"""
