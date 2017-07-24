import pandas as pd

from pyomo.core.base import (Constraint, Param, Set, RangeSet, Var, Objective,
                             Block, Suffix, Expression, value, SortComponents)


scalar_types = set(['Param', 'Var', 'Expression', 'Constraint', 'Objective'])


def write_dataframes(model):
    valid_ctypes = set([Var, Constraint, Param, Set, RangeSet,
                        Expression, Objective, Block])
    objects = []
    scalars = dict()
    for typ in scalar_types:
        scalars[typ] = []
    for obj in model.component_objects(active=True, descend_into=False,
                                       sort=SortComponents.alphaOrder):
        if obj.type() in valid_ctypes:
            objects.append(obj)
        if obj.type().__name__ in scalars and obj.dim() == 0:
            scalars[obj.type().__name__].append(obj)


    # Table of Contents
    toc_data = {'Name'  : [obj.name for obj in objects],
                'Type'  : [obj.type().__name__ for obj in objects],
                'Dim'   : [obj.dim() for obj in objects],
                'Count' : [len(obj) for obj in objects],
                'Doc'   : [obj.doc for obj in objects]}
    TOC = (pd.DataFrame(toc_data)
        .set_index('Name')[['Type', 'Dim', 'Count', 'Doc']])
    # Store model/block name in TOC dataframe
    # Used with default file naming and with blocks
    # Replace brackets, which are illegal sheet name characters
    TOC.name = model.name.replace('[', '(').replace(']', ')')


    has_dual = has_rc = False
    for suf in model.component_data_objects(Suffix, active=True):
        if (suf.name == 'dual' and suf.import_enabled()):
            has_dual = True
        elif (suf.name == 'rc' and suf.import_enabled()):
            has_rc = True


    # Set up dataframes for Scalar Sheet
    scalar_frames = []

    # Scalar Params
    param_data = {'Param' : [obj.name for obj in scalars['Param']],
                  'Value' : [obj.value for obj in scalars['Param']],
                  'Doc'   : [obj.doc for obj in scalars['Param']]}
    scalar_frames.append(pd.DataFrame(param_data)
        .set_index('Param')[['Value','Doc']])

    # Scalar Vars
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

    # Scalar Expressions
    expr_data = {'Expression' : [obj.name for obj in scalars['Expression']],
                 'Value'      : [value(obj) for obj in scalars['Expression']],
                 'Doc'        : [obj.doc for obj in scalars['Expression']]}
    scalar_frames.append(pd.DataFrame(expr_data)
        .set_index('Expression')[['Value','Doc']])

    # Scalar Constraints
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

    # Scalar Objectives
    objctv_data = {'Objective' : [obj.name for obj in scalars['Objective']],
                   'Value'     : [value(obj) for obj in scalars['Objective']],
                   'Doc'       : [obj.doc for obj in scalars['Objective']]}
    scalar_frames.append(pd.DataFrame(objctv_data)
        .set_index('Objective')[['Value','Doc']])


    # Set up dataframes for each object
    obj_frames = []

    for obj in objects:
        if obj.type() is Block and obj.dim() == 0:
            obj_frames.append(write_dataframes(obj))
            continue

        data = dict()

        if obj.type() is Block and obj.dim() > 0:
            if obj.dim() == 1:
                indices = [obj.index_set().name]
            else:
                indices = [ind.name for ind in obj._implicit_subsets]
            for ind in indices:
                data[ind] = []

            data['Block'] = []

            for ind, dobj in sorted(obj.iteritems(), key=lambda item: item[0]):
                for i in range(len(indices)):
                    data[indices[i]].append(ind if obj.dim() == 1 else ind[i])
                data['Block'].append(dobj.name)

            # Indexed blocks are stored as a list containing all of their
            # constituent blocks, in order to keep them grouped together for
            # sheet linking purposes. Use string to denote indexed block.
            df = pd.DataFrame(data).set_index(indices)
            indx_blk = [obj, df]

            for dobj in sorted(obj.itervalues(), key=lambda val: val.index()):
                indx_blk.append(write_dataframes(dobj))

            obj_frames.append(indx_blk)

        elif (obj.type() in (Var, Constraint, Param, Objective, Expression) and
            obj.dim() > 0):
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
            elif obj.type() in (Param, Objective):
                data['Value'] = []

            if obj.type() in (Var, Constraint):
                for ind, dobj in sorted(obj.iteritems(),
                                        key=lambda item: item[0]):
                    for i in range(len(indices)):
                        data[indices[i]].append(ind if obj.dim() == 1 else
                                                ind[i])
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

            elif obj.type() in (Param, Objective, Expression):
                for ind, val in sorted(obj.iteritems(),
                                       key=lambda item: item[0]):
                    for i in range(len(indices)):
                        data[indices[i]].append(ind if obj.dim() == 1 else
                                                ind[i])
                    data['Value'].append(val if obj.type() is Param else
                                         value(val))

            df = pd.DataFrame(data).set_index(indices)

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

    return [TOC, scalar_frames, obj_frames]


def write_excel(all_frames, filename=None, engine=None,
                writer=None, parent=None, parent_row=None):
    TOC, scalar_frames, obj_frames = all_frames

    block = TOC.name

    if writer is None:
        if filename is None:
            # Default filename is model name, stored in top-level TOC.name
            filename = TOC.name + '.xlsx'
        writer = pd.ExcelWriter(filename, engine=engine)


    # Table of Contents
    toc_name = 'TOC' if parent is None else block
    TOC.to_excel(writer, sheet_name=toc_name,
                 startrow=0 if parent is None else 1)

    # Scalar Sheet
    scalar_startrow = 1
    scalar_name = 'Scalar' if parent is None else block + '.Scalar'
    for df in scalar_frames:
        if df.empty:
            continue
        df.to_excel(writer, sheet_name=scalar_name, startrow=scalar_startrow,
                    merge_cells=False)
        scalar_startrow += len(df.index) + 2


    if writer.engine == 'xlsxwriter':
        # Create link for each TOC entry to their respective sheet
        toc = writer.book.get_worksheet_by_name(toc_name)
        link_format = writer.book.add_format({'color'     : 'blue',
                                              'underline' : 1,
                                              'bold'      : 1,
                                              'align'     : 'center'})
        for i in range(len(TOC.index)):
            if TOC['Type'][i] in scalar_types and TOC['Dim'][i] == 0:
                sheet = scalar_name
            else:
                # Use TOC['Name'] -> TOC.index
                sheet = TOC.index[i].replace('[', '(').replace(']', ')')
            name = TOC.index[i]
            # If there is a parent TOC, move the row down 1 for each entry
            i += int(bool(parent))
            toc.write_url(row=i + 1, col=0, url="internal:'%s'!A1" % sheet,
                          cell_format=link_format, string=name)

        # Place 'TOC' link on the Scalar sheet
        if scalar_startrow > 1:
            scalar_sheet = writer.book.get_worksheet_by_name(scalar_name)
            scalar_sheet.write_url('A1', "internal:'%s'!A1" % toc_name,
                                   string=toc_name)

        # Place TOC link on child TOC page
        if parent is not None:
            toc.write_url('A1', "internal:'%s'!A%s" % (parent, parent_row),
                          string=parent)

    elif writer.engine[:8] == 'openpyxl':
        # Engine is actually 'openpyxl22', only check that it starts
        # with 'openpyxl' in case there might be other versions
        from openpyxl.styles import Font, Color, colors
        from openpyxl.utils import get_column_letter
        from copy import copy

        # Create link for each TOC entry to their respective sheet
        toc = writer.book.get_sheet_by_name(toc_name)
        # Use theme=10 to get color to change when link is clicked
        link_color = Color(theme=10)
        name_font = Font(bold=True, underline='single', color=link_color)
        for i in range(len(TOC.index)):
            if TOC['Type'][i] in scalar_types and TOC['Dim'][i] == 0:
                sheet = scalar_name
            else:
                sheet = TOC.index[i].replace('[', '(').replace(']', ')')
            # If there is a parent TOC, move the row down 1 for each entry
            i += int(bool(parent))
            toc.cell(row=i + 2, column=1).hyperlink = "#'%s'!A1" % sheet
            toc.cell(row=i + 2, column=1).font = name_font

        # Place 'TOC' link on the Scalar sheet
        toc_font = copy(name_font)
        toc_font.bold = False
        if scalar_startrow > 1:
            scalar_sheet = writer.book.get_sheet_by_name(scalar_name)
            scalar_sheet['A1'].value = toc_name
            scalar_sheet['A1'].hyperlink = "#'%s'!A1" % toc_name
            scalar_sheet['A1'].font = toc_font

        # Place TOC link on child TOC page
        if parent is not None:
            toc['A1'].value = parent
            toc['A1'].hyperlink = "#'%s'!A%s" % (parent, parent_row)
            toc['A1'].font = toc_font


    for item in obj_frames:

        if type(item) is list and type(item[0]) is pd.DataFrame:
            # This is a singleton block's list of dataframes
            # The first item in the list if the TOC dataframe
            child_toc = item[0]
            # Pass row of this block's entry in this TOC so that the child
            # block's TOC can link back to its row in the parent TOC
            toc_row = TOC.index.get_loc(child_toc.name) + 2 + int(bool(parent))
            write_excel(item, writer=writer, parent_row=toc_row,
                        parent='TOC' if parent is None else block)
            # This is the only case where the extra info at
            # the top of the sheet is not needed.
            continue

        if type(item) is list:
            # This is an indexed block, whose first item is the object
            # List items are: object, dataframe listing all subblocks,
            # and then each subblock in the form of a list.
            obj, df = item[0], item[1]
            obj_name = obj.name.replace('[', '(').replace(']', ')')
            df.to_excel(writer, sheet_name=obj_name, startrow=2,
                             merge_cells=False)

            col = df.index.nlevels
            i = 0
            for blk in item[2:]:
                child_toc = blk[0]
                name = child_toc.name
                parent_row = i + 4 + int(bool(parent))
                write_excel(blk, writer=writer, parent_row=parent_row,
                            parent=obj_name)

                # Create link for each subblock
                if writer.engine == 'xlsxwriter':
                    sheet = writer.book.get_worksheet_by_name(obj_name)
                    sheet.write_url(row=i + 3, col=col, string=name,
                                    url="internal:'%s'!A1" % name)

                elif writer.engine[:8] == 'openpyxl':
                    sheet = writer.book.get_sheet_by_name(obj_name)
                    cell = sheet.cell(row=i + 4, column=col + 1)
                    cell.hyperlink = "#'%s'!A1" % name
                    cell.font = toc_font
                i += 1

        else:
            # type is tuple -> regular object and dataframe
            obj, df = item
            obj_name = obj.name.replace('[', '(').replace(']', ')')

            # When cells are merged, things like the autofilter do not
            # work. For example the autofilter only recognizes the first row
            # of the merged block when filtering.
            df.to_excel(writer, sheet_name=obj_name, startrow=2,
                        merge_cells=False)

        cols = df.index.nlevels + len(df.columns)
        toc_row = TOC.index.get_loc(obj.name) + 2 + int(bool(parent))

        if writer.engine == 'xlsxwriter':
            ws = writer.book.get_worksheet_by_name(obj_name)

            ws.write_url('A1', "internal:'%s'!A%s" % (toc_name, toc_row),
                         string=toc_name)
            ws.write('A2', obj.name)
            ws.write('B2', obj.type().__name__)
            ws.write('C2', obj.doc)

            ws.autofilter(2, 0, 2, cols - 1)

        elif writer.engine[:8] == 'openpyxl':
            ws = writer.book.get_sheet_by_name(obj_name)

            toc_row = TOC.index.get_loc(obj.name) + 2 + int(bool(parent))

            ws['A1'].value = toc_name
            ws['A1'].hyperlink = "#'%s'!A%s" % (toc_name, toc_row)
            ws['A1'].font = toc_font

            ws['A2'] = obj.name
            ws['B2'] = obj.type().__name__
            ws['C2'] = obj.doc

            # Why is this line causing a bug? It happens when the name of the
            # sheet has a parenthesis in it...
            # The code will still work, but Excel needs to repair the file first
            ws.auto_filter.ref = "A3:%s3" % get_column_letter(cols)


    if parent is None:
        writer.save()

    return filename




"""
TODO:
Indexed Sets, RangeSets
Disjunct
active column
"""
