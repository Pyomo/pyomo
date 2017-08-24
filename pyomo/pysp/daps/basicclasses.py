# Started by DLW, December 2016
"""The prime directive: write the code and its comments as if the code
were intended for a real user. Researchers can script from "outside."
If researchers need the code to be factored differently, that is OK,
but do not violate the prime directive by putting in code that only
supports research other than in research scripts!!!!

These are mostly base classes (some are glorified functions, but that
can remember some things) and functions for out-of-the-box use.  I am
trying to keep a few use cases in mind: time series and non-time
series.  I think we need to be multi-stage and multi-variate from the
start.  However, it might be reasonable to start by treating two-stage
as a special case. I am trying to put in a little "sample code" for
both abstract and concrete models.

The *big* thing at this point is that raw scenario data needs to be
find its way into a dictionary (perhaps of dictionaries). For concrete
models, that will generally be whatever it needs to be and the
scenario creation callback will deal with it when it gets it. For
abstract models (and some concrete models), the outer index is a Param
name and the inner indexes (if present) are full param index strings.

"""

import os
import sys
import shutil
import copy
import json
import networkx
import pyomo.pysp.scenariotree.tree_structure_model as ptm
from collections import OrderedDict

#==============================================================================
def daps_data_dir():
    """ Somehow return the path to the node, tree and scenario data.
    This is intended for optional use by programmers of daps and optiona
    use by programmers of PySP callbacks.
    """
    if "DAPS_DATA_DIR" not in os.environ:
        print ("Warning: DAPS_DATA_DIR enviornment variable not defined.")
        print ("Defulting to 'stoch_dir'")
        return 'stoch_dir'
    else:
        return os.environ["DAPS_DATA_DIR"]
    
#==============================================================================
class Raw_Node_Data:
    """ The raw data for one node of the scenario tree.
    There is really nothing stopping this thing from having a full
    scenario, but it is only responsible for being correct up to
    the stage of its node. 
    (When a scenario is assembled it will be done in stage order.)
    NOTE: one probably wants the node name to be "conditional" because
    the scenario names are assembled by concatenation of node names.
    Primary public data elements:
    name: string
    parentname: string
    prob: float in [0,1]
    valdict: a dictionary (perhaps of dictinaries) where the ultimate
             values define the data for the scenario tree node
    """

    def __init__(self, filespec = None,
                 dictin = None,
                 name = None,
                 parentname = None,
                 prob = None):

        #==========
        def from_dict(self, dictin, name, parentname, prob):
            """
            A simple constructor body that takes the data from its arguments.
            """
            if prob is None or prob < 0 or prob > 1.0:
                raise RuntimeError('Internal error: Raw_Node_Data; ' +
                                   "bad prob="+str(prob))
            self.valdict = OrderedDict()
            for pname in dictin:
                if isinstance(dictin[pname], dict):
                    if pname not in self.valdict:
                        self.valdict[pname] = {}
                    for pindex in dictin[pname]:
                        self.valdict[pname][pindex] = dictin[pname][pindex]
                else:
                    self.valdict[pname] =  dictin[pname][pindex]
            self.name = name
            self.parent_name = parentname
            self.prob = prob
            
            
        #==========
        def from_file(self, filespec):
            """
            The filespec argument can be a full path, 
            but the filename part of the string must be of a special form.
            Not much error checking here; the caller should do that.
            """
            def pcheck(parts, token, partloc):
                """Gag and die if the token is not parts[partloc]"""
                if len(parts) <= partloc or parts[partloc] != token:
                    raise RuntimeError('Internal error in Raw_Node_Data: ' + \
                                   'misplaced '+token+' in filespec='+filespec)

            path, filename = os.path.split(filespec)
            parts = filename.split('-')
            pcheck(parts, 'NODE', 0)
            pcheck(parts, 'PARENT', 2)
            pcheck(parts, 'PROB', 4)

            self.name = parts[1]
            self.parent_name = parts[3]
            # the prob can cause trouble with period or dash so maxsplit is used
            eparts = filespec.rsplit(sep='.', maxsplit=1)
            if len(eparts) != 2:
                raise RuntimeError('Internal error in Raw_Node_Data: ' + \
                                   'missing extension in filespec='+filespec)
            self.extension = eparts[1]
            # eparts[0] should end with the prob
            self.prob = float(eparts[0][eparts[0].find('PROB')+5:])
            if self.prob < 0 or self.prob > 1:
                raise RuntimeError('Internal error in Raw_Node_Data: ' + \
                                   'bad prob in filespec='+filespec)

            if self.extension == 'json':
                with open(filespec) as infile:
                    self.valdict = json.load(infile)
            elif self.extension == 'csv':
                # use pandas
                print ("csv files not yet supported....")
                sys.exit(0)
            elif self.extension == 'ssv':
                # crude little dlw format: semi-colon seperate values
                self.valdict = {}
                with open(filespec) as infile:
                    for iline in infile:
                        line = iline.strip()
                        if line == "":
                            continue
                        if line[0] == "#":
                            continue
                        if line.find('#') != -1:
                            raise RuntimeError("Comment beyond first char "\
                                               + "in ssv line=" + line
                                               + "\n in filespec=" + filespec)

                        parts = line.split(';')
                        for i in range(len(parts)):
                            parts[i] = parts[i].strip()
                        if len(parts) == 2:
                            self.valdict[parts[0]] = parts[1]
                        elif len(parts) == 3:
                            if parts[0] not in self.valdict:
                                self.valdict[parts[0]] = {}
                            self.valdict[parts[0]][parts[1]] = parts[2]
                        else:
                            raise RuntimeError("Bad ssv line=" + line
                                                +"\n in filespec="+filespec)
            else:
                raise RuntimeError('Internal error in Raw_Node_Data: ' + \
                                   'unknown ext='+self.extension + \
                                   ' in filespec='+filespec)

        #==========
        # The body of __init__
        if filespec is not None and dictin is not None:
            raise RuntimeError('Internal error: Raw_Node_Data; ' +
                               "both constructor options given")
        if filespec is not None:
            if name is not None \
                or parentname is not None \
                or prob is not None:
                raise RuntimeError('Internal error: Raw_Node_Data; ' +
                               "filename must give tree info not args.")
            from_file(self,filespec)
            
        elif dictin is not None:
            from_dict(self,dictin, name, parentname, prob)
            
        else:
            raise RuntimeError('Internal error: Raw_Node_Data; ' +
                               "neither constructor option given")

    #==================
    def write_json(self):
        """ construct the file name and write a json file """
        pass

    #==================
    def write_csv(self):
        """ construct the file name and write a csv file """
        pass
    
#==============================================================================
class Raw_Scenario_Data:
    """ This can write a raw scenario file and/or feed
    data to PySP classes. It could have a variety
    of constructor-like-things. It is envisioned to be the main
    statistical workhorse (or it will direct the work
    or will read from a file if something else does the work).
    Basic idea: use a dictionary with indexes that can be
    used to substitute into a template when making
    a PySP scenario, or used for communication with
    other software. The dictionary entries may be
    dictionaries; in that case the outer
    index gives the Param name and the inner indexes
    are full indexes for the data for abstract models 
    (interpreted by concrete models however they want to.)
    Note: as of Dec 2016, there is no vision for a *raw* template,
    but one could be supported (a node zero might be better).
    The main data elements are named in __init__
    name: string
    prob: float in [0,1]
    nodelist: stage ordered list of RawNodeData objects
    valdict: a dictionary (perhaps of dictinaries) where the ultimate
             values define the data for the scenario
    """
    def __init__(self, node_data_list):
        """ node_data_list is a list of Raw_Node_Data classes, which
        is assumed to be in node order.
        IMPORTANT: To maximize flexibility, Raw_Node_Data objects can
        have more data than is needed for the node, but they must
        be correct up to the node as well as complete and correct at the node. 
        That way, this code can/must just keep 
        overwriting or adding without paying attention to which it is doing.
        """
        self.valdict = {} # may be values or dictionaries with values
        self.name = "Scenario"
        self.prob = 1
        self.nodelist = node_data_list

        # deal with integer indexes mainly for error checking
        num_nodes = len(node_data_list)
        if num_nodes == 0:
            raise RuntimeError('Raw_Scenario_data: empty Node_Data_List')
        
        for i in range(num_nodes):
            nd = node_data_list[i]
            self.name += "_"+nd.name
            self.prob = self.prob * nd.prob  # node probs are conditional
            if self.prob > 1:
                print ("**NOTE: node probabilities should be conditional.")
                print ("Cryptic note: name so far="+self.name)
                raise RuntimeError('Raw_Scenario_data: prob exceeds 1')
            # first, do some error checking, which is why I need i
            if i == 0:
                cname = nd.name
                if nd.parent_name != 'NONE' and nd.parent_name != 'ROOT':
                    raise RuntimeError('Internal: Raw_Scenario_Data: ' + \
                                       'first  parent_name!=NONE or ROOT')
                if nd.parent_name == 'NONE' and nd.name != 'ROOT':
                    raise RuntimeError('Internal: Raw_Scenario_Data: ' + \
                                       'if parent is NONE must be ROOT')
            else:
                if nd.parent_name != cname:
                    print ("Cryptic note: name so far="+self.name)
                    raise RuntimeError('Internal: Raw_Scenario_Data: ' + \
                                       'wrong parent for '+nd.name)
                cname = nd.name
            # now overwite or add data
            for pname in nd.valdict:
                if isinstance(nd.valdict[pname], dict):
                    if pname not in self.valdict:
                        self.valdict[pname] = {}
                    for pindex in nd.valdict[pname]:
                        self.valdict[pname][pindex] = nd.valdict[pname][pindex]
                else:
                    self.valdict[pname] = nd.valdict[pname]

        
#==============================================================================
class PySP_Scenario_Template:
    """ Has the basics of what is needed for a scenario
    but portions will be filled in as data is supplied.
    My vision is that this could be populated using
    the deterministic data file 
    maybe with a token to indicate where to start and
    stop writing data from the raw data (for AMPL files).
    For concrete models, maybe it is a JSON, panda or csv file
    or maybe you don't need a template for concrete models.
    """
    def __init__(self):
        # The form of the template data depends on the application.
        # As of Dec 2016, there are only two kinds: AMPL Text lines and
        # dictionaries.
        self.templatedata = None

    #=============================
    def read_AMPL_template_with_tokens(self, infilespec=None, start_token=None, end_token=None):
        """ Read an ampl format file with tokens to indicate scenario data start and stop
        assumes there is just one start and one stop and it is at the start of the line
        inputs:
            infilename: the name of the template file
            start_token: string that starts the lines to be replaced (e.g. '#STARTSCEN')
            stop_token: string that starts the lines to be replaced (e.g. '#ENDSCEN')
        output:
            puts the lines of the file in a list and remembers the start and stop indexes
            (NOTE: in theory one could have an AMPL template without tokens, but that
             would need to use pyomo to parse it really, so then you could replace
             data using the raw data values)
        """
        self.start_line = None
        self.end_line = None
        self.templatedata = []

        with open(infilespec,'r') as infile:
            for line in infile:
                if start_token is not None and line.find(start_token) == 0:
                    self.start_line = len(self.templatedata)
                elif end_token is not None and line.find(end_token) == 0:
                    self.end_line = len(self.templatedata)
                self.templatedata.append(line)
                
    #===========================
    def read_JSON_template(self, infilename=None):
        """ Read a json template, which is presumed to be a dictionary,
        perhaps of dictionaries. This should correspond the raw scenario data
        dictionary, at least in places, so that the scenario data can overwrite
        the template data as needed. This is just an example for
        concrete modelers. Pandas could also be used in similar fashion.
        """
        with open(filespec) as infile:
            self.templatedata = json.load(infile)

#==============================================================================
class PySP_Scenario:
    """ Has everything needed to be able to write a PySP scenario file and can do so.
    A constructor takes as input a template, raw data and a probability.
    The main idea here is that the raw data can come in as a dictionary of
    dictionaries and go out however it needs to go out.
    """
    def __init__(self, raw, template=None):
        """Inputs:
        template: a PySP_Scenario_Template object.
        raw: a Raw_Scenario_Data object, which it remembers.
        Processing:
        Combine the template and raw data to produce a full
        scenario appropriate for PySP.
        Data structure can vary:
          - a dictionary of dictionaries to be interpreted later.
          - or a list with AMPL format strings ready to write to a file
        output:
        can write an AMPL format input file or a csv file, or JSON, etc.
        """
        self.name = raw.name
        self.prob = raw.prob
        self.raw = raw  # we can get at the tree
        # wht the scenario data looks like depends on the situation
        self.scenariodata = None
        # Figure out what we are template and/or raw data we are dealing with
        # and deal with them.
        # As of Dec 2016, there are only AMPL text scenarios and
        # dictionary scenarios.
        
        #==========
        def make_AMPL_scenario(self, raw, template):
            """ Replace the appropriate lines in the template with the scenario lines.
                Note that AMPL template data are assumed to be complete lines.
                As of Dec 2016 there is no error checking.
            """
            self.scenariodata = []
            # start with the template
            if template is not None:
                if hasattr(template, 'start_line') \
                   and template.start_line is not None:
                    for i in range(template.start_line):
                        self.scenariodata.append(template.templatedata[i])
                else:
                    for line in template.templatedata:
                        self.scenariodata.append(line)
                    
            # Put in the scenario specific lines, which are in a dict.
            # Do not try to be efficient.
            for pname in raw.valdict:
                if isinstance(raw.valdict[pname], dict):
                    # Dec 2016 this will totally NOT work in general... TBD fix it in general...
                    # If you had to live with this, then the tuples for indexes
                    # for indexes would have to come in with spaces and not commas.
                    # Feb 2017: might be fixed... not sure...
                    self.scenariodata.append("param: " + pname + " := ")
                    for pindex in raw.valdict[pname]:
                        """
                        self.scenariodata.append( \
                                            "param " + pname \
                                            + "[" + str(pindex) + "] := " \
                                            + str(raw.valdict[pname][pindex]) \
                                            + " ;\n")
                        """
                        self.scenariodata.append(str(pindex) + " " \
                                            + str(raw.valdict[pname][pindex]) \
                                            + " ")
                    self.scenariodata.append(" ;\n")
                else:
                    self.scenariodata.append( \
                                            "param " + pname + " := " \
                                            + str(raw.valdict[pname]) + ";\n")
            # back to the template
            # if it has one, it has both...
            if template is not None:
                if hasattr(template, 'start_line') \
                   and template.start_line is not None:
                    for i in range(template.end_line+1, len(template.templatedata)):
                        self.scenariodata.append(template.templatedata[i])

        #==========
        def make_dict_scenario(self, raw, template = None):
            """ Replace the appropriate data in the template with the raw scenario data.
                Or add to it!!
                Note: I am using index variable names like pname, but the interpretation
                is really left to the model that gets this.
            """
            if template is not None:
                self.scenariodata = copy.deepcopy(template)
            else:
                self.scenariodata = {}
            for pname in raw.valdict:
                if isinstance(raw.valdict[pname], dict):
                    if pname not in self.scenariodata:
                        self.scenariodata[pname] = {}
                    for pindex in raw.valdict[pname]:
                        self.scenariodata[pname][pindex] = raw.valdict[pname][pindex]
                else:
                    self.scenariodata[pname] = raw.valdict[pname]

        if isinstance(template, dict):
            make_dict_scenario(self, raw=raw, template=template)
        else: # assume AMPL if not dict as of Dec 2016
            make_AMPL_scenario(self, raw=raw, template=template)
            
#==============================================================================
class PySP_Tree_Template:
    """ This thing knows about the stages.
    It does not know how many scenarios there are, or how
    they are organized into a tree.
    As of Dec 2016, I am just assuming this comes from an AMPL Data file
    and the file has everyting about the Vars and Costs that is needed.
    Cryptic note: as of dec 2016 we are not going to check for "Scenarios"
    so that one could use a valid ScenarioStructure.dat file and
    I think when we put the new Scenarios at the bottom, it will be OK.
    
    Feb 2017: Added capability for a json template that simply a 
        dictionary. As is the case for the AMPL template, this dictionary
        is assumed to be basically OK when constructing this object and it is up
        to code that uses it to complain about it. We will check 
        to make sure it has the right high level keys (e.g. Stages).
    """
    def __init__(self):
        self.templatedata = None
        self.SourceFileSpec = None
        self.StageNames = None

    #==========
    def read_json_template(self, filespec):
        """
        Read a json format scenario tree template file from filespec.
        The template is presumed to be a dictionary (perhaps of dictionaries)
        with stage_names, Vars, and costs.
        """
        def checkforkey(key, td):
            if key not in td:
                raise RuntimeError("Expected '"+key+"' in file=" + \
                                   str(self.SourceFileSpec)) 

        self.SourceFileSpec = filespec

        with open(filespec) as infile:
            self.templatedata = json.load(infile)

        checkforkey("Stages", self.templatedata)
        checkforkey("StageVariables", self.templatedata)
        checkforkey("StageCost", self.templatedata)

        self.StageNames = self.templatedata["Stages"]
        if type(self.StageNames) is not list:
            raise RuntimeError("Expected Stages to be a list in file=" \
                               + str(self.filespec)) 
            
    #==========
    def read_AMPL_template(self, filespec):
        """
        Read an AMPL format scenario tree template file from filespec.
        """
        self.SourceFileSpec = filespec
        self.templatedata = []
        # read the whole file, then find the stages list.
        with open(filespec,'r') as infile:
            for line in infile:
                self.templatedata.append(line)

        # this could be done in the same loop, but why?
        for i in range(len(self.templatedata)):        
            line = self.templatedata[i] # typing aide
            loc = line.find("set Stages")
            cloc = line.find("#")
            if loc > -1 and (cloc == -1 or cloc < loc):
                # process the stage names
                # first, assemble the string
                stagestr = line
                sloc = line.find(";")
                j = i # note: we have the line indexed by i
                while sloc == -1 or sloc < cloc:
                    j += 1
                    try:
                        line = self.templatedata[j]
                    except:
                        raise RuntimeError('Error seeking semi-colon '
                                           + ' after "set Stages :="' \
                                           + 'in filespec='+filespec)
                    stagestr += line
                    cloc = line.find("#")
                    sloc = line.find(";")
                break
        self.StageNames = []
        try:
            stagestr = stagestr[stagestr.find('=')+1:stagestr.find(';')].strip()
            self.StageNames = stagestr.split(' ')
        except:
            raise RuntimeError('Error processing stage names in filespec=' + \
                               filespec)

#==============================================================================
class PySP_Tree:
    """ Has a list of scenarios and can write input files suitable for PySP.
    Instead of writing a ScenarioStructure.dat file, it might return a 
    a model object.
    One constructor will take a list of scenarios and the tree
    template (which will have the number of stages).
    """    
    def __init__(self, template, scenarios, raw_nodes):
        """inputs: template is a PySP template class object
                   scenarios is a list of PySP scenario class objects
        """
        epsilon = 1e-6   # for testing probs
        self.template = template
        self.scenarios = scenarios
        # check scenario names for uniqueness and check total prob
        totalprob = 0.0
        scenarionames = []
        for scen in scenarios:
            sname = scen.name
            if scen in scenarionames:
                raise RuntimeError('Internal: Duplicate Scenario Name='+sname)
            totalprob += scen.prob
        if totalprob > 1.0 + epsilon:
            raise RuntimeError('Internal: > 1.0, totalprob='+totalprob)

        """
        In this "default" version, we are going to extract the tree information
        from the scenarios. One can envision deriving classes where that
        information is passed in by whatever created the scenarios in the first
        place. For now, keep it simple.
        """
        """
        We need to guard against (or allow) "conditional" 
        raw node names, so we
        will form the node names by concatentation, which could get ugly.
        They are the dict indexes, by the way.
        Dec 2016: Let's assume that if there was a ROOT node given (uncommon)
        for any scenario, it was given for all.... (do I use this assumption?)
        """
        self.NodeNames = ['Node_ROOT']
        self.NodeStage = {'Node_ROOT':0}  # zero based
        self.NodeProb = {'Node_ROOT':1.0}
        self.NodeKids = {'Node_ROOT': []}
        self.LeafNode = {}  # indexed by scenario name
        self.ScenForLeaf = {} # index by leaf node name, give scenario name

        rootoffset = 0 
        for scen in scenarios:
            nodename = "Node_ROOT"  # *not* raw; PySP name
            for i in range(len(scen.raw.nodelist)):
                nd = scen.raw.nodelist[i]
                if nd.name == 'ROOT': # assumed uncommon, but allowed
                    rootoffset = 1
                    continue
                pname = nodename           # parent name
                nodename += "_"+nd.name
                if nodename not in self.NodeNames:
                    self.NodeNames.append(nodename)
                    self.NodeStage[nodename] = i + 1 - rootoffset
                    self.NodeProb[nodename] = nd.prob # conditional
                    self.NodeKids[nodename] = []
                if nodename in self.NodeKids[pname]:
                    raise RuntimeError('Duplicate created node name='+nodename)
                self.NodeKids[pname].append(nodename)
                self.LeafNode[scen.name] = nodename  # last assignment sticks..

        for scen in scenarios:
            self.ScenForLeaf[self.LeafNode[scen.name]] = scen.name # invert

                
    #====================
    def Write_ScenarioStructure_dat_file(self, filespec):
        """ write the AMPL format file as needed by PySP to filespec"""
        with open(filespec,"wt") as f:
            for line in self.template.templatedata:
                f.write(line)

            f.write("\nset Nodes :=\n")
            for ndname in self.NodeNames:
                f.write("   "+ndname+'\n')
            f.write(";\n")
            
            f.write("\nparam NodeStage :=\n")
            for ndname in self.NodeNames:
                stg = self.template.StageNames[self.NodeStage[ndname]]
                f.write("   "+ndname+' '+stg+'\n')
            f.write(";\n\n")

            for ndname in self.NodeNames:
                if len(self.NodeKids[ndname]) > 0:
                    f.write("set Children["+ndname+"] := \n")
                    for kid in self.NodeKids[ndname]:
                        f.write("   "+kid)
                    f.write(";\n")

            f.write("\nparam ConditionalProbability :=\n")
            for ndname in self.NodeNames:
                f.write("   "+ndname+' '+str(self.NodeProb[ndname])+'\n')
            f.write(";\n")

            f.write("\nset Scenarios :=\n")
            for scen in self.scenarios:
                f.write("   "+scen.name)
            f.write(";\n")

            f.write("\nparam ScenarioLeafNode :=\n")
            for scen in self.scenarios:
                f.write("   "+scen.name+' '+self.LeafNode[scen.name]+'\n')
            f.write(";\n")

    #====================
    def as_concrete_model(self):
        """ This returns the scenario tree as a concrete Pyomo model.
        ASSUMES that the scenario template is a dictionary.
        """
        if not isinstance(self.template.templatedata, dict):
            raise RuntimeError (
                "Attempt to construct a concrete tree model from non-dict")

        G = networkx.DiGraph()
        for nname in self.NodeNames:
            if len(self.NodeKids[nname]) > 0:
                G.add_node(nname)
            else:
                G.add_node(nname, scenario = self.ScenForLeaf[nname])
        for nname in self.NodeNames:
            for kidname in self.NodeKids[nname]:
                G.add_edge(nname, kidname, probability = self.NodeProb[kidname])

        tree_model = ptm.ScenarioTreeModelFromNetworkX(
            G,
            scenario_name_attribute = "scenario",
            stage_names = self.template.templatedata["Stages"])

        ttSV = self.template.templatedata["StageVariables"]
        for stagename in self.template.templatedata["Stages"]:
            stagenum = self.template.StageNames.index(stagename)+1
            stageobject = tree_model.Stages[stagenum]
            if stagename in ttSV:  # not all stages have Vars sometimes
                for varstring in ttSV[stagename]:
                    tree_model.StageVariables[stageobject].add(varstring)
            tree_model.StageCost[stageobject] \
                = self.template.templatedata["StageCost"][stagename]

        return tree_model

#=========================================================================
def do_2Stage_AMPL_dir(DirName,
                      TreeTemplateFileName, \
                      ScenTemplateFileName = None, \
                      start_token = None, end_token = None):
    """ DirName gives a directory, which is scanned for raw node data
    to process along with template files given by the repsective args.
    Assumes that there is no ROOT data file.
    If there is a scenario template, assumes it is AMPL format as of Dec 2016
    """
    treetemp = PySP_Tree_Template()
    spec = os.path.join(DirName, TreeTemplateFileName)
    treetemp.read_AMPL_template(spec)

    if ScenTemplateFileName is not None:
        spec = os.path.join(DirName, ScenTemplateFileName)
        scentemp = PySP_Scenario_Template()
        scentemp.read_AMPL_template_with_tokens(spec, start_token, end_token)
    else:
        scentemp = None

    rawnodelist = []
    for filename in os.listdir(DirName):
        if filename.startswith("NODE"):
            rawnodelist.append(Raw_Node_Data(os.path.join(DirName, filename)))

    # Since this is two stages and there is no ROOT file
    # we don't need to put together the raw nodes to make raw
    # scenarios; each node is a scenario.

    pyspscenlist = []
    for r in rawnodelist:
        innerlist = [] # going to a singleton
        innerlist.append(r)  # so silly to append
        rs = Raw_Scenario_Data(innerlist)
        PySPScen = PySP_Scenario(raw = rs, template = scentemp)
        pyspscenlist.append(PySPScen)
        fname = os.path.join(DirName, PySPScen.name + ".dat")
        with open(fname, "wt") as f:
            for line in PySPScen.scenariodata:
                f.write(line)

    # Note that there was no need to keep the raw scenario for
    # our purposes, but we need the raw nodes along with the PySP scenarios.
        
    tree = PySP_Tree(treetemp, pyspscenlist, rawnodelist)
    tree.Write_ScenarioStructure_dat_file(os.path.join(DirName,'ScenarioStructure.dat'))

#=========================================================================
def Tree_2Stage_json_dir(DirName,
                      TreeTemplateFileName, \
                      ScenTemplateFileName = None):
    """ DirName gives a directory, which is scanned for raw node data
    to process along with template files given by the repsective args.
    SIDE EFFECT: creates a scenario json file for each scenario named
    scenarioname.json
    where scenarioname is replaced by the scenario name.
    Assumes that there is no ROOT data file.
    As of Dec 2016, assume there is no template for the scenarios, either.
    Returns the scenario tree as a concrete model.
    """
    # the use of shutil.copy in the function is the trouble as of Dec 2016
    if ScenTemplateFileName is not None:
        raise RuntimeError('ScenTemplateFileName used where not supported.')
        
    treetemp = PySP_Tree_Template()
    spec = os.path.join(DirName, TreeTemplateFileName)
    treetemp.read_json_template(spec)
    
    rawnodelist = []  # so we can give them to the tree constructor
    rawfilelist = []  # so we can copy them
    for filename in os.listdir(DirName):
        if filename.startswith("NODE") and filename.endswith(".json"):
            rawnodelist.append(Raw_Node_Data(os.path.join(DirName, filename)))
            rawfilelist.append(filename)

    # Since this is two stages and there is no ROOT file
    # we don't need to put together the raw nodes to make raw
    # scenarios; each node is a scenario.

    # Not much is done during construction of PySP scenarios in this case,
    # but the tree constructor wants the list of objects, so we make it.
    pyspscenlist = []
    for i in range(len(rawnodelist)):
        innerlist = [] # going to be a singleton
        innerlist.append(rawnodelist[i])  # so silly to append
        rs = Raw_Scenario_Data(innerlist)
        PySPScen = PySP_Scenario(raw = rs, template = None)
        pyspscenlist.append(PySPScen)
        # Do the "side effect" file copy
        sourcename = os.path.join(DirName, rawfilelist[i])
        targetname = os.path.join(DirName, PySPScen.name + ".json")
        #print ('Debug: copy',sourcename, targetname)
        shutil.copy(sourcename, targetname)

    # Note that there was no need to keep the raw scenario for
    # our purposes, but we need the raw nodes along with the PySP scenarios.

    tree = PySP_Tree(treetemp, pyspscenlist, rawnodelist)
    tree_model = tree.as_concrete_model()

    return tree_model


