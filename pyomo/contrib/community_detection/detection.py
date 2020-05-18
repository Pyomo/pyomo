"""Community detection code"""

from contrib.community_detection import community_graph
import community


from pyutilib.misc import import_file
from pyomo.core import ConcreteModel
from pyomo.environ import *
'''
************************************************************************************************************************
Done this past week:
- combined detect_communities and create_community_graph
    - made a one line dictionary comprehension
- Add documentation to the new files
- Make stuff snake case not camel case
- Write the test cases for constraint and variable community detection functions
    - add files like alan.py as test cases
- Make one master community_graph function (structure similar to detect_communities)
- maybe make community_map count from 1 to n_communities instead of zero to n_communities - 1 (the documentation is 
incorrect right now)
- put test cases in separate functions
- try using a seed for the random number generator (in the test function to prevent heuristic-ness)

- take unweighted and weighted as an optional parameter
    - make U deepcopy of W right before adding in weights (OR MAYBE NOT BC INEFFICIENT!) ******* <- ASK!
    - answer: just change variable name
- made a one-line comprehension instead of triple for loop for constraint node graphs
- restructured code for community_graph as a whole; also resolved potential sorted tuple error
- interesting parallels that make it fun to see whether I can generalize things or not
- Add underscores for lower-level functions
    - why is pycharm crying when I did it below (_generate_model_graph)
- also test performance in another file (the huge for loop (no spreadsheet))
    - see what is taking the most time
- check code with random seed parameter
- put models into functions defined in the test function file
- tried to profile pycharm
- worked on the write_to_file function
- should I have the function to write edge and adjacency lists in detection.py or in the community graph file?
    - put it into the community graph file
- delete current community_graph and replace with community_graph 

************************************************************************************************************************
Questions:
- how to profile on pycharm (bashsupport plugin installed)
    - wait for meeting Friday, consider meeting with qi (he leaves by the end of may)
- Why does the file writing handle an empty string path in the way it does (all the way into the community_detection
 folder); is this a sign that the directories I create are poorly named?
- tried to not make everything a string for the constraint list comprehension (didn't work)

************************************************************************************************************************
To do:
- ask if the create model function in the test case seems good, if so, then do that for all the others
    - instead, import the small files that are already on pyomo (see the link he just sent)

- then profile the code on a sizable file (look up how to profile on pycharm)

- fix comments and docstrings within the functions
    - clean up the files to be pushed to pyomo
- Take a look at the stuff david sent you
- Don't forget your rst file
- use if name = main thing for testing

- do a pull request onto your own code, don't accept changes until zedong and qi have looked over your code
    - be comfortable with github
    - read over the materials

Schedule:
- finish this code by Sunday night
- also read github and the work!
- make a pull request on monday night
- first week read up on stuff (github and the work)


************************************************************************************************************************
- Options for summer project:
    - center cut (or 2nd order approx. or regularization)
    - CORAMIN project
    - reviving old mindtpy code (Extended Cutting Plane (ECP), Partial Surrogate Cuts (PSC), Bender's Decomposition)
    - implement non-grossmann contributions (Extended supporting hyperplane, decomposition outer-approximation (germany paper)
    
    - ideas:
        - decomposition outer-approximation (germany paper)
        - ECP and then ESP
        - CORAMIN
'''

'''
Old questions:
- Discuss summer research (in terms of what you talked about with prof schneider)
    - really get something out of research by not spreading yourself too thin 
        - getting published
        - meeting with Prof Grossmann as well
    - which will also set me up for a potential grad school application
    - Also, TAing doesn't start until Summer 2 (if I get the summer TA job) and will continue 
    through Fall (if I get the Fall TA job as well)
- Show David the test file and see how else to call those files or what to do instead of calling those files
- also something kind of cool is that for some reason pycharm is able to run the enormous files that would cause
spyder to crash (but 'exit code -1073741571 (0xC00000FD)' instead of 'exit code 0')
'''


def detect_communities(model, node_type='v', with_objective=True, weighted_graph=True, write_to_path=None, random_seed=None):
    """
    Detects communities in a graph of variables and constraints

    This function takes in a Pyomo optimization model, organizes the variables and constraints into a graph of nodes
     and edges, and then uses Louvain community detection to create a dictionary of the communities of the nodes.
     Either variables or constraints can be chosen as the nodes.

    Args:
        model (Block): a Pyomo model or block to be used for community detection
        node_type : a string that specifies the dictionary to be returned; 'v' returns a dictionary with communities
        based on variable nodes, 'c' returns a dictionary with communities based on constraint nodes, and any other
        input returns an error message
        with_objective: an optional Boolean argument that specifies whether or not the objective function will be
        treated as a node/constraint (depending on what node_type is specified (see prior argument))
        weighted_graph: an optional Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model
        write_to_path: an optional argument that takes in a path for edge lists and adjacency lists to be saved
        random_seed : takes in an integer to use as the seed number for the Louvain community detection

    Return:
        community_map: a Python dictionary whose keys are integers from one to the number of communities with
        values that are lists of the nodes in the given community
    """

    #if node_type != 'v' and node_type != 'c':
     #   print("Invalid input: Specify node_type 'v' or 'c' for function detect_communities")
      #  return None

    model_graph = community_graph._generate_model_graph(model, node_type=node_type, with_objective=with_objective,
                                                            weighted_graph=weighted_graph, write_to_path=write_to_path)

    partition_of_graph = community.best_partition(model_graph, random_state=random_seed)
    n_communities = int(len(set(partition_of_graph.values())))
    community_map = {nth_community: [] for nth_community in range(n_communities)}

    for node in partition_of_graph:
        nth_community = partition_of_graph[node]
        community_map[nth_community].append(node)

    return community_map



models_location = 'D:\College\Sophomore Year\PSE Research\Current Work\Relevant\Rewritten Models'

# First test case
#file = 'pedigree_sim2000.py'
file = 'alan.py'
exfile = import_file(models_location + '\\' + file)
model = exfile.create_model()

print(detect_communities(model, node_type='v'))


def create_model():
    model = ConcreteModel()
    model.x = Var()
    model.y = Var()
    model.o = Objective(expr=model.x + model.y)

    return model

model = create_model()
print(detect_communities(model))


"""
models_location = 'D:\College\Sophomore Year\PSE Research\Current Work\Relevant\Rewritten Models'

# First test case
#file = 'pedigree_sim2000.py'
file = 'm3.py'
exfile = import_file(models_location + '\\' + file)
model = exfile.create_model()

detect_communities(model, node_type='f')

#detect_communities(model, node_type='v', with_objective=True, weighted_graph=True, write_to_path='D:\College\Sophomore Year\PSE Research')

# Use m3.py
#print(detect_communities(model, random_seed=5))
#print(detect_communities(model, random_seed=6))

'''
test_info = community_graph._generate_model_graph(model, node_type='v', with_objective=True)

yay = True
for i in range(1000):
    test1 = community.best_partition(test_info, random_state=5)
    test2 = community.best_partition(test_info, random_state=5)
    test3 = community.best_partition(test_info, random_state=5)
    test4 = community.best_partition(test_info, random_state=5)
    if not (test1 == test2 == test3 == test4):
        print('BAD TEST CASE!')
        yay = False
        break

if yay: print('Yay!')
'''
"""
