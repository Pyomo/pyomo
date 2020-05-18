"""
Created by Rahul Joglekar, 8:44PM, 5/12/2020
"""


import os
from pyomo.environ import *
from pyomo.contrib.community_detection.detection import *
from pyomo.core import ConcreteModel
from pyutilib.misc import import_file
import community
import time
from collections import defaultdict
import pandas as pd
import networkx as nx


def give_me_an_update(start, count):
    print(f"Community detection {count}: {time.time() - start} seconds.")


overall_start = time.time()

troubleFiles = ['meanvarxsc.py', 'color_lab6b_4x20.py', 'pedigree_sp_top4_250.py', 'pedigree_sp_top5_250.py',
                'watercontamination0202r.py']

# If models are in the same folder then just make this variable equal to '/'
models_location = 'D:\College\Sophomore Year\PSE Research\Current Work\Relevant\Rewritten Models'

for file in os.listdir(models_location):

    fileSize = os.path.getsize(models_location + '\\' + file)

    if file in troubleFiles: #or fileSize > 976000:
        continue

    elif file.startswith('squfl010-025.py'):
        #file.endswith('.py'): #and (file + '.UnweightedAdjList') not in os.listdir(output_folder):

        print(file, ': file_size =', fileSize)

        exfile = import_file(models_location + '\\' + file)

        model = exfile.create_model()

        #storage_location = str(output_folder) + '\\' + file

        # Time the graph generation
        start_community = time.time()
        start = time.time()
        count = 0
        community_map_v_unweighted_without = detect_communities(model, node_type='v',with_objective=False,weighted_graph=False)
        count += 1
        give_me_an_update(start, count)
        start = time.time()
        community_map_v_weighted_without = detect_communities(model, node_type='v',with_objective=False,weighted_graph=True)
        count += 1
        give_me_an_update(start, count)
        start = time.time()
        community_map_v_unweighted_with = detect_communities(model, node_type='v',with_objective=True,weighted_graph=False)
        count += 1
        give_me_an_update(start, count)
        start = time.time()
        community_map_v_weighted_with = detect_communities(model, node_type='v',with_objective=True,weighted_graph=True)
        count += 1
        give_me_an_update(start, count)
        start = time.time()
        community_map_c_unweighted_without = detect_communities(model, node_type='c',with_objective=False,weighted_graph=False)
        count += 1
        give_me_an_update(start, count)
        start = time.time()
        community_map_c_weighted_without = detect_communities(model, node_type='c',with_objective=False,weighted_graph=True)
        count += 1
        give_me_an_update(start, count)
        start = time.time()
        community_map_c_unweighted_with = detect_communities(model, node_type='c',with_objective=True,weighted_graph=False)
        count += 1
        give_me_an_update(start, count)
        start = time.time()
        community_map_c_weighted_with = detect_communities(model, node_type='c',with_objective=True,weighted_graph=True)
        end_community = time.time()
        count += 1
        give_me_an_update(start, count)

        avg_time_community_map_creation = round((end_community - start_community)/8, 3)
        print(f"Average time for {file} was {avg_time_community_map_creation} seconds.")

overall_end = time.time()
print('OVERALL: ', overall_end - overall_start)

