"""
Created by Rahul Joglekar, 5:54PM, 5/12/2020
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


overallStart = time.time()

troubleFiles = ['meanvarxsc.py', 'color_lab6b_4x20.py', 'pedigree_sp_top4_250.py', 'pedigree_sp_top5_250.py',
                'watercontamination0202r.py']

# If models are in the same folder then just make this variable equal to '/'
models_location = 'D:\College\Sophomore Year\PSE Research\Current Work\Relevant\Rewritten Models'

# Modify this string to name the new folder that will be created
#output_folder = 'NewIntersectionGraphInfo'

#if output_folder not in os.listdir():
    #os.mkdir(output_folder)

solutions = defaultdict(list)

keys = []

solutions.fromkeys(keys, [])

for file in os.listdir(models_location):

    fileSize = os.path.getsize(models_location + '\\' + file)

    if file in troubleFiles or fileSize > 976000:
        continue

    elif file.endswith('.py'): #and (file + '.UnweightedAdjList') not in os.listdir(output_folder):

        print(file, ': file_size =', fileSize)

        exfile = import_file(models_location + '\\' + file)

        model = exfile.create_model()

        #storage_location = str(output_folder) + '\\' + file

        # Time the graph generation
        start = time.time()
        start_community = time.time()

        community_map_v_unweighted_without = detect_communities(model, node_type='v',with_objective=False,weighted_graph=False)
        community_map_v_weighted_without = detect_communities(model, node_type='v',with_objective=False,weighted_graph=True)
        community_map_v_unweighted_with = detect_communities(model, node_type='v',with_objective=True,weighted_graph=False)
        community_map_v_weighted_with = detect_communities(model, node_type='v',with_objective=True,weighted_graph=True)
        community_map_c_unweighted_without = detect_communities(model, node_type='c',with_objective=False,weighted_graph=False)
        community_map_c_weighted_without = detect_communities(model, node_type='c',with_objective=False,weighted_graph=True)
        community_map_c_unweighted_with = detect_communities(model, node_type='c',with_objective=True,weighted_graph=False)
        community_map_c_weighted_with = detect_communities(model, node_type='c',with_objective=True,weighted_graph=True)

        end_community = time.time()

        avg_time_community_map_creation = round((end_community - start_community)/8, 3)
        print(f"Average time to create community map for {file} was {avg_time_community_map_creation} seconds.")

        solutions['Model Name'].append(file)
        solutions['Average time for Community Detection'].append(avg_time_community_map_creation)
        solutions['nCommunities_v_unweighted_without'].append(len(community_map_v_unweighted_without))
        solutions['nCommunities_v_weighted_without'].append(len(community_map_v_weighted_without))
        solutions['nCommunities_v_unweighted_with'].append(len(community_map_v_unweighted_with))
        solutions['nCommunities_v_weighted_with'].append(len(community_map_v_weighted_with))
        solutions['nCommunities_c_unweighted_without'].append(len(community_map_c_unweighted_without))
        solutions['nCommunities_c_weighted_without'].append(len(community_map_c_weighted_without))
        solutions['nCommunities_c_unweighted_with'].append(len(community_map_c_unweighted_with))
        solutions['nCommunities_c_weighted_with'].append(len(community_map_c_weighted_with))

        end = time.time()
        print(f"Done! This file took {end - start} seconds to create all 8 graphs and make the excel row.")
        print('\n')

sol_total = pd.DataFrame.from_dict(solutions)

sol_total.to_excel('D:\College\Sophomore Year\PSE Research\Current Work\Relevant\Community_Graph_Stuff.xlsx')

overallEnd = time.time()
print('OVERALL: ', overallEnd - overallStart)


# OldOne:
"""
squfl030-150.py 755075
Done! 1437.2149405479431
4196.7 total
"""

# NewOne:
"""
squfl030-150.py 829702
Done! 1404.2582640647888
"""

# NewerOne:
"""
squfl030-150.py : file_size = 829702
Done! 306.11998319625854
"""

"""
Even Newer:
squfl030-150.py : file_size = 807060
Done! 109.0726203918457
"""
# bigBois = ['color_lab2_4x0.py', 'meanvar-orl400_05_e_8.py', 'pedigree_ex1058.py', 'pedigree_sim2000.py', 'pedigree_sim400.py', 'pedigree_sp_top5_200.py', 'portfol_classical200_2.py', 'unitcommit_200_100_1_mod_8.py', 'unitcommit_200_100_2_mod_8.py', 'unitcommit_50_20_2_mod_8.py', 'watercontamination0202.py', 'watercontamination0303.py', 'watercontamination0303r.py']




"""
alan.py : file_size = 1988
Average time to create community map for alan.py was 0.002 seconds.
Done! This file took 0.019946575164794922 seconds to create all 8 graphs and make the excel row.


ball_mk2_10.py : file_size = 2091
Average time to create community map for ball_mk2_10.py was 0.002 seconds.
Done! This file took 0.015956878662109375 seconds to create all 8 graphs and make the excel row.


ball_mk2_30.py : file_size = 4471
Average time to create community map for ball_mk2_30.py was 0.01 seconds.
Done! This file took 0.0822901725769043 seconds to create all 8 graphs and make the excel row.


ball_mk3_10.py : file_size = 2384
Average time to create community map for ball_mk3_10.py was 0.002 seconds.
Done! This file took 0.014959096908569336 seconds to create all 8 graphs and make the excel row.


ball_mk3_20.py : file_size = 3889
Average time to create community map for ball_mk3_20.py was 0.004 seconds.
Done! This file took 0.03596758842468262 seconds to create all 8 graphs and make the excel row.


ball_mk3_30.py : file_size = 5369
Average time to create community map for ball_mk3_30.py was 0.009 seconds.
Done! This file took 0.07280325889587402 seconds to create all 8 graphs and make the excel row.


ball_mk4_05.py : file_size = 2253
Average time to create community map for ball_mk4_05.py was 0.002 seconds.
Done! This file took 0.017523765563964844 seconds to create all 8 graphs and make the excel row.


ball_mk4_10.py : file_size = 3588
Average time to create community map for ball_mk4_10.py was 0.006 seconds.
Done! This file took 0.04886937141418457 seconds to create all 8 graphs and make the excel row.


ball_mk4_15.py : file_size = 4923
Average time to create community map for ball_mk4_15.py was 0.012 seconds.
Done! This file took 0.09873461723327637 seconds to create all 8 graphs and make the excel row.


batch.py : file_size = 9604
Average time to create community map for batch.py was 0.048 seconds.
Done! This file took 0.3831944465637207 seconds to create all 8 graphs and make the excel row.


batch0812.py : file_size = 24849
Average time to create community map for batch0812.py was 0.258 seconds.
Done! This file took 2.0662035942077637 seconds to create all 8 graphs and make the excel row.


batchdes.py : file_size = 3833
Average time to create community map for batchdes.py was 0.007 seconds.
Done! This file took 0.05985236167907715 seconds to create all 8 graphs and make the excel row.


batchs101006m.py : file_size = 109071
Average time to create community map for batchs101006m.py was 3.331 seconds.
Done! This file took 26.649457931518555 seconds to create all 8 graphs and make the excel row.


batchs121208m.py : file_size = 161904
Average time to create community map for batchs121208m.py was 6.973 seconds.
Done! This file took 55.78130483627319 seconds to create all 8 graphs and make the excel row.


batchs151208m.py : file_size = 189891
Average time to create community map for batchs151208m.py was 9.287 seconds.
Done! This file took 74.29948234558105 seconds to create all 8 graphs and make the excel row.


batchs201210m.py : file_size = 248067
Average time to create community map for batchs201210m.py was 15.952 seconds.
Done! This file took 127.61509895324707 seconds to create all 8 graphs and make the excel row.


clay0203h.py : file_size = 18390
Average time to create community map for clay0203h.py was 0.178 seconds.
Done! This file took 1.4261741638183594 seconds to create all 8 graphs and make the excel row.


clay0203m.py : file_size = 6703
Average time to create community map for clay0203m.py was 0.039 seconds.
Done! This file took 0.3082244396209717 seconds to create all 8 graphs and make the excel row.


clay0204h.py : file_size = 31233
Average time to create community map for clay0204h.py was 0.531 seconds.
Done! This file took 4.244704008102417 seconds to create all 8 graphs and make the excel row.


clay0204m.py : file_size = 10534
Average time to create community map for clay0204m.py was 0.082 seconds.
Done! This file took 0.654287576675415 seconds to create all 8 graphs and make the excel row.


clay0205h.py : file_size = 46993
Average time to create community map for clay0205h.py was 1.307 seconds.
Done! This file took 10.455208778381348 seconds to create all 8 graphs and make the excel row.


clay0205m.py : file_size = 15330
Average time to create community map for clay0205m.py was 0.177 seconds.
Done! This file took 1.4122793674468994 seconds to create all 8 graphs and make the excel row.


clay0303h.py : file_size = 22124
Average time to create community map for clay0303h.py was 0.227 seconds.
Done! This file took 1.8176252841949463 seconds to create all 8 graphs and make the excel row.


clay0303m.py : file_size = 7961
Average time to create community map for clay0303m.py was 0.047 seconds.
Done! This file took 0.3749966621398926 seconds to create all 8 graphs and make the excel row.


clay0304h.py : file_size = 36417
Average time to create community map for clay0304h.py was 0.682 seconds.
Done! This file took 5.4574103355407715 seconds to create all 8 graphs and make the excel row.


clay0304m.py : file_size = 12190
Average time to create community map for clay0304m.py was 0.108 seconds.
Done! This file took 0.8663372993469238 seconds to create all 8 graphs and make the excel row.


clay0305h.py : file_size = 53501
Average time to create community map for clay0305h.py was 1.433 seconds.
Done! This file took 11.466949939727783 seconds to create all 8 graphs and make the excel row.


clay0305m.py : file_size = 17424
Average time to create community map for clay0305m.py was 0.216 seconds.
Done! This file took 1.724860429763794 seconds to create all 8 graphs and make the excel row.


cvxnonsep_normcon20.py : file_size = 2737
Average time to create community map for cvxnonsep_normcon20.py was 0.004 seconds.
Done! This file took 0.03390979766845703 seconds to create all 8 graphs and make the excel row.


cvxnonsep_normcon20r.py : file_size = 4934
Average time to create community map for cvxnonsep_normcon20r.py was 0.016 seconds.
Done! This file took 0.12666058540344238 seconds to create all 8 graphs and make the excel row.


cvxnonsep_normcon30.py : file_size = 3643
Average time to create community map for cvxnonsep_normcon30.py was 0.009 seconds.
Done! This file took 0.07279324531555176 seconds to create all 8 graphs and make the excel row.


cvxnonsep_normcon30r.py : file_size = 6960
Average time to create community map for cvxnonsep_normcon30r.py was 0.034 seconds.
Done! This file took 0.27045464515686035 seconds to create all 8 graphs and make the excel row.


cvxnonsep_normcon40.py : file_size = 4523
Average time to create community map for cvxnonsep_normcon40.py was 0.014 seconds.
Done! This file took 0.11034989356994629 seconds to create all 8 graphs and make the excel row.


cvxnonsep_normcon40r.py : file_size = 8840
Average time to create community map for cvxnonsep_normcon40r.py was 0.058 seconds.
Done! This file took 0.4668557643890381 seconds to create all 8 graphs and make the excel row.


cvxnonsep_nsig20.py : file_size = 2835
Average time to create community map for cvxnonsep_nsig20.py was 0.004 seconds.
Done! This file took 0.031951189041137695 seconds to create all 8 graphs and make the excel row.


cvxnonsep_nsig20r.py : file_size = 5189
Average time to create community map for cvxnonsep_nsig20r.py was 0.017 seconds.
Done! This file took 0.13915109634399414 seconds to create all 8 graphs and make the excel row.


cvxnonsep_nsig30.py : file_size = 3833
Average time to create community map for cvxnonsep_nsig30.py was 0.009 seconds.
Done! This file took 0.0688166618347168 seconds to create all 8 graphs and make the excel row.


cvxnonsep_nsig30r.py : file_size = 7349
Average time to create community map for cvxnonsep_nsig30r.py was 0.036 seconds.
Done! This file took 0.28427958488464355 seconds to create all 8 graphs and make the excel row.


cvxnonsep_nsig40.py : file_size = 4775
Average time to create community map for cvxnonsep_nsig40.py was 0.015 seconds.
Done! This file took 0.11664557456970215 seconds to create all 8 graphs and make the excel row.


cvxnonsep_nsig40r.py : file_size = 9482
Average time to create community map for cvxnonsep_nsig40r.py was 0.061 seconds.
Done! This file took 0.4862501621246338 seconds to create all 8 graphs and make the excel row.


cvxnonsep_pcon20.py : file_size = 2953
Average time to create community map for cvxnonsep_pcon20.py was 0.005 seconds.
Done! This file took 0.042571306228637695 seconds to create all 8 graphs and make the excel row.


cvxnonsep_pcon20r.py : file_size = 4983
Average time to create community map for cvxnonsep_pcon20r.py was 0.021 seconds.
Done! This file took 0.16551852226257324 seconds to create all 8 graphs and make the excel row.


cvxnonsep_pcon30.py : file_size = 3992
Average time to create community map for cvxnonsep_pcon30.py was 0.009 seconds.
Done! This file took 0.07375812530517578 seconds to create all 8 graphs and make the excel row.


cvxnonsep_pcon30r.py : file_size = 7112
Average time to create community map for cvxnonsep_pcon30r.py was 0.043 seconds.
Done! This file took 0.34133172035217285 seconds to create all 8 graphs and make the excel row.


cvxnonsep_pcon40.py : file_size = 5007
Average time to create community map for cvxnonsep_pcon40.py was 0.015 seconds.
Done! This file took 0.11766791343688965 seconds to create all 8 graphs and make the excel row.


cvxnonsep_pcon40r.py : file_size = 9218
Average time to create community map for cvxnonsep_pcon40r.py was 0.09 seconds.
Done! This file took 0.7160873413085938 seconds to create all 8 graphs and make the excel row.


cvxnonsep_psig20.py : file_size = 2700
Average time to create community map for cvxnonsep_psig20.py was 0.002 seconds.
Done! This file took 0.014961957931518555 seconds to create all 8 graphs and make the excel row.


cvxnonsep_psig20r.py : file_size = 5132
Average time to create community map for cvxnonsep_psig20r.py was 0.019 seconds.
Done! This file took 0.1491093635559082 seconds to create all 8 graphs and make the excel row.


cvxnonsep_psig30.py : file_size = 3608
Average time to create community map for cvxnonsep_psig30.py was 0.004 seconds.
Done! This file took 0.03195476531982422 seconds to create all 8 graphs and make the excel row.


cvxnonsep_psig30r.py : file_size = 7142
Average time to create community map for cvxnonsep_psig30r.py was 0.037 seconds.
Done! This file took 0.29919862747192383 seconds to create all 8 graphs and make the excel row.


cvxnonsep_psig40.py : file_size = 4514
Average time to create community map for cvxnonsep_psig40.py was 0.007 seconds.
Done! This file took 0.05286049842834473 seconds to create all 8 graphs and make the excel row.


cvxnonsep_psig40r.py : file_size = 9280
Average time to create community map for cvxnonsep_psig40r.py was 0.064 seconds.
Done! This file took 0.5081536769866943 seconds to create all 8 graphs and make the excel row.


du-opt.py : file_size = 64367
Average time to create community map for du-opt.py was 0.028 seconds.
Done! This file took 0.2224586009979248 seconds to create all 8 graphs and make the excel row.


du-opt5.py : file_size = 63716
Average time to create community map for du-opt5.py was 0.026 seconds.
Done! This file took 0.20847535133361816 seconds to create all 8 graphs and make the excel row.


enpro48pb.py : file_size = 30080
Average time to create community map for enpro48pb.py was 0.508 seconds.
Done! This file took 4.061587333679199 seconds to create all 8 graphs and make the excel row.


enpro56pb.py : file_size = 26145
Average time to create community map for enpro56pb.py was 0.373 seconds.
Done! This file took 2.9813287258148193 seconds to create all 8 graphs and make the excel row.


ex4.py : file_size = 9536
Average time to create community map for ex4.py was 0.041 seconds.
Done! This file took 0.3276054859161377 seconds to create all 8 graphs and make the excel row.


fac1.py : file_size = 3799
Average time to create community map for fac1.py was 0.011 seconds.
Done! This file took 0.0907602310180664 seconds to create all 8 graphs and make the excel row.


fac2.py : file_size = 9234
Average time to create community map for fac2.py was 0.065 seconds.
Done! This file took 0.5176162719726562 seconds to create all 8 graphs and make the excel row.


fac3.py : file_size = 9228
Average time to create community map for fac3.py was 0.062 seconds.
Done! This file took 0.49695301055908203 seconds to create all 8 graphs and make the excel row.


flay02h.py : file_size = 6676
Average time to create community map for flay02h.py was 0.038 seconds.
Done! This file took 0.30663561820983887 seconds to create all 8 graphs and make the excel row.


flay02m.py : file_size = 2432
Average time to create community map for flay02m.py was 0.005 seconds.
Done! This file took 0.03987884521484375 seconds to create all 8 graphs and make the excel row.


flay03h.py : file_size = 16932
Average time to create community map for flay03h.py was 0.236 seconds.
Done! This file took 1.8906190395355225 seconds to create all 8 graphs and make the excel row.


flay03m.py : file_size = 3997
Average time to create community map for flay03m.py was 0.015 seconds.
Done! This file took 0.12063813209533691 seconds to create all 8 graphs and make the excel row.


flay04h.py : file_size = 33606
Average time to create community map for flay04h.py was 0.813 seconds.
Done! This file took 6.504581451416016 seconds to create all 8 graphs and make the excel row.


flay04m.py : file_size = 6748
Average time to create community map for flay04m.py was 0.038 seconds.
Done! This file took 0.30014848709106445 seconds to create all 8 graphs and make the excel row.


flay05h.py : file_size = 52894
Average time to create community map for flay05h.py was 2.075 seconds.
Done! This file took 16.598747491836548 seconds to create all 8 graphs and make the excel row.


flay05m.py : file_size = 8898
Average time to create community map for flay05m.py was 0.08 seconds.
Done! This file took 0.6422839164733887 seconds to create all 8 graphs and make the excel row.


flay06h.py : file_size = 78409
Average time to create community map for flay06h.py was 4.417 seconds.
Done! This file took 35.33653521537781 seconds to create all 8 graphs and make the excel row.


flay06m.py : file_size = 12228
Average time to create community map for flay06m.py was 0.131 seconds.
Done! This file took 1.044862985610962 seconds to create all 8 graphs and make the excel row.


fo7.py : file_size = 24461
Average time to create community map for fo7.py was 0.448 seconds.
Done! This file took 3.5846781730651855 seconds to create all 8 graphs and make the excel row.


fo7_2.py : file_size = 24618
Average time to create community map for fo7_2.py was 0.454 seconds.
Done! This file took 3.6333043575286865 seconds to create all 8 graphs and make the excel row.


fo7_ar25_1.py : file_size = 28993
Average time to create community map for fo7_ar25_1.py was 0.508 seconds.
Done! This file took 4.066170930862427 seconds to create all 8 graphs and make the excel row.


fo7_ar2_1.py : file_size = 28982
Average time to create community map for fo7_ar2_1.py was 0.513 seconds.
Done! This file took 4.10402512550354 seconds to create all 8 graphs and make the excel row.


fo7_ar3_1.py : file_size = 28994
Average time to create community map for fo7_ar3_1.py was 0.586 seconds.
Done! This file took 4.6914591789245605 seconds to create all 8 graphs and make the excel row.


fo7_ar4_1.py : file_size = 28775
Average time to create community map for fo7_ar4_1.py was 0.538 seconds.
Done! This file took 4.30544376373291 seconds to create all 8 graphs and make the excel row.


fo7_ar5_1.py : file_size = 28980
Average time to create community map for fo7_ar5_1.py was 0.473 seconds.
Done! This file took 3.7808408737182617 seconds to create all 8 graphs and make the excel row.


fo8.py : file_size = 31948
Average time to create community map for fo8.py was 0.576 seconds.
Done! This file took 4.610668182373047 seconds to create all 8 graphs and make the excel row.


fo8_ar25_1.py : file_size = 37669
Average time to create community map for fo8_ar25_1.py was 0.775 seconds.
Done! This file took 6.196429252624512 seconds to create all 8 graphs and make the excel row.


fo8_ar2_1.py : file_size = 37657
Average time to create community map for fo8_ar2_1.py was 0.787 seconds.
Done! This file took 6.293216228485107 seconds to create all 8 graphs and make the excel row.


fo8_ar3_1.py : file_size = 37675
Average time to create community map for fo8_ar3_1.py was 0.81 seconds.
Done! This file took 6.482666254043579 seconds to create all 8 graphs and make the excel row.


fo8_ar4_1.py : file_size = 37433
Average time to create community map for fo8_ar4_1.py was 0.813 seconds.
Done! This file took 6.501575469970703 seconds to create all 8 graphs and make the excel row.


fo8_ar5_1.py : file_size = 37657
Average time to create community map for fo8_ar5_1.py was 0.78 seconds.
Done! This file took 6.23831844329834 seconds to create all 8 graphs and make the excel row.


fo9.py : file_size = 39501
Average time to create community map for fo9.py was 0.971 seconds.
Done! This file took 7.770205736160278 seconds to create all 8 graphs and make the excel row.


fo9_ar25_1.py : file_size = 46894
Average time to create community map for fo9_ar25_1.py was 1.224 seconds.
Done! This file took 9.792768001556396 seconds to create all 8 graphs and make the excel row.


fo9_ar2_1.py : file_size = 46878
Average time to create community map for fo9_ar2_1.py was 1.263 seconds.
Done! This file took 10.10597825050354 seconds to create all 8 graphs and make the excel row.


fo9_ar3_1.py : file_size = 46900
Average time to create community map for fo9_ar3_1.py was 1.236 seconds.
Done! This file took 9.890606164932251 seconds to create all 8 graphs and make the excel row.


fo9_ar4_1.py : file_size = 46614
Average time to create community map for fo9_ar4_1.py was 1.333 seconds.
Done! This file took 10.665416717529297 seconds to create all 8 graphs and make the excel row.


fo9_ar5_1.py : file_size = 46860
Average time to create community map for fo9_ar5_1.py was 1.187 seconds.
Done! This file took 9.49561357498169 seconds to create all 8 graphs and make the excel row.


gams01.py : file_size = 350271
Average time to create community map for gams01.py was 13.593 seconds.
Done! This file took 108.74221110343933 seconds to create all 8 graphs and make the excel row.


gbd.py : file_size = 1442
Average time to create community map for gbd.py was 0.001 seconds.
Done! This file took 0.007976055145263672 seconds to create all 8 graphs and make the excel row.


hybriddynamic_fixed.py : file_size = 9946
Average time to create community map for hybriddynamic_fixed.py was 0.062 seconds.
Done! This file took 0.4936344623565674 seconds to create all 8 graphs and make the excel row.


ibs2.py : file_size = 585682
Average time to create community map for ibs2.py was 128.893 seconds.
Done! This file took 1031.1462042331696 seconds to create all 8 graphs and make the excel row.


jit1.py : file_size = 5566
Average time to create community map for jit1.py was 0.017 seconds.
Done! This file took 0.13460111618041992 seconds to create all 8 graphs and make the excel row.


m3.py : file_size = 5526
Average time to create community map for m3.py was 0.028 seconds.
Done! This file took 0.22594308853149414 seconds to create all 8 graphs and make the excel row.


m6.py : file_size = 17906
Average time to create community map for m6.py was 0.271 seconds.
Done! This file took 2.170128345489502 seconds to create all 8 graphs and make the excel row.


m7.py : file_size = 24206
Average time to create community map for m7.py was 0.478 seconds.
Done! This file took 3.820037364959717 seconds to create all 8 graphs and make the excel row.


m7_ar25_1.py : file_size = 28416
Average time to create community map for m7_ar25_1.py was 0.588 seconds.
Done! This file took 4.701040267944336 seconds to create all 8 graphs and make the excel row.


m7_ar2_1.py : file_size = 28514
Average time to create community map for m7_ar2_1.py was 0.605 seconds.
Done! This file took 4.837021589279175 seconds to create all 8 graphs and make the excel row.


m7_ar3_1.py : file_size = 28451
Average time to create community map for m7_ar3_1.py was 0.628 seconds.
Done! This file took 5.021639108657837 seconds to create all 8 graphs and make the excel row.


m7_ar4_1.py : file_size = 28400
Average time to create community map for m7_ar4_1.py was 0.602 seconds.
Done! This file took 4.813596248626709 seconds to create all 8 graphs and make the excel row.


m7_ar5_1.py : file_size = 28441
Average time to create community map for m7_ar5_1.py was 0.597 seconds.
Done! This file took 4.779046535491943 seconds to create all 8 graphs and make the excel row.


meanvarx.py : file_size = 6288
Average time to create community map for meanvarx.py was 0.029 seconds.
Done! This file took 0.22806334495544434 seconds to create all 8 graphs and make the excel row.


netmod_dol1.py : file_size = 331445
Average time to create community map for netmod_dol1.py was 73.449 seconds.
Done! This file took 587.5920193195343 seconds to create all 8 graphs and make the excel row.


netmod_dol2.py : file_size = 466929
Average time to create community map for netmod_dol2.py was 134.57 seconds.
Done! This file took 1076.5591406822205 seconds to create all 8 graphs and make the excel row.


netmod_kar1.py : file_size = 71322
Average time to create community map for netmod_kar1.py was 3.995 seconds.
Done! This file took 31.959614992141724 seconds to create all 8 graphs and make the excel row.


netmod_kar2.py : file_size = 71322
Average time to create community map for netmod_kar2.py was 3.947 seconds.
Done! This file took 31.579572439193726 seconds to create all 8 graphs and make the excel row.


no7_ar25_1.py : file_size = 29061
Average time to create community map for no7_ar25_1.py was 0.617 seconds.
Done! This file took 4.934477806091309 seconds to create all 8 graphs and make the excel row.


no7_ar2_1.py : file_size = 29050
Average time to create community map for no7_ar2_1.py was 0.62 seconds.
Done! This file took 4.9608542919158936 seconds to create all 8 graphs and make the excel row.


no7_ar3_1.py : file_size = 29062
Average time to create community map for no7_ar3_1.py was 0.629 seconds.
Done! This file took 5.034695148468018 seconds to create all 8 graphs and make the excel row.


no7_ar4_1.py : file_size = 28843
Average time to create community map for no7_ar4_1.py was 0.599 seconds.
Done! This file took 4.794709920883179 seconds to create all 8 graphs and make the excel row.


no7_ar5_1.py : file_size = 29048
Average time to create community map for no7_ar5_1.py was 0.584 seconds.
Done! This file took 4.668791770935059 seconds to create all 8 graphs and make the excel row.


nvs03.py : file_size = 1225
Average time to create community map for nvs03.py was 0.001 seconds.
Done! This file took 0.0040280818939208984 seconds to create all 8 graphs and make the excel row.


nvs10.py : file_size = 1262
Average time to create community map for nvs10.py was 0.0 seconds.
Done! This file took 0.003979682922363281 seconds to create all 8 graphs and make the excel row.


nvs11.py : file_size = 1588
Average time to create community map for nvs11.py was 0.001 seconds.
Done! This file took 0.007978439331054688 seconds to create all 8 graphs and make the excel row.


nvs12.py : file_size = 2137
Average time to create community map for nvs12.py was 0.001 seconds.
Done! This file took 0.01096963882446289 seconds to create all 8 graphs and make the excel row.


nvs15.py : file_size = 1311
Average time to create community map for nvs15.py was 0.0 seconds.
Done! This file took 0.00394439697265625 seconds to create all 8 graphs and make the excel row.


o7.py : file_size = 24578
Average time to create community map for o7.py was 0.482 seconds.
Done! This file took 3.852247476577759 seconds to create all 8 graphs and make the excel row.


o7_2.py : file_size = 24735
Average time to create community map for o7_2.py was 0.489 seconds.
Done! This file took 3.9124960899353027 seconds to create all 8 graphs and make the excel row.


o7_ar25_1.py : file_size = 29110
Average time to create community map for o7_ar25_1.py was 0.626 seconds.
Done! This file took 5.009578466415405 seconds to create all 8 graphs and make the excel row.


o7_ar2_1.py : file_size = 29099
Average time to create community map for o7_ar2_1.py was 0.624 seconds.
Done! This file took 4.993535280227661 seconds to create all 8 graphs and make the excel row.


o7_ar3_1.py : file_size = 29111
Average time to create community map for o7_ar3_1.py was 0.613 seconds.
Done! This file took 4.901932239532471 seconds to create all 8 graphs and make the excel row.


o7_ar4_1.py : file_size = 28892
Average time to create community map for o7_ar4_1.py was 0.639 seconds.
Done! This file took 5.110163927078247 seconds to create all 8 graphs and make the excel row.


o7_ar5_1.py : file_size = 29097
Average time to create community map for o7_ar5_1.py was 0.618 seconds.
Done! This file took 4.942325830459595 seconds to create all 8 graphs and make the excel row.


o8_ar4_1.py : file_size = 37879
Average time to create community map for o8_ar4_1.py was 0.953 seconds.
Done! This file took 7.621399641036987 seconds to create all 8 graphs and make the excel row.


o9_ar4_1.py : file_size = 47095
Average time to create community map for o9_ar4_1.py was 1.457 seconds.
Done! This file took 11.65309476852417 seconds to create all 8 graphs and make the excel row.


pedigree_ex485.py : file_size = 250068
Average time to create community map for pedigree_ex485.py was 10.221 seconds.
Done! This file took 81.76724696159363 seconds to create all 8 graphs and make the excel row.


pedigree_ex485_2.py : file_size = 250074
Average time to create community map for pedigree_ex485_2.py was 10.566 seconds.
Done! This file took 84.52830743789673 seconds to create all 8 graphs and make the excel row.


pedigree_sp_top4_300.py : file_size = 217529
Average time to create community map for pedigree_sp_top4_300.py was 11.493 seconds.
Done! This file took 91.94299793243408 seconds to create all 8 graphs and make the excel row.


pedigree_sp_top4_350tr.py : file_size = 109962
Average time to create community map for pedigree_sp_top4_350tr.py was 2.576 seconds.
Done! This file took 20.60657835006714 seconds to create all 8 graphs and make the excel row.


portfol_buyin.py : file_size = 16461
Average time to create community map for portfol_buyin.py was 0.015 seconds.
Done! This file took 0.1186838150024414 seconds to create all 8 graphs and make the excel row.


portfol_card.py : file_size = 16569
Average time to create community map for portfol_card.py was 0.017 seconds.
Done! This file took 0.13559174537658691 seconds to create all 8 graphs and make the excel row.


portfol_classical050_1.py : file_size = 78827
Average time to create community map for portfol_classical050_1.py was 1.407 seconds.
Done! This file took 11.258504867553711 seconds to create all 8 graphs and make the excel row.


portfol_roundlot.py : file_size = 16111
Average time to create community map for portfol_roundlot.py was 0.014 seconds.
Done! This file took 0.11057114601135254 seconds to create all 8 graphs and make the excel row.


procurement2mot.py : file_size = 104026
Average time to create community map for procurement2mot.py was 8.218 seconds.
Done! This file took 65.74488759040833 seconds to create all 8 graphs and make the excel row.


ravempb.py : file_size = 24906
Average time to create community map for ravempb.py was 0.329 seconds.
Done! This file took 2.632291793823242 seconds to create all 8 graphs and make the excel row.


risk2bpb.py : file_size = 78122
Average time to create community map for risk2bpb.py was 4.225 seconds.
Done! This file took 33.80276393890381 seconds to create all 8 graphs and make the excel row.


rsyn0805h.py : file_size = 46882
Average time to create community map for rsyn0805h.py was 1.534 seconds.
Done! This file took 12.268717765808105 seconds to create all 8 graphs and make the excel row.


rsyn0805m.py : file_size = 30637
Average time to create community map for rsyn0805m.py was 0.61 seconds.
Done! This file took 4.879138946533203 seconds to create all 8 graphs and make the excel row.


rsyn0805m02h.py : file_size = 108978
Average time to create community map for rsyn0805m02h.py was 8.126 seconds.
Done! This file took 65.01103186607361 seconds to create all 8 graphs and make the excel row.


rsyn0805m02m.py : file_size = 73397
Average time to create community map for rsyn0805m02m.py was 3.307 seconds.
Done! This file took 26.45740556716919 seconds to create all 8 graphs and make the excel row.


rsyn0805m03h.py : file_size = 171969
Average time to create community map for rsyn0805m03h.py was 19.47 seconds.
Done! This file took 155.75915670394897 seconds to create all 8 graphs and make the excel row.


rsyn0805m03m.py : file_size = 118099
Average time to create community map for rsyn0805m03m.py was 7.767 seconds.
Done! This file took 62.13285040855408 seconds to create all 8 graphs and make the excel row.


rsyn0805m04h.py : file_size = 242519
Average time to create community map for rsyn0805m04h.py was 37.256 seconds.
Done! This file took 298.0485279560089 seconds to create all 8 graphs and make the excel row.


rsyn0805m04m.py : file_size = 167881
Average time to create community map for rsyn0805m04m.py was 15.791 seconds.
Done! This file took 126.32872009277344 seconds to create all 8 graphs and make the excel row.


rsyn0810h.py : file_size = 52658
Average time to create community map for rsyn0810h.py was 1.904 seconds.
Done! This file took 15.232783317565918 seconds to create all 8 graphs and make the excel row.


rsyn0810m.py : file_size = 33361
Average time to create community map for rsyn0810m.py was 0.777 seconds.
Done! This file took 6.213551759719849 seconds to create all 8 graphs and make the excel row.


rsyn0810m02h.py : file_size = 123869
Average time to create community map for rsyn0810m02h.py was 12.14 seconds.
Done! This file took 97.11819243431091 seconds to create all 8 graphs and make the excel row.


rsyn0810m02m.py : file_size = 82627
Average time to create community map for rsyn0810m02m.py was 4.037 seconds.
Done! This file took 32.29929757118225 seconds to create all 8 graphs and make the excel row.


rsyn0810m03h.py : file_size = 196377
Average time to create community map for rsyn0810m03h.py was 24.967 seconds.
Done! This file took 199.73701405525208 seconds to create all 8 graphs and make the excel row.


rsyn0810m03m.py : file_size = 133415
Average time to create community map for rsyn0810m03m.py was 10.542 seconds.
Done! This file took 84.33316326141357 seconds to create all 8 graphs and make the excel row.


rsyn0810m04h.py : file_size = 277019
Average time to create community map for rsyn0810m04h.py was 47.991 seconds.
Done! This file took 383.9251434803009 seconds to create all 8 graphs and make the excel row.


rsyn0810m04m.py : file_size = 190104
Average time to create community map for rsyn0810m04m.py was 20.644 seconds.
Done! This file took 165.15566635131836 seconds to create all 8 graphs and make the excel row.


rsyn0815h.py : file_size = 60010
Average time to create community map for rsyn0815h.py was 2.488 seconds.
Done! This file took 19.901718854904175 seconds to create all 8 graphs and make the excel row.


rsyn0815m.py : file_size = 36945
Average time to create community map for rsyn0815m.py was 1.029 seconds.
Done! This file took 8.23559856414795 seconds to create all 8 graphs and make the excel row.


rsyn0815m02h.py : file_size = 142044
Average time to create community map for rsyn0815m02h.py was 13.084 seconds.
Done! This file took 104.67552208900452 seconds to create all 8 graphs and make the excel row.


rsyn0815m02m.py : file_size = 93653
Average time to create community map for rsyn0815m02m.py was 4.874 seconds.
Done! This file took 38.99169301986694 seconds to create all 8 graphs and make the excel row.


rsyn0815m03h.py : file_size = 225817
Average time to create community map for rsyn0815m03h.py was 28.685 seconds.
Done! This file took 229.47824358940125 seconds to create all 8 graphs and make the excel row.


rsyn0815m03m.py : file_size = 151422
Average time to create community map for rsyn0815m03m.py was 11.727 seconds.
Done! This file took 93.81614756584167 seconds to create all 8 graphs and make the excel row.


rsyn0815m04h.py : file_size = 318285
Average time to create community map for rsyn0815m04h.py was 53.43 seconds.
Done! This file took 427.4407608509064 seconds to create all 8 graphs and make the excel row.


rsyn0815m04m.py : file_size = 215917
Average time to create community map for rsyn0815m04m.py was 23.493 seconds.
Done! This file took 187.94636940956116 seconds to create all 8 graphs and make the excel row.


rsyn0820h.py : file_size = 65364
Average time to create community map for rsyn0820h.py was 2.599 seconds.
Done! This file took 20.791420936584473 seconds to create all 8 graphs and make the excel row.


rsyn0820m.py : file_size = 39241
Average time to create community map for rsyn0820m.py was 0.928 seconds.
Done! This file took 7.427139759063721 seconds to create all 8 graphs and make the excel row.


rsyn0820m02h.py : file_size = 156043
Average time to create community map for rsyn0820m02h.py was 19.158 seconds.
Done! This file took 153.26499390602112 seconds to create all 8 graphs and make the excel row.


rsyn0820m02m.py : file_size = 102073
Average time to create community map for rsyn0820m02m.py was 7.043 seconds.
Done! This file took 56.3442964553833 seconds to create all 8 graphs and make the excel row.


rsyn0820m03h.py : file_size = 248813
Average time to create community map for rsyn0820m03h.py was 34.784 seconds.
Done! This file took 278.270072221756 seconds to create all 8 graphs and make the excel row.


rsyn0820m03m.py : file_size = 165402
Average time to create community map for rsyn0820m03m.py was 13.75 seconds.
Done! This file took 109.99783277511597 seconds to create all 8 graphs and make the excel row.


rsyn0820m04h.py : file_size = 350772
Average time to create community map for rsyn0820m04h.py was 61.901 seconds.
Done! This file took 495.20710825920105 seconds to create all 8 graphs and make the excel row.


rsyn0820m04m.py : file_size = 236388
Average time to create community map for rsyn0820m04m.py was 26.99 seconds.
Done! This file took 215.9205605983734 seconds to create all 8 graphs and make the excel row.


rsyn0830h.py : file_size = 77735
Average time to create community map for rsyn0830h.py was 3.471 seconds.
Done! This file took 27.764745473861694 seconds to create all 8 graphs and make the excel row.


rsyn0830m.py : file_size = 45222
Average time to create community map for rsyn0830m.py was 1.159 seconds.
Done! This file took 9.270223617553711 seconds to create all 8 graphs and make the excel row.


rsyn0830m02h.py : file_size = 188651
Average time to create community map for rsyn0830m02h.py was 19.874 seconds.
Done! This file took 158.99487590789795 seconds to create all 8 graphs and make the excel row.


rsyn0830m02m.py : file_size = 121743
Average time to create community map for rsyn0830m02m.py was 8.774 seconds.
Done! This file took 70.19030213356018 seconds to create all 8 graphs and make the excel row.


rsyn0830m03h.py : file_size = 299728
Average time to create community map for rsyn0830m03h.py was 44.131 seconds.
Done! This file took 353.04607582092285 seconds to create all 8 graphs and make the excel row.


rsyn0830m03m.py : file_size = 197023
Average time to create community map for rsyn0830m03m.py was 17.267 seconds.
Done! This file took 138.13563656806946 seconds to create all 8 graphs and make the excel row.


rsyn0830m04h.py : file_size = 422005
Average time to create community map for rsyn0830m04h.py was 89.53 seconds.
Done! This file took 716.2369484901428 seconds to create all 8 graphs and make the excel row.


rsyn0830m04m.py : file_size = 284016
Average time to create community map for rsyn0830m04m.py was 39.138 seconds.
Done! This file took 313.1069748401642 seconds to create all 8 graphs and make the excel row.


rsyn0840h.py : file_size = 90440
Average time to create community map for rsyn0840h.py was 4.874 seconds.
Done! This file took 38.99336099624634 seconds to create all 8 graphs and make the excel row.


rsyn0840m.py : file_size = 51107
Average time to create community map for rsyn0840m.py was 1.451 seconds.
Done! This file took 11.609430074691772 seconds to create all 8 graphs and make the excel row.


rsyn0840m02h.py : file_size = 221954
Average time to create community map for rsyn0840m02h.py was 26.83 seconds.
Done! This file took 214.64016556739807 seconds to create all 8 graphs and make the excel row.


rsyn0840m02m.py : file_size = 141484
Average time to create community map for rsyn0840m02m.py was 9.918 seconds.
Done! This file took 79.34181571006775 seconds to create all 8 graphs and make the excel row.


rsyn0840m03h.py : file_size = 352620
Average time to create community map for rsyn0840m03h.py was 64.258 seconds.
Done! This file took 514.0658314228058 seconds to create all 8 graphs and make the excel row.


rsyn0840m03m.py : file_size = 230053
Average time to create community map for rsyn0840m03m.py was 25.993 seconds.
Done! This file took 207.94426155090332 seconds to create all 8 graphs and make the excel row.


rsyn0840m04h.py : file_size = 495763
Average time to create community map for rsyn0840m04h.py was 131.009 seconds.
Done! This file took 1048.074509382248 seconds to create all 8 graphs and make the excel row.


rsyn0840m04m.py : file_size = 332735
Average time to create community map for rsyn0840m04m.py was 57.118 seconds.
Done! This file took 456.9413228034973 seconds to create all 8 graphs and make the excel row.


slay04h.py : file_size = 20534
Average time to create community map for slay04h.py was 0.38 seconds.
Done! This file took 3.0438966751098633 seconds to create all 8 graphs and make the excel row.


slay04m.py : file_size = 7239
Average time to create community map for slay04m.py was 0.054 seconds.
Done! This file took 0.4328806400299072 seconds to create all 8 graphs and make the excel row.


slay05h.py : file_size = 33634
Average time to create community map for slay05h.py was 0.741 seconds.
Done! This file took 5.924642562866211 seconds to create all 8 graphs and make the excel row.


slay05m.py : file_size = 11165
Average time to create community map for slay05m.py was 0.097 seconds.
Done! This file took 0.7749521732330322 seconds to create all 8 graphs and make the excel row.


slay06h.py : file_size = 49902
Average time to create community map for slay06h.py was 1.694 seconds.
Done! This file took 13.548687219619751 seconds to create all 8 graphs and make the excel row.


slay06m.py : file_size = 16121
Average time to create community map for slay06m.py was 0.196 seconds.
Done! This file took 1.5648562908172607 seconds to create all 8 graphs and make the excel row.


slay07h.py : file_size = 69449
Average time to create community map for slay07h.py was 3.433 seconds.
Done! This file took 27.46102738380432 seconds to create all 8 graphs and make the excel row.


slay07m.py : file_size = 22206
Average time to create community map for slay07m.py was 0.391 seconds.
Done! This file took 3.1316206455230713 seconds to create all 8 graphs and make the excel row.


slay08h.py : file_size = 92171
Average time to create community map for slay08h.py was 6.583 seconds.
Done! This file took 52.66729474067688 seconds to create all 8 graphs and make the excel row.


slay08m.py : file_size = 29206
Average time to create community map for slay08m.py was 0.694 seconds.
Done! This file took 5.551708221435547 seconds to create all 8 graphs and make the excel row.


slay09h.py : file_size = 118166
Average time to create community map for slay09h.py was 10.061 seconds.
Done! This file took 80.48880767822266 seconds to create all 8 graphs and make the excel row.


slay09m.py : file_size = 37216
Average time to create community map for slay09m.py was 1.076 seconds.
Done! This file took 8.60849118232727 seconds to create all 8 graphs and make the excel row.


slay10h.py : file_size = 147645
Average time to create community map for slay10h.py was 15.337 seconds.
Done! This file took 122.69309329986572 seconds to create all 8 graphs and make the excel row.


slay10m.py : file_size = 46196
Average time to create community map for slay10m.py was 1.774 seconds.
Done! This file took 14.188486814498901 seconds to create all 8 graphs and make the excel row.


smallinvDAXr1b010-011.py : file_size = 19303
Average time to create community map for smallinvDAXr1b010-011.py was 0.037 seconds.
Done! This file took 0.29775118827819824 seconds to create all 8 graphs and make the excel row.


smallinvDAXr1b020-022.py : file_size = 19303
Average time to create community map for smallinvDAXr1b020-022.py was 0.039 seconds.
Done! This file took 0.30809783935546875 seconds to create all 8 graphs and make the excel row.


smallinvDAXr1b050-055.py : file_size = 19303
Average time to create community map for smallinvDAXr1b050-055.py was 0.037 seconds.
Done! This file took 0.29218006134033203 seconds to create all 8 graphs and make the excel row.


smallinvDAXr1b100-110.py : file_size = 19305
Average time to create community map for smallinvDAXr1b100-110.py was 0.038 seconds.
Done! This file took 0.30516743659973145 seconds to create all 8 graphs and make the excel row.


smallinvDAXr1b150-165.py : file_size = 19305
Average time to create community map for smallinvDAXr1b150-165.py was 0.041 seconds.
Done! This file took 0.3261291980743408 seconds to create all 8 graphs and make the excel row.


smallinvDAXr1b200-220.py : file_size = 19305
Average time to create community map for smallinvDAXr1b200-220.py was 0.043 seconds.
Done! This file took 0.34108877182006836 seconds to create all 8 graphs and make the excel row.


smallinvDAXr2b010-011.py : file_size = 19303
Average time to create community map for smallinvDAXr2b010-011.py was 0.04 seconds.
Done! This file took 0.32128310203552246 seconds to create all 8 graphs and make the excel row.


smallinvDAXr2b020-022.py : file_size = 19303
Average time to create community map for smallinvDAXr2b020-022.py was 0.051 seconds.
Done! This file took 0.4119741916656494 seconds to create all 8 graphs and make the excel row.


smallinvDAXr2b050-055.py : file_size = 19303
Average time to create community map for smallinvDAXr2b050-055.py was 0.057 seconds.
Done! This file took 0.45725274085998535 seconds to create all 8 graphs and make the excel row.


smallinvDAXr2b100-110.py : file_size = 19305
Average time to create community map for smallinvDAXr2b100-110.py was 0.056 seconds.
Done! This file took 0.4478020668029785 seconds to create all 8 graphs and make the excel row.


smallinvDAXr2b150-165.py : file_size = 19305
Average time to create community map for smallinvDAXr2b150-165.py was 0.048 seconds.
Done! This file took 0.3850841522216797 seconds to create all 8 graphs and make the excel row.


smallinvDAXr2b200-220.py : file_size = 19305
Average time to create community map for smallinvDAXr2b200-220.py was 0.045 seconds.
Done! This file took 0.35707736015319824 seconds to create all 8 graphs and make the excel row.


smallinvDAXr3b010-011.py : file_size = 19302
Average time to create community map for smallinvDAXr3b010-011.py was 0.051 seconds.
Done! This file took 0.4056673049926758 seconds to create all 8 graphs and make the excel row.


smallinvDAXr3b020-022.py : file_size = 19302
Average time to create community map for smallinvDAXr3b020-022.py was 0.047 seconds.
Done! This file took 0.37296009063720703 seconds to create all 8 graphs and make the excel row.


smallinvDAXr3b050-055.py : file_size = 19302
Average time to create community map for smallinvDAXr3b050-055.py was 0.041 seconds.
Done! This file took 0.3301560878753662 seconds to create all 8 graphs and make the excel row.


smallinvDAXr3b100-110.py : file_size = 19304
Average time to create community map for smallinvDAXr3b100-110.py was 0.051 seconds.
Done! This file took 0.40595293045043945 seconds to create all 8 graphs and make the excel row.


smallinvDAXr3b150-165.py : file_size = 19304
Average time to create community map for smallinvDAXr3b150-165.py was 0.042 seconds.
Done! This file took 0.3393678665161133 seconds to create all 8 graphs and make the excel row.


smallinvDAXr3b200-220.py : file_size = 19304
Average time to create community map for smallinvDAXr3b200-220.py was 0.052 seconds.
Done! This file took 0.41492772102355957 seconds to create all 8 graphs and make the excel row.


smallinvDAXr4b010-011.py : file_size = 19301
Average time to create community map for smallinvDAXr4b010-011.py was 0.047 seconds.
Done! This file took 0.3789856433868408 seconds to create all 8 graphs and make the excel row.


smallinvDAXr4b020-022.py : file_size = 19301
Average time to create community map for smallinvDAXr4b020-022.py was 0.051 seconds.
Done! This file took 0.4059572219848633 seconds to create all 8 graphs and make the excel row.


smallinvDAXr4b050-055.py : file_size = 19301
Average time to create community map for smallinvDAXr4b050-055.py was 0.042 seconds.
Done! This file took 0.33304309844970703 seconds to create all 8 graphs and make the excel row.


smallinvDAXr4b100-110.py : file_size = 19303
Average time to create community map for smallinvDAXr4b100-110.py was 0.045 seconds.
Done! This file took 0.35702061653137207 seconds to create all 8 graphs and make the excel row.


smallinvDAXr4b150-165.py : file_size = 19303
Average time to create community map for smallinvDAXr4b150-165.py was 0.042 seconds.
Done! This file took 0.3351407051086426 seconds to create all 8 graphs and make the excel row.


smallinvDAXr4b200-220.py : file_size = 19303
Average time to create community map for smallinvDAXr4b200-220.py was 0.052 seconds.
Done! This file took 0.41651248931884766 seconds to create all 8 graphs and make the excel row.


smallinvDAXr5b010-011.py : file_size = 19301
Average time to create community map for smallinvDAXr5b010-011.py was 0.046 seconds.
Done! This file took 0.3717153072357178 seconds to create all 8 graphs and make the excel row.


smallinvDAXr5b020-022.py : file_size = 19301
Average time to create community map for smallinvDAXr5b020-022.py was 0.048 seconds.
Done! This file took 0.38547801971435547 seconds to create all 8 graphs and make the excel row.


smallinvDAXr5b050-055.py : file_size = 19301
Average time to create community map for smallinvDAXr5b050-055.py was 0.045 seconds.
Done! This file took 0.3630342483520508 seconds to create all 8 graphs and make the excel row.


smallinvDAXr5b100-110.py : file_size = 19303
Average time to create community map for smallinvDAXr5b100-110.py was 0.042 seconds.
Done! This file took 0.33606553077697754 seconds to create all 8 graphs and make the excel row.


smallinvDAXr5b150-165.py : file_size = 19303
Average time to create community map for smallinvDAXr5b150-165.py was 0.043 seconds.
Done! This file took 0.3432316780090332 seconds to create all 8 graphs and make the excel row.


smallinvDAXr5b200-220.py : file_size = 19303
Average time to create community map for smallinvDAXr5b200-220.py was 0.043 seconds.
Done! This file took 0.34267425537109375 seconds to create all 8 graphs and make the excel row.


squfl010-025.py : file_size = 43955
Average time to create community map for squfl010-025.py was 1.319 seconds.
Done! This file took 10.554311752319336 seconds to create all 8 graphs and make the excel row.


squfl010-040.py : file_size = 69761
Average time to create community map for squfl010-040.py was 2.902 seconds.
Done! This file took 23.213780164718628 seconds to create all 8 graphs and make the excel row.


squfl010-080.py : file_size = 138533
Average time to create community map for squfl010-080.py was 10.857 seconds.
Done! This file took 86.85352349281311 seconds to create all 8 graphs and make the excel row.


squfl015-060.py : file_size = 156876
Average time to create community map for squfl015-060.py was 13.187 seconds.
Done! This file took 105.49416160583496 seconds to create all 8 graphs and make the excel row.


squfl015-080.py : file_size = 211465
Average time to create community map for squfl015-080.py was 22.979 seconds.
Done! This file took 183.83018898963928 seconds to create all 8 graphs and make the excel row.


squfl020-040.py : file_size = 139992
Average time to create community map for squfl020-040.py was 10.176 seconds.
Done! This file took 81.41198205947876 seconds to create all 8 graphs and make the excel row.


squfl020-050.py : file_size = 176164
Average time to create community map for squfl020-050.py was 16.328 seconds.
Done! This file took 130.62774968147278 seconds to create all 8 graphs and make the excel row.


squfl020-150.py : file_size = 537893
Average time to create community map for squfl020-150.py was 140.64 seconds.
Done! This file took 1125.1160695552826 seconds to create all 8 graphs and make the excel row.


squfl025-025.py : file_size = 109755
Average time to create community map for squfl025-025.py was 6.433 seconds.
Done! This file took 51.4675350189209 seconds to create all 8 graphs and make the excel row.


squfl025-030.py : file_size = 131289
Average time to create community map for squfl025-030.py was 11.41 seconds.
Done! This file took 91.28232049942017 seconds to create all 8 graphs and make the excel row.


squfl025-040.py : file_size = 175492
Average time to create community map for squfl025-040.py was 16.49 seconds.
Done! This file took 131.9231607913971 seconds to create all 8 graphs and make the excel row.


squfl030-100.py : file_size = 536736
Average time to create community map for squfl030-100.py was 151.311 seconds.
Done! This file took 1210.4884350299835 seconds to create all 8 graphs and make the excel row.


squfl030-150.py : file_size = 807060
Average time to create community map for squfl030-150.py was 319.879 seconds.
Done! This file took 2559.033420085907 seconds to create all 8 graphs and make the excel row.


squfl040-080.py : file_size = 572592
Average time to create community map for squfl040-080.py was 159.564 seconds.
Done! This file took 1276.5092933177948 seconds to create all 8 graphs and make the excel row.


sssd08-04.py : file_size = 9358
Average time to create community map for sssd08-04.py was 0.064 seconds.
Done! This file took 0.5086398124694824 seconds to create all 8 graphs and make the excel row.


sssd12-05.py : file_size = 14159
Average time to create community map for sssd12-05.py was 0.122 seconds.
Done! This file took 0.9783835411071777 seconds to create all 8 graphs and make the excel row.


sssd15-04.py : file_size = 13287
Average time to create community map for sssd15-04.py was 0.091 seconds.
Done! This file took 0.7240326404571533 seconds to create all 8 graphs and make the excel row.


sssd15-06.py : file_size = 19388
Average time to create community map for sssd15-06.py was 0.207 seconds.
Done! This file took 1.6525523662567139 seconds to create all 8 graphs and make the excel row.


sssd15-08.py : file_size = 25555
Average time to create community map for sssd15-08.py was 0.357 seconds.
Done! This file took 2.8563601970672607 seconds to create all 8 graphs and make the excel row.


sssd16-07.py : file_size = 23405
Average time to create community map for sssd16-07.py was 0.326 seconds.
Done! This file took 2.6060314178466797 seconds to create all 8 graphs and make the excel row.


sssd18-06.py : file_size = 21894
Average time to create community map for sssd18-06.py was 0.264 seconds.
Done! This file took 2.1143479347229004 seconds to create all 8 graphs and make the excel row.


sssd18-08.py : file_size = 28859
Average time to create community map for sssd18-08.py was 0.431 seconds.
Done! This file took 3.4477789402008057 seconds to create all 8 graphs and make the excel row.


sssd20-04.py : file_size = 15991
Average time to create community map for sssd20-04.py was 0.129 seconds.
Done! This file took 1.032240867614746 seconds to create all 8 graphs and make the excel row.


sssd20-08.py : file_size = 30935
Average time to create community map for sssd20-08.py was 0.496 seconds.
Done! This file took 3.9653942584991455 seconds to create all 8 graphs and make the excel row.


sssd22-08.py : file_size = 33258
Average time to create community map for sssd22-08.py was 0.541 seconds.
Done! This file took 4.329420328140259 seconds to create all 8 graphs and make the excel row.


sssd25-04.py : file_size = 18909
Average time to create community map for sssd25-04.py was 0.21 seconds.
Done! This file took 1.676516056060791 seconds to create all 8 graphs and make the excel row.


sssd25-08.py : file_size = 36492
Average time to create community map for sssd25-08.py was 0.638 seconds.
Done! This file took 5.1033830642700195 seconds to create all 8 graphs and make the excel row.


stockcycle.py : file_size = 44759
Average time to create community map for stockcycle.py was 1.36 seconds.
Done! This file took 10.879903316497803 seconds to create all 8 graphs and make the excel row.


st_miqp1.py : file_size = 1476
Average time to create community map for st_miqp1.py was 0.001 seconds.
Done! This file took 0.011994123458862305 seconds to create all 8 graphs and make the excel row.


st_miqp2.py : file_size = 1415
Average time to create community map for st_miqp2.py was 0.001 seconds.
Done! This file took 0.004984378814697266 seconds to create all 8 graphs and make the excel row.


st_miqp3.py : file_size = 1136
Average time to create community map for st_miqp3.py was 0.0 seconds.
Done! This file took 0.0030243396759033203 seconds to create all 8 graphs and make the excel row.


st_miqp4.py : file_size = 1635
Average time to create community map for st_miqp4.py was 0.001 seconds.
Done! This file took 0.007954597473144531 seconds to create all 8 graphs and make the excel row.


st_miqp5.py : file_size = 3941
Average time to create community map for st_miqp5.py was 0.003 seconds.
Done! This file took 0.026925086975097656 seconds to create all 8 graphs and make the excel row.


st_test1.py : file_size = 1461
Average time to create community map for st_test1.py was 0.001 seconds.
Done! This file took 0.004989147186279297 seconds to create all 8 graphs and make the excel row.


st_test2.py : file_size = 1617
Average time to create community map for st_test2.py was 0.002 seconds.
Done! This file took 0.01296234130859375 seconds to create all 8 graphs and make the excel row.


st_test3.py : file_size = 2512
Average time to create community map for st_test3.py was 0.003 seconds.
Done! This file took 0.027944087982177734 seconds to create all 8 graphs and make the excel row.


st_test4.py : file_size = 1839
Average time to create community map for st_test4.py was 0.001 seconds.
Done! This file took 0.010923624038696289 seconds to create all 8 graphs and make the excel row.


st_test5.py : file_size = 3038
Average time to create community map for st_test5.py was 0.005 seconds.
Done! This file took 0.03989243507385254 seconds to create all 8 graphs and make the excel row.


st_test6.py : file_size = 2434
Average time to create community map for st_test6.py was 0.004 seconds.
Done! This file took 0.028919219970703125 seconds to create all 8 graphs and make the excel row.


st_test8.py : file_size = 4719
Average time to create community map for st_test8.py was 0.013 seconds.
Done! This file took 0.10567855834960938 seconds to create all 8 graphs and make the excel row.


st_testgr1.py : file_size = 2499
Average time to create community map for st_testgr1.py was 0.003 seconds.
Done! This file took 0.024973392486572266 seconds to create all 8 graphs and make the excel row.


st_testgr3.py : file_size = 5319
Average time to create community map for st_testgr3.py was 0.02 seconds.
Done! This file took 0.15853071212768555 seconds to create all 8 graphs and make the excel row.


st_testph4.py : file_size = 1765
Average time to create community map for st_testph4.py was 0.002 seconds.
Done! This file took 0.013963460922241211 seconds to create all 8 graphs and make the excel row.


syn05h.py : file_size = 7181
Average time to create community map for syn05h.py was 0.032 seconds.
Done! This file took 0.2582845687866211 seconds to create all 8 graphs and make the excel row.


syn05m.py : file_size = 4003
Average time to create community map for syn05m.py was 0.012 seconds.
Done! This file took 0.09970450401306152 seconds to create all 8 graphs and make the excel row.


syn05m02h.py : file_size = 16652
Average time to create community map for syn05m02h.py was 0.214 seconds.
Done! This file took 1.7124245166778564 seconds to create all 8 graphs and make the excel row.


syn05m02m.py : file_size = 10746
Average time to create community map for syn05m02m.py was 0.083 seconds.
Done! This file took 0.6602301597595215 seconds to create all 8 graphs and make the excel row.


syn05m03h.py : file_size = 26201
Average time to create community map for syn05m03h.py was 0.445 seconds.
Done! This file took 3.559483766555786 seconds to create all 8 graphs and make the excel row.


syn05m03m.py : file_size = 16967
Average time to create community map for syn05m03m.py was 0.219 seconds.
Done! This file took 1.752309799194336 seconds to create all 8 graphs and make the excel row.


syn05m04h.py : file_size = 36559
Average time to create community map for syn05m04h.py was 0.818 seconds.
Done! This file took 6.547490358352661 seconds to create all 8 graphs and make the excel row.


syn05m04m.py : file_size = 24133
Average time to create community map for syn05m04m.py was 0.41 seconds.
Done! This file took 3.279203176498413 seconds to create all 8 graphs and make the excel row.


syn10h.py : file_size = 12753
Average time to create community map for syn10h.py was 0.11 seconds.
Done! This file took 0.8816163539886475 seconds to create all 8 graphs and make the excel row.


syn10m.py : file_size = 6615
Average time to create community map for syn10m.py was 0.033 seconds.
Done! This file took 0.26329469680786133 seconds to create all 8 graphs and make the excel row.


syn10m02h.py : file_size = 31529
Average time to create community map for syn10m02h.py was 0.634 seconds.
Done! This file took 5.075424671173096 seconds to create all 8 graphs and make the excel row.


syn10m02m.py : file_size = 19725
Average time to create community map for syn10m02m.py was 0.279 seconds.
Done! This file took 2.2340283393859863 seconds to create all 8 graphs and make the excel row.


syn10m03h.py : file_size = 49710
Average time to create community map for syn10m03h.py was 1.5 seconds.
Done! This file took 11.99886178970337 seconds to create all 8 graphs and make the excel row.


syn10m03m.py : file_size = 32063
Average time to create community map for syn10m03m.py was 0.675 seconds.
Done! This file took 5.403546094894409 seconds to create all 8 graphs and make the excel row.


syn10m04h.py : file_size = 69573
Average time to create community map for syn10m04h.py was 2.851 seconds.
Done! This file took 22.806041717529297 seconds to create all 8 graphs and make the excel row.


syn10m04m.py : file_size = 46359
Average time to create community map for syn10m04m.py was 1.304 seconds.
Done! This file took 10.429110527038574 seconds to create all 8 graphs and make the excel row.


syn15h.py : file_size = 20063
Average time to create community map for syn15h.py was 0.248 seconds.
Done! This file took 1.987654685974121 seconds to create all 8 graphs and make the excel row.


syn15m.py : file_size = 10008
Average time to create community map for syn15m.py was 0.075 seconds.
Done! This file took 0.6023890972137451 seconds to create all 8 graphs and make the excel row.


syn15m02h.py : file_size = 49539
Average time to create community map for syn15m02h.py was 1.552 seconds.
Done! This file took 12.412803173065186 seconds to create all 8 graphs and make the excel row.


syn15m02m.py : file_size = 30758
Average time to create community map for syn15m02m.py was 0.641 seconds.
Done! This file took 5.131277561187744 seconds to create all 8 graphs and make the excel row.


syn15m03h.py : file_size = 77954
Average time to create community map for syn15m03h.py was 3.483 seconds.
Done! This file took 27.863486528396606 seconds to create all 8 graphs and make the excel row.


syn15m03m.py : file_size = 50026
Average time to create community map for syn15m03m.py was 1.728 seconds.
Done! This file took 13.824168682098389 seconds to create all 8 graphs and make the excel row.


syn15m04h.py : file_size = 109136
Average time to create community map for syn15m04h.py was 6.751 seconds.
Done! This file took 54.00656795501709 seconds to create all 8 graphs and make the excel row.


syn15m04m.py : file_size = 71817
Average time to create community map for syn15m04m.py was 3.038 seconds.
Done! This file took 24.30304193496704 seconds to create all 8 graphs and make the excel row.


syn20h.py : file_size = 25411
Average time to create community map for syn20h.py was 0.429 seconds.
Done! This file took 3.4328627586364746 seconds to create all 8 graphs and make the excel row.


syn20m.py : file_size = 12211
Average time to create community map for syn20m.py was 0.111 seconds.
Done! This file took 0.885629415512085 seconds to create all 8 graphs and make the excel row.


syn20m02h.py : file_size = 63407
Average time to create community map for syn20m02h.py was 2.384 seconds.
Done! This file took 19.072988748550415 seconds to create all 8 graphs and make the excel row.


syn20m02m.py : file_size = 39155
Average time to create community map for syn20m02m.py was 0.991 seconds.
Done! This file took 7.924803972244263 seconds to create all 8 graphs and make the excel row.


syn20m03h.py : file_size = 100063
Average time to create community map for syn20m03h.py was 5.75 seconds.
Done! This file took 46.000977516174316 seconds to create all 8 graphs and make the excel row.


syn20m03m.py : file_size = 63835
Average time to create community map for syn20m03m.py was 2.494 seconds.
Done! This file took 19.952648162841797 seconds to create all 8 graphs and make the excel row.


syn20m04h.py : file_size = 140687
Average time to create community map for syn20m04h.py was 10.97 seconds.
Done! This file took 87.75726699829102 seconds to create all 8 graphs and make the excel row.


syn20m04m.py : file_size = 92032
Average time to create community map for syn20m04m.py was 5.536 seconds.
Done! This file took 44.28955340385437 seconds to create all 8 graphs and make the excel row.


syn30h.py : file_size = 37831
Average time to create community map for syn30h.py was 0.862 seconds.
Done! This file took 6.893537521362305 seconds to create all 8 graphs and make the excel row.


syn30m.py : file_size = 18044
Average time to create community map for syn30m.py was 0.219 seconds.
Done! This file took 1.7512900829315186 seconds to create all 8 graphs and make the excel row.


syn30m02h.py : file_size = 94512
Average time to create community map for syn30m02h.py was 5.044 seconds.
Done! This file took 40.35008788108826 seconds to create all 8 graphs and make the excel row.


syn30m02m.py : file_size = 58681
Average time to create community map for syn30m02m.py was 2.084 seconds.
Done! This file took 16.67441964149475 seconds to create all 8 graphs and make the excel row.


syn30m03h.py : file_size = 149821
Average time to create community map for syn30m03h.py was 12.765 seconds.
Done! This file took 102.12089085578918 seconds to create all 8 graphs and make the excel row.


syn30m03m.py : file_size = 95697
Average time to create community map for syn30m03m.py was 5.312 seconds.
Done! This file took 42.49539303779602 seconds to create all 8 graphs and make the excel row.


syn30m04h.py : file_size = 211154
Average time to create community map for syn30m04h.py was 20.444 seconds.
Done! This file took 163.55269145965576 seconds to create all 8 graphs and make the excel row.


syn30m04m.py : file_size = 138454
Average time to create community map for syn30m04m.py was 9.519 seconds.
Done! This file took 76.15034985542297 seconds to create all 8 graphs and make the excel row.


syn40h.py : file_size = 50557
Average time to create community map for syn40h.py was 1.274 seconds.
Done! This file took 10.195732116699219 seconds to create all 8 graphs and make the excel row.


syn40m.py : file_size = 23976
Average time to create community map for syn40m.py was 0.39 seconds.
Done! This file took 3.116626024246216 seconds to create all 8 graphs and make the excel row.


syn40m02h.py : file_size = 126698
Average time to create community map for syn40m02h.py was 9.308 seconds.
Done! This file took 74.46779155731201 seconds to create all 8 graphs and make the excel row.


syn40m02m.py : file_size = 78065
Average time to create community map for syn40m02m.py was 3.209 seconds.
Done! This file took 25.673341035842896 seconds to create all 8 graphs and make the excel row.


syn40m03h.py : file_size = 202199
Average time to create community map for syn40m03h.py was 19.45 seconds.
Done! This file took 155.6009156703949 seconds to create all 8 graphs and make the excel row.


syn40m03m.py : file_size = 127678
Average time to create community map for syn40m03m.py was 7.919 seconds.
Done! This file took 63.35461902618408 seconds to create all 8 graphs and make the excel row.


syn40m04h.py : file_size = 286908
Average time to create community map for syn40m04h.py was 38.704 seconds.
Done! This file took 309.6329822540283 seconds to create all 8 graphs and make the excel row.


syn40m04m.py : file_size = 184669
Average time to create community map for syn40m04m.py was 16.873 seconds.
Done! This file took 134.98699808120728 seconds to create all 8 graphs and make the excel row.


synthes1.py : file_size = 1800
Average time to create community map for synthes1.py was 0.001 seconds.
Done! This file took 0.01196742057800293 seconds to create all 8 graphs and make the excel row.


synthes2.py : file_size = 2596
Average time to create community map for synthes2.py was 0.004 seconds.
Done! This file took 0.0348811149597168 seconds to create all 8 graphs and make the excel row.


synthes3.py : file_size = 3783
Average time to create community map for synthes3.py was 0.011 seconds.
Done! This file took 0.08481693267822266 seconds to create all 8 graphs and make the excel row.


tls12.py : file_size = 167535
Average time to create community map for tls12.py was 16.523 seconds.
Done! This file took 132.1864902973175 seconds to create all 8 graphs and make the excel row.


tls2.py : file_size = 6253
Average time to create community map for tls2.py was 0.027 seconds.
Done! This file took 0.2174234390258789 seconds to create all 8 graphs and make the excel row.


tls4.py : file_size = 16702
Average time to create community map for tls4.py was 0.233 seconds.
Done! This file took 1.8669581413269043 seconds to create all 8 graphs and make the excel row.


tls5.py : file_size = 25530
Average time to create community map for tls5.py was 0.447 seconds.
Done! This file took 3.577434539794922 seconds to create all 8 graphs and make the excel row.


tls6.py : file_size = 35433
Average time to create community map for tls6.py was 0.846 seconds.
Done! This file took 6.770889759063721 seconds to create all 8 graphs and make the excel row.


tls7.py : file_size = 57296
Average time to create community map for tls7.py was 2.194 seconds.
Done! This file took 17.55201482772827 seconds to create all 8 graphs and make the excel row.


unitcommit1.py : file_size = 391612
Average time to create community map for unitcommit1.py was 44.215 seconds.
Done! This file took 353.7211265563965 seconds to create all 8 graphs and make the excel row.


OVERALL:  22139.359604120255"""






