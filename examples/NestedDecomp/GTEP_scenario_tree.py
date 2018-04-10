import networkx as nx
import collections

#Build scneario tree
G = nx.DiGraph()

node_origin = ['O']
scenarios = ['L','R','H']
stages = range(1,6)

#Create a set all the nodes in the scenario tree
N = ['F']
NP = ['F']
N1 = list(node_origin)
N2 = []
N3 = []
N4 = []
N5 = []
for t in stages:
    for s2 in scenarios:
        if t==2:
            N2.append('O'+str(s2))
        elif t==3:
            for s3 in scenarios:
                N3.append('O'+str(s2)+str(s3))
        elif t==4:
            for s3 in scenarios:
                for s4 in scenarios:
                    N4.append('O'+str(s2)+str(s3)+str(s4))
        elif t==5:
            for s3 in scenarios:
                for s4 in scenarios:
                    for s5 in scenarios:
                        N5.append('O'+str(s2)+str(s3)+str(s4)+str(s5))
N.extend(N1)
N.extend(N2)
N.extend(N3)
N.extend(N4)
N.extend(N5)
NP.extend(N1)
NP.extend(N2)
NP.extend(N3)
NP.extend(N4)

N_stage = collections.OrderedDict()
N_stage[1] = N1
N_stage[2] = N2
N_stage[3] = N3
N_stage[4] = N4
N_stage[5] = N5

#get weights based on childen nodes
single_prob = {}
single_prob['L'] = 0.25
single_prob['R'] = 0.5
single_prob['H'] = 0.25

prob = collections.OrderedDict()
for n in N1:
    prob[n]=1
for n in N2:
    m = n[1:2]
    prob[n] = single_prob[m]
for n in N3:
    m = n[1:2]
    o = n[2:3]
    prob[n] = single_prob[m]*single_prob[o]
for n in N4:
    m = n[1:2]
    o = n[2:3]
    l = n[3:4]
    prob[n] = single_prob[m]*single_prob[o]*single_prob[l]
for n in N5:
    m = n[1:2]
    o = n[2:3]
    l = n[3:4]
    q = n[4:5]
    prob[n] = single_prob[m]*single_prob[o]*single_prob[l]*single_prob[q]

#add nodes to the tree
G.add_nodes_from(N)

#add edges to the tree
for n in N1:
    G.add_edge('F', n)
    for n2 in N2:
        G.add_edge(n, n2)
for n2 in N2:
    for n3 in N3:
        first_2_letters_n2 = n2[:2]
        first_2_letters_n3 = n3[:2]
        if first_2_letters_n2 == first_2_letters_n3:
            G.add_edge(n2, n3)
for n3 in N3:
    for n4 in N4:
        first_3_letters_n3 = n3[:3]
        first_3_letters_n4 = n4[:3]
        if first_3_letters_n3 == first_3_letters_n4:
            G.add_edge(n3, n4)
for n4 in N4:
    for n5 in N5:
        first_4_letters_n4 = n4[:4]
        first_4_letters_n5 = n5[:4]
        if first_4_letters_n4 == first_4_letters_n5:
            G.add_edge(n4, n5)

E = G.edges()

PN = collections.OrderedDict() #parental nodes
for n in N:
    for n_ in N:
        if n_ in G.adj[n]:
            PN[n_] = [n]

# CN = collections.OrderedDict() #children nodes
# for n_ in N:
#     if n_ not in N_stage[5]:
#         CN_list = []
#         for n in G.adj[n_]:
#             CN_list.extend([n])
#         CN[n_] = CN_list
