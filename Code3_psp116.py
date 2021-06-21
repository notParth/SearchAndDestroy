import numpy as np
import random as rand
import matplotlib
import matplotlib.pyplot as plt

# makes a map of a given size with a target randomly set
def make_map(size):
    map = np.full([size, size], -1)
    # setting cells to random terrains
    # Terrain types:
    # 1 = flat
    # 2 = hilly
    # 3 = forested
    # 4 = maze of caves 
    for i in range(size):
        for j in range(size):
            p = rand.random()
            if p <= 0.25:
                map[i][j] = 1
            elif p <= 0.5:
                map[i][j] = 2
            elif p <= 0.75:
                map[i][j] = 3
            else:
                map[i][j] = 4
    return map


# setting the target
# target is represented as 10 + *type of terrain*
def set_target(map, size, terrain_type):
    target_x = rand.randint(0, size-1)
    target_y = rand.randint(0, size-1)
    if terrain_type != 0:
        while(map[target_x][target_y] != terrain_type):
            target_x = rand.randint(0, size-1)
            target_y = rand.randint(0, size-1)
    map[target_x][target_y] += 10

# used by agent to query cell
def query(map, coordinates):
    x = coordinates[0]
    y = coordinates[1]
    cell = map[x][y]
    if cell > 10:
        cell_type = cell - 10
        p = rand.random()
        if cell_type == 1:
            if p > 0.1:
                return True
        elif cell_type == 2:
            if p > 0.3:
                return True
        elif cell_type == 3:
            if p > 0.7:
                return True
        else:
            if p > 0.9:
                return True
    return False

# print the map
# FL = FLAT
# HI = HILLs
# FO = FORESTED
# CA = CAVES
def print_map(map, size):
    print('-'*(size*5), end="")
    print()
    for x in map:
        for y in x:
            if y > 10:
                y =- 10
            if y == 1:
                cell = 'FL'
            elif y == 2:
                cell = 'HI'
            elif y == 3:
                cell = 'FO'
            else:
                cell = 'CA'
            print(f'| {cell} ', end="")
        print("|")
        print('-'*(size*5))

# manhattan distance
def manh_dist(point_one, point_two):
    return abs(point_one[0] - point_two[0]) + abs(point_two[1] - point_one[1])

# used by agents to choose a cell to query
# finds the cell with the maximum probability
# if there are ties with prob, it chooses based on manhattan distance from agent's location
# if there are ties with dist, it chooses randomly
def find_max_cell(belief, size, agent_location):
    max = 0
    possible_query_candidates = []
    for i in belief:
        for j in i:
            if j > max:
                max = j
    for i in range(size):
        for j in range(size):
            if belief[i][j] == max:
                possible_query_candidates.append((i, j))
    if len(possible_query_candidates) == 1:
        return possible_query_candidates[0]
    else:
        shortest_dist = 500
        for i in possible_query_candidates:
            d = manh_dist(i, agent_location)
            if d < shortest_dist:
                shortest_dist = d
        final_candidates = []
        for i in possible_query_candidates:
            d = manh_dist(i, agent_location)
            if d == shortest_dist:
                final_candidates.append(i)
        if len(final_candidates) == 1:
            return final_candidates[0]
        else:
            return final_candidates[rand.randint(0, len(final_candidates)-1)]

# method to return what a cell's terrain type is
# used by agent to know what a cell's type is
def cell_type(map, coordinates):
    x = coordinates[0]
    y = coordinates[1]
    if map[x][y] > 10:
        return map[x][y] - 10
    return map[x][y]

# returns the FNR for the particular terrain
# used by the agent
def FNR(map, cell):
    type = cell_type(map, cell)
    if type == 1:
        return 0.1
    elif type == 2:
        return 0.3
    elif type == 3:
        return 0.7
    else:
        return 0.9

# agent one uses rule one
def agent_one(map, size):
    score = 0
    initial_belief = 1 / (size**2)
    belief = np.full([size, size], initial_belief)
    agent_location = (rand.randint(0, size-1), rand.randint(0, size-1))
    while True:
        # cell to be queried cell
        query_cell = find_max_cell(belief, size, agent_location)
        # travel to the query cell
        score += manh_dist(agent_location, query_cell)
        agent_location = query_cell
        score += 1
        if query(map, query_cell):
            return score
        else:
            query_x = query_cell[0]
            query_y = query_cell[1]
            FNR_query = FNR(map, query_cell)
            B_query = belief[query_x][query_y]
            denominator = ((FNR_query*B_query) + (1-B_query))
            for i in range(size):
                for j in range(size):
                    if i == query_x and j == query_y:
                        belief[i][j] = (belief[i][j]*FNR_query)/denominator
                    else:
                        belief[i][j] = (belief[i][j])/denominator

# agent two uses rule two
def agent_two(map, size):
    score = 0
    initial_belief = 1 / (size**2)
    belief = np.full([size, size], initial_belief)
    contains_prob = np.full([size, size], initial_belief)
    agent_location = (rand.randint(0, size-1), rand.randint(0, size-1))
    while True:
        # cell to be queried cell
        query_cell = find_max_cell(contains_prob, size, agent_location)
        # travel to the query cell
        score += manh_dist(agent_location, query_cell)
        agent_location = query_cell
        score += 1
        if query(map, query_cell):
            return score
        else:
            query_x = query_cell[0]
            query_y = query_cell[1]
            FNR_query = FNR(map, query_cell)
            B_query = belief[query_x][query_y]
            denominator = ((FNR_query*B_query) + (1-B_query))
            for i in range(size):
                for j in range(size):
                    if i == query_x and j == query_y:
                        belief[i][j] = (belief[i][j]*FNR_query)/denominator
                    else:
                        belief[i][j] = (belief[i][j])/denominator
                    contains_prob[i][j] = belief[i][j] * (1 - FNR(map, (i, j)))

# improved agent
def improved_agent(map, size):
    score = 0
    initial_belief = 1 / (size**2)
    belief = np.full([size, size], initial_belief)
    contains_prob = np.full([size, size], initial_belief)
    agent_location = (rand.randint(0, size-1), rand.randint(0, size-1))
    while True:
        # cell to be queried cell
        query_cell = find_max_cell(belief, size, agent_location)
        # travel to the query cell
        score += manh_dist(agent_location, query_cell)
        agent_location = query_cell
        query_cell_type = cell_type(map, query_cell)
        # we search multiple times depending on the query cell's terrain type
        # my representation of terrain types allows for this: number of searches = 2 * cell terrain type
        loop = query_cell_type ** 2
        for ite in range(loop):
            score += 1
            if query(map, query_cell):
                return score
            else:
                query_x = query_cell[0]
                query_y = query_cell[1]
                FNR_query = FNR(map, query_cell)
                B_query = belief[query_x][query_y]
                denominator = ((FNR_query*B_query) + (1-B_query))
                for i in range(size):
                    for j in range(size):
                        if i == query_x and j == query_y:
                            belief[i][j] = (belief[i][j]*FNR_query)/denominator
                        else:
                            belief[i][j] = (belief[i][j])/denominator
                        contains_prob[i][j] = belief[i][j] * (1 - FNR(map, (i, j)))


size = 50
data_one = []
data_two = []
data_three = []
runs = 25
labels = ['Random', 'Flats', 'Hills', 'Forested', 'Caves']

for i in range(0, 5):
    print(i)
    sum_one = 0
    sum_two = 0
    sum_three = 0
    for run in range(runs):
        print(run)
        map = make_map(size)
        set_target(map, size, i)
        sum_three += improved_agent(map, size)
        sum_two += agent_two(map, size)
        sum_one += agent_one(map, size)
    data_one.append(sum_one/runs)
    data_two.append(sum_two/runs)
    data_three.append(sum_three/runs)
print(data_one)
print(data_two)
print(data_three)
width = 0.3
labels = ['Standard', 'Flats', 'Hills', 'Forested', 'Caves']
plt.xticks(range(len(data_one)), labels)
plt.xlabel('Target cell terrain')
plt.ylabel('Score')
plt.title('Agent performance comparision')
plt.bar(np.arange(len(data_one)), data_one, width=width, label='Basic agent one')
plt.bar(np.arange(len(data_two))+ width, data_two, width=width, label='Basic agent two')
plt.bar(np.arange(len(data_three))+ width*2, data_three, width=width, label='Improved agent')
plt.legend()
plt.show()