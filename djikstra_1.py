def popmin(pqueue):
    # A (ascending or min) priority queue keeps element with
    # lowest priority on top. So pop function pops out the element with
    # lowest value. It can be implemented as sorted or unsorted array
    # (dictionary in this case) or as a tree (lowest priority element is
    # root of tree)
    lowest = 1000
    keylowest = None
    for key in pqueue:
        if pqueue[key] < lowest:
            lowest = pqueue[key]
            keylowest = key
    del pqueue[keylowest]
    return keylowest
 
def dijkstra(graph, start):
    # Using priority queue to keep track of minium distance from start
    # to a vertex.
    pqueue = {} # vertex: distance to start
    dist = {}   # vertex: distance to start
    pred = {}   # vertex: previous (predecesor) vertex in shortest path
 
    # initializing dictionaries
    for v in graph:
        dist[v] = 1000
        pred[v] = -1
    dist[start] = 0
    for v in graph:
        pqueue[v] = dist[v] # equivalent to push into queue
 
    while pqueue:
        u = popmin(pqueue) # for priority queues, pop will get the element with smallest value
        for v in graph[u].keys(): # for each neighbor of u
            w = graph[u][v] # distance u to v
            newdist = dist[u] + w
            if (newdist < dist[v]): # is new distance shorter than one in dist?
                # found new shorter distance. save it
                pqueue[v] = newdist
                dist[v] = newdist
                pred[v] = u
 
    return dist, pred
 
graph = {0 : {1:6, 2:8},
1 : {4:11},
2 : {3: 9},
3 : {},
4 : {5:3},
5 : {2: 7, 3:4}}
 
dist, pred = dijkstra(graph, 0)
print "Predecesors in shortest path:"
for v in pred: print "%s: %s" % (v, pred[v])
print "Shortest distance from each vertex:"
for v in dist: print "%s: %s" % (v, dist[v])