#!/usr/bin/python

from Queue import PriorityQueue
import rospy

def isValidNeighboor(y_coord, x_coord, h, w):
    return (y_coord >= 0 and y_coord < h-1 and x_coord >= 0 and x_coord < w-1)

def getNeighbors(current_node, frame_height, frame_width):
    list_of_neighbors = []
    helper_explorer = [(-1,0), (+1,0), (-1,+1), (0,+1), (+1,+1)] # left, right, lower left, lower center, lower right
    y_coord = current_node[0]
    x_coord = current_node[1]

    # validate each of the 5 possible neihbors
    # TODO : Only need to validate on the limit columns to be more efficient
    for h in helper_explorer:
        t_neigh = (y_coord + h[1], x_coord + h[0])
        if isValidNeighboor(t_neigh[0], t_neigh[1], frame_height, frame_width):
            list_of_neighbors.append(t_neigh)
    
    return list_of_neighbors



def dynamicDijkstraRoadDetection(weightedSumMatrix, vp_x):

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Entered dynamicDijkstraRoadDetection") 
    
    frame_height, frame_width = weightedSumMatrix.shape
    
    # we will stop procesing once we hit the lowest row of the weightedSumMatrix
    Y_GOAL_COORD = frame_height - 1
    
    # create frontier queue for exploring
    frontier = PriorityQueue()

    # list to track back the path taken
    path_to_node = {}
    path_to_node[(0, vp_x)] = None

    # dictionary to track cost to each node
    # initial node has cost of 0
    current_cost_to_node = {}
    current_cost_to_node[(0, vp_x)] = 0

    # push the first element, we must start from vp_y
    frontier.put((0, vp_x))

    while not frontier.empty():
        current_node = frontier.get()
        # print "current node :", current_node[0], current_node[1]
        # if current node Y coordinate is goal we are done
        if current_node[0] == Y_GOAL_COORD:
            print ("we made it boys")
            break
        
        for neigh in getNeighbors(current_node, frame_height, frame_width):
            y_neigh, x_neigh = neigh
            new_cost = current_cost_to_node[current_node] + weightedSumMatrix[y_neigh,x_neigh]
            
            # check if cost should be updated
            if neigh not in current_cost_to_node or new_cost < current_cost_to_node[neigh]:
                # update cost
                current_cost_to_node[neigh] = new_cost
                frontier.put(neigh,new_cost)

                # update path 
                path_to_node[neigh] = current_node
    
    current = current_node 
    path = []
    while current != (0, vp_x): 
       path.append(current)
       current = path_to_node[current]

    path.append((0, vp_x)) # optional
    path.reverse() # optional

    # print "Path found \n", path

    # ------------------------ ROS LOG ---------------------- #
    rospy.logdebug("Exiting dynamicDijkstraRoadDetection") 
    
    return path



# ---------------------------------- DEPRECIATED ---------------------------------- # 
# # Crate graph to apply disjtra over
    # costGraph = Graph(frame_height*frame_width*4)
    
    # # only making nodes below VP
    # nodeIDMatrix = np.empty(shape=depth_frame.shape, dtype=np.uint32)
    # # p(x,y) -> nodesID can be done by
    # for y in range(vp_y,frame_height):
    #     temp = np.arange(frame_width)
    #     nodeIDMatrix[y,:] = temp + (y-vp_y)*frame_width
    # # print "NodeIDMatrix ", nodeIDMatrix
    # # print "VP ID :", nodeIDMatrix[vp_y, vp_x]

    # # normalizedWeightsMatrix = np.interp(weightedSumMatrix, (weightedSumMatrix.min(), weightedSumMatrix.max()), (0, 255))
    # # cv2.imshow('normalized weight', normalizedWeightsMatrix)
    
    # # create links to apply dikstra
    # for y in range (vp_y, frame_height-1):
    #     for x in range (1,frame_width-1):
    # # for y in range (vp_y, vp_y+10):
    # # for y in range (vp_y, vp_y+50):
    # #     for x in range (1,frame_width-1):

    #         # Add weights for neighbors
    #         curNode = nodeIDMatrix[y,x]

    #         # left
    #         leftNeigh = nodeIDMatrix[y,x-1]
    #         costGraph.addEdge(curNode, leftNeigh, weightedSumMatrix[y,x-1])
            
    #         # # right
    #         # rightNeigh = nodeIDMatrix[y,x+1]
    #         # costGraph.addEdge(curNode, rightNeigh, weightedSumMatrix[y,x+1])
            
    #         # lower left
    #         lowerLeftNeigh = nodeIDMatrix[y+1,x-1]
    #         costGraph.addEdge(curNode, lowerLeftNeigh, weightedSumMatrix[y+1,x-1])
            
    #         # down
    #         downNeigh = nodeIDMatrix[y+1,x]
    #         costGraph.addEdge(curNode, downNeigh, weightedSumMatrix[y+1,x])
            
    #         # lower right
    #         lowerRightNeigh = nodeIDMatrix[y+1,x+1]
    #         costGraph.addEdge(curNode, lowerRightNeigh, weightedSumMatrix[y+1,x+1])
    
    # distances  = np.asarray(costGraph.dijkstra(nodeIDMatrix[vp_y, vp_x]), dtype=np.uint64)
    # # print ("Distances: \n", distances)
    
    
    # # normalizedDistances = np.interp(distances, (distances.min(), distances.max()), (0, 255))
    # # print(normalizedDistances)

    # visualiseShortestsCosts = np.zeros(shape=(frame_height, frame_width), dtype=np.uint8)
    # ## fill matrix
    # # for y in range (vp_y, vp_y+10):
    # for y in range (vp_y, vp_y+50):
    #     for x in range (1,frame_width-1):
    #         nodeID = (y-vp_y)*frame_width + x
    #         distToNode = distances[nodeID]
    #         distToNode = np.interp(distToNode, (0, 20000 ), (0, 255))
    #         visualiseShortestsCosts[y,x] = distToNode
    #         # print ("Point {},{} has id {} --- dist: {}".format(y,x,nodeID,distToNode))
    
    # cv2.imshow('Distances', visualiseShortestsCosts)
    