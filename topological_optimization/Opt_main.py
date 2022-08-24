from cmath import exp
from copy import deepcopy
from cmath import exp
from random import random

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from networkx import circular_layout, spectral_layout, shell_layout, spring_layout

from Out2Csv import Out2Csv


class Opt_main:
    @staticmethod
    def getEigenvalue(matrix):
        eigenvalue, _ = np.linalg.eig(matrix)
        return eigenvalue

    @staticmethod
    def f_value(matrix):
        eigenvalue = Opt_main.getEigenvalue(matrix)
        eigenvalue.sort()
        # print(len(eigenvalue))
        # print(eigenvalue[1], eigenvalue[-1])
        # return eigenvalue[-1]
        return eigenvalue[1] / eigenvalue[-1]

    @staticmethod
    def adj2degree(adjacencyMatrix):
        degreeList = []
        for neighbors in adjacencyMatrix:
            degree = 0
            for neighbor in neighbors:
                degree += neighbor
            degreeList.append(degree)

        return np.diag(degreeList)

    def __init__(self, nodeNum, degree):
        self.nodeNum = nodeNum
        self.degree = degree
        self.err = -1
        self.times = 10000
        adjacencyList = []
        # self.edgeSet = set(())
        self.edgeSet = []
        for i in range(nodeNum):
            currentNeighbors = [0] * nodeNum
            for j in range(degree):
                currentNeighbors[(i + j + 1) % nodeNum] = 1
                newEdge = [i, (i + j + 1) % nodeNum]
                newEdge.sort()
                if newEdge not in self.edgeSet:
                    self.edgeSet.append(deepcopy(newEdge))

            adjacencyList.append(currentNeighbors)

        print(self.edgeSet)
        self.adjacencyMatrix = np.array(adjacencyList)
        self.adjacencyMatrix += self.adjacencyMatrix.T
        print(self.adjacencyMatrix)

        self.degreeMatrix = Opt_main.adj2degree(self.adjacencyMatrix)

        print(self.degreeMatrix)

        self.L = self.degreeMatrix - self.adjacencyMatrix
        # print(self.L)

        self.out2Csv = Out2Csv("f_value.csv")

    def rand(self):
        """
        get a random graph
        :return:
        """
        newEdgeSet = []
        newAdjacencyList = []

        for i in range(self.nodeNum):
            currentNeighbors = [0] * self.nodeNum
            currentRemainDegree = self.degree

            sampleList = list(range(self.nodeNum))
            sampleList.remove(i)
            selectedList = random.sample(sampleList, self.degree * 2)
            print(selectedList)
            for nodeId in selectedList:
                currentNeighbors[nodeId] = 1
                currentRemainDegree -= 1
                newEdge = [i, nodeId]
                if newEdge.sort() not in newEdgeSet:
                    newEdgeSet.append(deepcopy(newEdge))

            newAdjacencyList.append(currentNeighbors)
            self.edgeSet = newEdgeSet

        self.adjacencyMatrix = np.array(newAdjacencyList)

        pass

    def updateMatrices(self, adjacencyMatrix):
        self.adjacencyMatrix = adjacencyMatrix
        self.degreeMatrix = Opt_main.adj2degree(adjacencyMatrix)
        self.L = self.degreeMatrix - self.adjacencyMatrix
        # print(self.L)

    def changeEdge(self, edge, newEdge):
        # print("remove:{},{};add:{},{}".format(edge[0], edge[1], newEdge[0], newEdge[1]))
        if list(newEdge).sort() in self.edgeSet:
            print("error: not new edge")

        self.adjacencyMatrix[edge[0]][edge[1]] = 0
        self.adjacencyMatrix[edge[1]][edge[0]] = 0
        self.adjacencyMatrix[newEdge[0]][newEdge[1]] = 1
        self.adjacencyMatrix[newEdge[1]][newEdge[0]] = 1
        self.updateMatrices(self.adjacencyMatrix)
        # self.draw()

    def clique(self):
        adjacencyList = []
        for i in range(self.nodeNum):
            currentNeighbor = [1] * self.nodeNum
            currentNeighbor[i] = 0
            adjacencyList.append(currentNeighbor)

        return adjacencyList

    def dep(self):
        adjacencyList = []
        for i in range(self.nodeNum):
            currentNeighbor = [0] * self.nodeNum
            adjacencyList.append(currentNeighbor)

        return adjacencyList

    def draw(self):
        graph = nx.Graph()
        for i in range(self.nodeNum):
            graph.add_node(i)

        for nodeStart in range(self.nodeNum):
            for nodeEnd in range(nodeStart + 1, self.nodeNum):
                if self.adjacencyMatrix[nodeStart][nodeEnd] == 1:
                    graph.add_edge(nodeStart, nodeEnd)

        nx.draw(graph, pos=circular_layout(graph), with_labels=True)
        plt.show()

    def isNeighbor(self, new_v, V):
        for v in V:
            if v is new_v:
                return True

            if self.adjacencyMatrix[v][new_v] == 1:
                return True

        return False

    def opt(self):
        for epoch in range(self.times):
            e_t = exp(self.err * (epoch + 1)).real
            # e_t = 0
            print(e_t)
            # selectedNode = epoch % self.nodeNum
            current_f_value = self.f_value(self.L)
            self.out2Csv.addPoint(str(current_f_value))
            print("----current f: {}------".format(current_f_value))
            print(self.L)
            max_delta = 0
            newEdge = (-1, -1)
            sourceEdge = (-1, -1)
            changeEdge = (0, 0)

            bAllNotChange = True
            for i in range(len(self.edgeSet)):
                edge = self.edgeSet[i]
                for j in range(i + 1, len(self.edgeSet)):
                    edge1 = self.edgeSet[j]
                    if not self.isNeighbor(edge1[0], edge) and not self.isNeighbor(edge1[1], edge):
                        self.changeEdge(tuple(edge), (edge[0], edge1[0]))
                        self.changeEdge(tuple(edge1), (edge[1], edge1[1]))

                        temp_f_value = self.f_value(self.L)
                        # print("temp_f_value: {}------".format(temp_f_value))
                        delta_f = temp_f_value - current_f_value
                        if delta_f + e_t > max_delta:
                            changeEdge = (i, j)
                            max_delta = delta_f

                        self.changeEdge((edge[0], edge1[0]), tuple(edge))
                        self.changeEdge((edge[1], edge1[1]), tuple(edge1))
                        # self.draw()

                # for nodeStart in range(self.nodeNum):
                #     for nodeEnd in range(nodeStart + 1, self.nodeNum):
                #         if self.adjacencyMatrix[nodeStart][nodeEnd] == 0:
                #
                #             self.changeEdge(tuple(edge), (nodeStart, nodeEnd))
                #
                #             temp_f_value = self.f_value(self.L)
                #             # print("temp_f_value: {}------".format(temp_f_value))
                #             delta_f = current_f_value - temp_f_value
                #             if delta_f + e_t > max_delta:
                #                 max_delta = delta_f
                #                 newEdge = (nodeStart, nodeEnd)
                #                 sourceEdge = (edge[0], edge[1])
                #
                #             self.changeEdge((nodeStart, nodeEnd), tuple(edge))

            if max_delta != 0:
                bAllNotChange = False

                # self.changeEdge(sourceEdge, newEdge)
                # self.edgeSet.remove(list(sourceEdge))
                # self.edgeSet.append(list(newEdge))

                edge = deepcopy(self.edgeSet[changeEdge[0]])
                edge1 = deepcopy(self.edgeSet[changeEdge[1]])

                print("remove:{}, add:{}".format(edge, (edge[0], edge1[0])))
                print("remove:{}, add:{}".format(edge1, (edge[1], edge1[1])))
                self.changeEdge(tuple(edge), (edge[0], edge1[0]))
                self.changeEdge(tuple(edge1), (edge[1], edge1[1]))
                # self.draw()
                self.edgeSet[changeEdge[0]][1] = edge1[0]
                self.edgeSet[changeEdge[0]].sort()
                print("new edge:" + str(self.edgeSet[changeEdge[0]]))
                self.edgeSet[changeEdge[1]][0] = edge[1]
                self.edgeSet[changeEdge[1]].sort()
                print("new edge:" + str(self.edgeSet[changeEdge[1]]))
                print("max:" + str(max_delta))

            if bAllNotChange:
                print("end!")
                break


if __name__ == '__main__':
    opt_main = Opt_main(10, 2)
    opt_main.draw()
    opt_main.rand()
    opt_main.draw()
    opt_main.opt()
    print(opt_main.L)
    opt_main.draw()
