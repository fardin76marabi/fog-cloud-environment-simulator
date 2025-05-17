# Import libraries
import matplotlib.pyplot as plt
import math
import random
import sys
import time
import numpy as np
import copy
from copy import deepcopy
from numpy.random import choice
from numpy import random


### Global Variables
myFile=open('system-report.txt', 'w') #open the file and write


cloud=-1
listOfTasks=[]
#listOfDoneTasks=[]
listOfColonies=[]
listOfNodes=[]
listOfManagers=[]
step_distance_matrix={} # storing step between nodes in the colony
managersBW_TransDelay = {}
managersCost_ResponseTime = {} 
taskIndex=0
colonyIndex=0
nodeIndex=0
subSlot=200 #ms
timeSlot=1000 #ms
coSlot=int(timeSlot/subSlot)
Coefficient=subSlot/timeSlot # timeSlot was 1000 before


max_bw=100 #Mbps
min_bw=10
#avg_bw=(max_bw+min_bw)/2
avg_bw=1000 #Mbps

max_propDelay=3 #ms
min_propDelay=1
avg_propDelay=(max_propDelay+min_propDelay)/2



####flags


#New1 Proposed 
new1Time=0

#New2 Proposed 
new2Time=0

#New3 Proposed 
new3Time=0

#SRR
simpleRRTurnF="critical"

#WRR
weigthedRRTurnF="critical"
weigthedRRUnit=100 ## equals to least cpu unit posiible
criticalUnit=weigthedRRUnit ## base for critical
aNormal=5  ## step size of increase in normal task
normalUnit=aNormal*weigthedRRUnit ##base for normal

#MQP
m=0 #keep the consequtives critical that gone in a row
M=2 #max of consequtive in a row
P=30 #for DynamicTimeSlot function
C=1000 #cpu border between critical and normal task

#MQP2
m2=0 #keep the consequtives critical that gone in a row
M2=2 #max of consequtive in a row
P2=30 #for DynamicTimeSlot function
C2=1000 #cpu border between critical and normal task


#Proposed Algorithm for scheduling
freeCpuThreshold=20 ## the threshold for cpu execution model 


# proposed algorithm cpu execution model based on slack

slackThr=100 #ms
slackOfTheTask=0
slackEnabeled=False
slackTime=-1

# proposed algorithm2 for case Y

slackThr2=100 #ms
slackOfTheTask2=0
slackEnabeled2=False
slackTime2=-1

# critical First Execution Alg
CFTime=-1


# global variable 
totalSlots=0

class IotDevice:
    def __init__(self,id):
        self.id=id


class Task:
    def __init__(self, id, cpu,src,insize,outsize,deadline,mem,type,generationTime):
        self.id= id
        self.cpu = cpu
        self.doneCPU=0
        self.insize = insize
        self.outsize = outsize
        self.deadline = deadline
        self.mem= mem
        self.type = type
        self.slackTime=-1 ####Newly added
        self.generationTime = generationTime
        self.schedulingTime=sys.maxsize
        self.schedulingDelay=sys.maxsize
        self.arrivalTime=sys.maxsize
        self.finishTime = sys.maxsize
        self.communicationDelay=sys.maxsize
        self.queuingDelay=sys.maxsize
        self.executionDelay=sys.maxsize
        self.responseTime=sys.maxsize
        self.scheduled=False
        self.finished=False
        self.rejected=False
        self.dst=-1                 # for routing
        self.next=-1                # for routing    
        self.src=src            # for routing
        self.rout=[]                # for routing

class Cloud:
    def __init__(self,id,name,cpu,mem):
        self.name=name
        self.id=id
        self.cpu=cpu
        self.mem=mem
        self.activeTime=0
        self.buffer=[]
        self.queueOfNormals=[]
        self.queueOfCriticals=[]
        self.queueOfDoneTasks=[]
        self.innerTable= {}    ##for later
        self.outterTable= {}   #for later   
        self.time=0
        self.criticalRun=0
        self.normalRun=0
        self.freeCpu=cpu
        self.normalCapacity=cpu # initial value is equal to cpu power

            
class Node:
    def __init__(self, name, id,parent,cpu,mem,energyActive,energyIdle,arrivalRateCritical,arrivalRateNormal):
        self.name = name
        self.id = id
        self.parent=parent
        self.cpu= cpu
        self.mem = mem
        self.buffer = []
        self.listOfTasks=[]
        self.time=0 ## for execution control
        self.energyActive = energyActive
        self.energyIdle = energyIdle
        self.activeTime=0
        self.arrivalRateCritical = arrivalRateCritical
        self.arrivalRateNormal = arrivalRateNormal
        self.queueOfNormals=[]
        self.queueOfCriticals=[]
        self.queueOfDoneTasks=[]
        self.innerTable= {}    #Manager Only         #Table of status of nodes in colony
        self.outterTable= {}   #Manager Only         #Table of status of all Managers
        self.criticalRun=0
        self.normalRun=0
        self.freeCpu=cpu
        self.normalCapacity=cpu
    def create_tasks(self,slot):
        global taskIndex
        cpu=mem=insize=outsize=deadline=generationTime=-1 
        src=self
        type=""
        newtasks=[]
        generationTime=-1 #initial value, changed based on generated random number for each task
        for x in range(self.arrivalRateCritical):##Critical tasks creation ##xx
            generationTime=random.randint(slot*timeSlot, ((slot+1)*timeSlot)-1) ## xx XX before was time subSlot
            cpu=int(random.normal(loc=400, scale=20, size=(1, 1)))  ##*MI change to million struction name size or numofinstructions
            mem=math.ceil(random.randint(128, 512)) #MB  ##step 64
            insize=math.ceil(random.randint(100, 10240)) #KB
            outsize=(math.ceil(random.randint(10, 50))/100)*(insize) # 10-50% of InSize
            deadline=int(random.normal(loc=200, scale=20, size=(1, 1)))  #ms #xx XX 1 or Coefficient 
            newtask=Task(taskIndex,cpu,src,insize,outsize,deadline,mem,"critical",generationTime)
            #self.buffer.append(newtask)
            newtasks.append(newtask)
            #listOfTasks.append(newtask) ##general list
            taskIndex=taskIndex+1
        for x in range( self.arrivalRateNormal):##Normal tasks creation
            generationTime=random.randint(slot*timeSlot, ((slot+1)*timeSlot)-1) ## xx XX before was time subSlot
            cpu=int(random.normal(loc=5000, scale=200, size=(1, 1))) 
            mem=math.ceil(random.randint(256, 1024)) ##step 64
            insize=math.ceil(random.randint(100, 10240)) #KB
            outsize=(math.ceil(random.randint(10, 50))/100)*(insize) # 10-50% of InSize
            deadline=int(random.normal(loc=2000, scale=100, size=(1, 1)))  #ms #xx XX 1 or Coefficient 
            newtask=Task(taskIndex,cpu,src,insize,outsize,deadline,mem,"normal",generationTime)
            #self.buffer.append(newtask) ##store in the buffer of the generative node
            #listOfTasks.append(newtask) ##general list
            newtasks.append(newtask)
            taskIndex=taskIndex+1
        (newtasks).sort(key=lambda task: task.generationTime) ##sort task based on generation Time
        return newtasks
    

class Colony:
    def __init__(self):
        global colonyIndex
        self.id=colonyIndex
        colonyIndex=colonyIndex+1
        self.routingTable= {}     ### ??? most be updated in self.update_routingTable
        self.listOfNodes=[]
        self.listOfComputingNodes=[]  ### list of nodes that can exe a task, including cloud
        self.nodeManager=-1
        numberOfNodes=random.randint(4, 8) ##number of nodes in each colony (3, 6) ##xx
        self.numOfNodes=numberOfNodes
        self.adjancyMatrix={}            ##Adjancy matrix of the colony
        self.matrixBW_Propagation={}     ## matrix BW and Pro Delay between nodes of a colony
        #self.step_distance_matrix={} ## for paper: store the distances between nodes in a colony

        self.create_nodes()         ## creating nodes in each colony
        self.create_connections()   ## creating connections in each colony
        self.check_connectivity()   ## no seperate node in colony
        self.createBW_Propagation()  
        self.select_manager()       ## selecting a node as manager in RANDOMLY WAY
        
        self.create_innerTable(self.nodeManager)
        #print("*****BEFORE****")
        #self.show_connection()
        #self.show_connection()
        self.update_routingTable() ### newly added for updating routing
        #print("*****AFTER****")
        #self.show_connection()
        
        #self.create_distanceMatrix(): ## store the number of step between each two nodes in colony
            #return

    def createBW_Propagation(self): #assign a random value for BW and Prop Delay of connected nodes
        for i in self.listOfNodes: ##initial value to a large number
            for j in self.listOfNodes:
                self.matrixBW_Propagation[(i.name,j.name)]=0,sys.maxsize
        for i in self.listOfNodes:
            for j in self.listOfNodes:
                if i==j:
                    continue
                if self.adjancyMatrix[(i.name,j.name)]==1: ## assign a value to the two selected node that are neighbor in adj matrix
                    if self.matrixBW_Propagation[(i.name,j.name)][0]==0: ## check wheater not been assigned before
                        bw=math.ceil(random.randint(min_bw,max_bw)) #Mbps ##xx 1000 or Coefficient or ??
                        propDelay=math.ceil(random.randint(min_propDelay,max_propDelay)) #ms #xx
                        self.matrixBW_Propagation[(i.name,j.name)]=bw,propDelay
                        self.matrixBW_Propagation[(j.name,i.name)]=bw,propDelay


    def update_routingTable(self): ## Creating routingTable ## need to be changed **********
        #print("***************************************")
        for node in self.listOfNodes:#extract adjanciesMatrix out of 2D Tuplelist of Colony adjMatrix
            dijkstra(self,self.adjancyMatrix, node,self.listOfNodes)## dist to other nodes and routs be saved where???
        #print("***************************************")
        

    def create_nodes(self): ## creating nodes in each colony #xx
        global nodeIndex
        for x in range(self.numOfNodes):
            id=nodeIndex
            name="Node"+(str(id))
            parent=self
            cpu=math.ceil(random.randint(4000, 16000)) ##MIPS
            mem=math.ceil(random.randint(4096, 16384))  ###MB step=512
            energyActive=math.ceil(random.randint(200, 600)) ##watt
            energyIdle=math.ceil((random.randint(5, 7)/10)) ##relate to active in[50-70]% *energyActive
            arrivalRateCritical=math.ceil(random.randint(2, 10)*Coefficient) ##tasksperslot (1, 5)
            arrivalRateNormal=math.ceil(random.randint(1, 4)*Coefficient)   ## (1, 5)
            node=Node(name,id,parent,cpu,mem,energyActive,energyIdle,arrivalRateCritical,arrivalRateNormal)
            self.listOfNodes.append(node) #colony list of nodes
            listOfNodes.append(node) #general list of nodes  ##newly added
            nodeIndex=nodeIndex+1
    def show_connection(self):
        print("Adjancey matrix of Colony :",self.id)
        for i in self.listOfNodes:
            for j in self.listOfNodes:
                print(self.adjancyMatrix[(i,j)],end=' ')
            print()
                
    def create_connections(self): ## creating connections in each colony ????
        p=20 ## probability of relation between two nodes
        for i in self.listOfNodes:
            for j in self.listOfNodes:
                x=random.randint(0, 101)
                if i==j:
                    self.adjancyMatrix[(i.name,j.name)]=0
                    #self.neighbor[i][j]=0
                    continue
                if (x<p):
                    self.adjancyMatrix[(i.name,j.name)]=1
                    self.adjancyMatrix[(j.name,i.name)]=1
                else:
                    self.adjancyMatrix[(i.name,j.name)]=0
                    self.adjancyMatrix[(j.name,i.name)]=0

    def check_connectivity(self):
        visited=[]
        traverse(self.listOfNodes[0],visited,self.listOfNodes,self.adjancyMatrix) ##start node
        while len(visited) != self.numOfNodes:
            visited.clear()
            n1=0
            n2=0
            while n1==n2:
                n1=random.randint(0,len(self.listOfNodes)-1)
                n2=random.randint(0,len(self.listOfNodes)-1)
            node1=self.listOfNodes[n1]
            node2=self.listOfNodes[n2]
            self.adjancyMatrix[(node1.name,node2.name)]=1
            self.adjancyMatrix[(node2.name,node1.name)]=1
            traverse(self.listOfNodes[0],visited,self.listOfNodes,self.adjancyMatrix)


    def select_manager(self):
        idOfmanager=random.randint(0, len(self.listOfNodes)-1)
        self.nodeManager=self.listOfNodes[idOfmanager]
        listOfManagers.append(self.nodeManager)

    def create_innerTable(self,manager):    ##initializing the table status of inner colony nodes
        for node in self.listOfNodes:
            if node==manager: ## If manager not to put it in the INNER TABLE
                continue
            manager.innerTable[node,"cpu"]=node.cpu
            manager.innerTable[node,"mem"]=node.mem
            manager.innerTable[node,"energyActive"]=node.energyActive
            manager.innerTable[node,"energyIdle"]=node.energyIdle
            manager.innerTable[node,"queueOfNormals"]=node.queueOfNormals
            manager.innerTable[node,"queueOfCriticals"]=node.queueOfCriticals
            manager.innerTable[node,"freeCapacity"]=node.cpu  ##newly added
            manager.innerTable[node,"normalCpuUsed"]=0  ##newly added
            manager.innerTable[node,"criticalCpuUsed"]=0  ##newly added
            manager.innerTable[node,"normalCapacity"]=node.cpu ## ?? xx XX


class FogCloud:
    def __init__(self):
        global cloud ## cloud is share between different case ?? XX xx
        cloud=Cloud(0,"cloud",int(22000),int(20480)) ##id name cpu ram
        listOfManagers.append(cloud)
        numberOfColonies=random.randint(1,2) ##(1,3) 
        create_colony(numberOfColonies)
        findComputingNodes(listOfColonies,cloud) ##find computing node of each Fog Cloud test case that can be chosen as a dst including cloud
        
        connect_mannagers()
        
        start_simmulation()


def start_simmulation():

    
    ## different cases:

    #case new1: new1 Proposed Algorithm (for cpu allc:critical first and sort normals in queu based on min deadline remained
    #+ sch: sort normals based on min deadline remained in manager's buffer + calculate free cpu of last subslot of each node and sort with highest one + map first task to first node
    caseNew1listOfColonies=deepcopy(listOfColonies)
    caseNew1listOfManagers=[]

    #case new2: new2 Proposed Algorithm (for cpu allc:same as new1
    #+ sch: same as new1 + a task is sent to a node if that node can finish the selected task with maximum 20% deadline violation
    caseNew2listOfColonies=deepcopy(listOfColonies)
    caseNew2listOfManagers=[]

    #case new3: new3 Proposed Algorithm 
    #+ sch: same as new1 + cpu allocation: total algo is like CF but among normals we consider a combinational factor of sjf & remained deadline
    caseNew3listOfColonies=deepcopy(listOfColonies)
    caseNew3listOfManagers=[]



    #case 0: Proposed Algorithm (slack cpu allocation + using last slot info for scheduling)
    case0listOfColonies=deepcopy(listOfColonies)
    case0listOfManagers=[]

    #case x: MQP and Propoesed scheduling
    caseXlistOfColonies=deepcopy(listOfColonies)
    caseXlistOfManagers=[]

    #case y: Proposed CPU allocation and Local scheduling
    caseYlistOfColonies=deepcopy(listOfColonies)
    caseYlistOfManagers=[]

    #case 1: with MQP execution model
    case1listOfColonies=deepcopy(listOfColonies)
    case1listOfManagers=[]

    #case 2 : the critical First Executing Algorithm
    case2listOfColonies=deepcopy(listOfColonies)
    case2listOfManagers=[]

    #case 3 : the Simple Round Robin Executing Algorithm
    case3listOfColonies=deepcopy(listOfColonies)
    case3listOfManagers=[]

    #case 4 : the Weighted Round Robin Executing Algorithm
    case4listOfColonies=deepcopy(listOfColonies)
    case4listOfManagers=[]

    #add each colony case for each different senario
    listOfColoniesTestCases=[]
    
    #listOfColoniesTestCases.append(caseNew1listOfColonies)
    listOfColoniesTestCases.append(caseNew2listOfColonies)
    #listOfColoniesTestCases.append(caseNew3listOfColonies)
    #listOfColoniesTestCases.append(case0listOfColonies)
    #listOfColoniesTestCases.append(caseXlistOfColonies)
    #listOfColoniesTestCases.append(caseYlistOfColonies)
    #listOfColoniesTestCases.append(case1listOfColonies)
    #listOfColoniesTestCases.append(case2listOfColonies)
    #listOfColoniesTestCases.append(case3listOfColonies)
    #listOfColoniesTestCases.append(case4listOfColonies)
    
    
    #list of result checker 
    #listOfDoneTasksCaseNew1=[]
    listOfDoneTasksCaseNew2=[]
    #listOfDoneTasksCaseNew3=[]
    #listOfDoneTasksCase0=[]
    #listOfDoneTasksCaseX=[]
    #listOfDoneTasksCaseY=[]
    #listOfDoneTasksCase1=[]
    #listOfDoneTasksCase2=[]
    #listOfDoneTasksCase3=[]
    #listOfDoneTasksCase4=[]

    # copying process
    BW_TransDelayNew1 = {}
    Cost_ResponseTimeNew1 = {}

    BW_TransDelayNew2 = {}
    Cost_ResponseTimeNew2 = {}

    BW_TransDelayNew3 = {}
    Cost_ResponseTimeNew3 = {}

    BW_TransDelay0 = {}
    Cost_ResponseTime0 = {}
    BW_TransDelayX = {}
    Cost_ResponseTimeX = {}
    BW_TransDelayY = {}
    Cost_ResponseTimeY = {}
    BW_TransDelay1 = {}
    Cost_ResponseTime1 = {}
    BW_TransDelay2 = {}
    Cost_ResponseTime2 = {}
    BW_TransDelay3 = {}
    Cost_ResponseTime3 = {}
    BW_TransDelay4 = {}
    Cost_ResponseTime4 = {}


    #managers_copy(caseNew1listOfManagers,caseNew1listOfColonies)
    managers_copy(caseNew2listOfManagers,caseNew2listOfColonies)
    #managers_copy(caseNew3listOfManagers,caseNew3listOfColonies)

    #managers_copy(case0listOfManagers,case0listOfColonies)
    #managers_copy(caseXlistOfManagers,caseXlistOfColonies)
    #managers_copy(caseYlistOfManagers,caseYlistOfColonies)
    #managers_copy(case1listOfManagers,case1listOfColonies)
    #managers_copy(case2listOfManagers,case2listOfColonies)
    #managers_copy(case3listOfManagers,case3listOfColonies)
    #managers_copy(case4listOfManagers,case4listOfColonies)
    
    global totalSlots 
    totalSlots=100 ## Number of time slots =10
    slot=0
    i =0 ## phase 2 controller
    
    print("----------------Start Simulation-----------")
    #for slot in range(totalSlots):
    while ((slot<totalSlots) | remainedTasks(caseNew1listOfColonies) | remainedTasks(case2listOfColonies)| (remainedTasks(case0listOfColonies)) | remainedTasks(caseXlistOfColonies) | remainedTasks(caseYlistOfColonies) |  remainedTasks(case1listOfColonies)| remainedTasks(case2listOfColonies) | remainedTasks(case3listOfColonies)| remainedTasks(case4listOfColonies)):
        # 
        print("---------------------------------------------------- ",slot,file=myFile)
        print("Beggening of slot ",slot,file=myFile)
        print("we are in ",slot*subSlot,file=myFile)
        print("",file=myFile)
        #print("Beggening of slot ",slot)
        #print("we are in ",slot*subSlot)
        #time.sleep(4)
        #phase 1 =managerScheduling
        #if slot %coSlot==0: ## update the table five time slot once
            #updateNodeOutterTable(case0listOfManagers,case0listOfColonies) ## just for proposed alg
            #updateNodeOutterTable(caseXlistOfManagers,caseXlistOfColonies) ## for test case only
        # 3 above lines is for multi colony

        resetUsedCPU(caseNew1listOfColonies) #reset cpu used in last subslot
        resetUsedCPU(caseNew2listOfColonies) #reset cpu used in last subslot
        resetUsedCPU(caseNew3listOfColonies) #reset cpu used in last subslot


        #managerSchedulingPhase("New1NormalProposedAlgorithm",New1NormalProposedAlgorithm,caseNew1listOfManagers,caseNew1listOfColonies,slot,BW_TransDelayNew1,Cost_ResponseTimeNew1 )###New1 Proposed alg ....
        managerSchedulingPhase("New2NormalProposedAlgorithm",New2NormalProposedAlgorithm,caseNew2listOfManagers,caseNew2listOfColonies,slot,BW_TransDelayNew2,Cost_ResponseTimeNew2 )###New2 Proposed alg ....        
        #managerSchedulingPhase("New3NormalProposedAlgorithm",New3NormalProposedAlgorithm,caseNew3listOfManagers,caseNew3listOfColonies,slot,BW_TransDelayNew3,Cost_ResponseTimeNew3 )###New3 Proposed alg ....        
        
        #managerSchedulingPhase("NormalProposedAlgorithm",NormalProposedAlgorithm,case0listOfManagers,case0listOfColonies,slot,BW_TransDelay0,Cost_ResponseTime0 )###Proposed alg ....
        #managerSchedulingPhase("NormalProposedAlgorithm2",NormalProposedAlgorithm2,caseXlistOfManagers,caseXlistOfColonies,slot,BW_TransDelayX,Cost_ResponseTimeX) ## check variable in proposed al in second time use here
        #managerSchedulingPhase("NormalAlgorithmLocal",NormalAlgorithmLocal,caseYlistOfManagers,caseYlistOfColonies,slot,BW_TransDelayY,Cost_ResponseTimeY) ## here too
        #managerSchedulingPhase("NormalAlgorithmLocal",NormalAlgorithmLocal,case1listOfManagers,case1listOfColonies,slot,BW_TransDelay1,Cost_ResponseTime1)
        #managerSchedulingPhase("NormalRandomAlgorithmNeighbor",NormalRandomAlgorithmNeighbor,case2listOfManagers,case2listOfColonies,slot,BW_TransDelay2,Cost_ResponseTime2)
        #managerSchedulingPhase("NormalRandomAlgorithmGlobal",NormalRandomAlgorithmGlobal,case3listOfManagers,case3listOfColonies,slot,BW_TransDelay3,Cost_ResponseTime3)
        #managerSchedulingPhase("NormalRandomAlgorithmGlobal",NormalRandomAlgorithmGlobal,case4listOfManagers,case4listOfColonies,slot,BW_TransDelay4,Cost_ResponseTime4)


        #phase2:(task creation + schedule N tasks to manager and C to the generator node)
        print("i --creating task test ",i,file=myFile)
        #time.sleep(3)
        if slot%coSlot==0: ## here timeslot is 1000 subSlot is 200 and co is 5
            #if slot>=totalSlots: ## no creating task at last slot 
                ## we add > rather = becasue we wanna continue the process foe end of simulation too
                #print("craeting tasks ",i%coSlot==0,file=myFile)
                #continue
            #xx XX ?? before we had above mention, but got replaced with a below if statement
            # the below conditon must be checked 
            if slot<=totalSlots-1:
                print("i mod coSlot==0 ",i%coSlot==0,file=myFile)
                #time.sleep(3)
                slotForTaskGenTimeMaking=int(slot/coSlot)
                taskCreatingPhase(listOfColoniesTestCases,slotForTaskGenTimeMaking) ###

        #innerNodePhase("New1NormalProposedAlgorithm",caseNew1listOfColonies,slot,BW_TransDelayNew1,Cost_ResponseTimeNew1,New1NormalProposedAlgorithm,nodeCriticalScheduling) ##New1Proposed alg ....
        innerNodePhase("New2NormalProposedAlgorithm",caseNew2listOfColonies,slot,BW_TransDelayNew2,Cost_ResponseTimeNew2,New2NormalProposedAlgorithm,nodeCriticalSchedulingProposed) ##New2Proposed alg ....
        #innerNodePhase("New3NormalProposedAlgorithm",caseNew3listOfColonies,slot,BW_TransDelayNew3,Cost_ResponseTimeNew3,New3NormalProposedAlgorithm,nodeCriticalScheduling) ##New2Proposed alg ....
        
        #innerNodePhase("NormalProposedAlgorithm",case0listOfColonies,slot,BW_TransDelay0,Cost_ResponseTime0,NormalProposedAlgorithm)###Proposed alg ....
        #innerNodePhase("NormalProposedAlgorithm",caseXlistOfColonies,slot,BW_TransDelayX,Cost_ResponseTimeX,NormalProposedAlgorithm2)
        #innerNodePhase("NormalAlgorithmLocal",caseYlistOfColonies,slot,BW_TransDelayY,Cost_ResponseTimeY,NormalAlgorithmLocal)
        #innerNodePhase("NormalAlgorithmLocal",case1listOfColonies,slot,BW_TransDelay1,Cost_ResponseTime1,NormalAlgorithmLocal)
        #innerNodePhase("NormalRandomAlgorithmNeighbor",case2listOfColonies,slot,BW_TransDelay2,Cost_ResponseTime2,NormalRandomAlgorithmNeighbor)
        #innerNodePhase("NormalRandomAlgorithmGlobal",case3listOfColonies,slot,BW_TransDelay3,Cost_ResponseTime3,NormalRandomAlgorithmGlobal)
        #innerNodePhase("NormalRandomAlgorithmGlobal",case4listOfColonies,slot,BW_TransDelay4,Cost_ResponseTime4,NormalRandomAlgorithmGlobal)
        

        #phase 3 = execution ## its time to nodes to execute tasks that arrived to queues of node   
        #executionPhase("New1proposedExecutingAlgorithm",New1proposedExecutingAlgorithm,caseNew1listOfColonies,listOfDoneTasksCaseNew1,True,slot) ###New1 Proposed alg ....
        executionPhase("New2proposedExecutingAlgorithm",New2proposedExecutingAlgorithm,caseNew2listOfColonies,listOfDoneTasksCaseNew2,True,slot) ###New2 Proposed alg ....
        #executionPhase("New3proposedExecutingAlgorithm",New3proposedExecutingAlgorithm,caseNew3listOfColonies,listOfDoneTasksCaseNew3,True,slot) ###New3 Proposed alg ....

        #executionPhase("proposedExecutingAlgorithm",proposedExecutingAlgorithm,case0listOfColonies,listOfDoneTasksCase0,True,slot) ###Proposed alg ....
        #executionPhase("MQPExecutingAlgorithm",MQPExecutingAlgorithm,caseXlistOfColonies,listOfDoneTasksCaseX,True,slot)  
        #executionPhase("proposedExecutingAlgorithm2",proposedExecutingAlgorithm2,caseYlistOfColonies,listOfDoneTasksCaseY,True,slot)  
        #executionPhase("MQPExecutingAlgorithm2",MQPExecutingAlgorithm2,case1listOfColonies,listOfDoneTasksCase1,True,slot)       
        #executionPhase("criticalFirstExecutingAlgorithm",criticalFirstExecutingAlgorithm,case2listOfColonies,listOfDoneTasksCase2,True,slot)
        #executionPhase("simpleRoundRobinExecutingAlgorithm",simpleRoundRobinExecutingAlgorithm,case3listOfColonies,listOfDoneTasksCase3,False,slot)
        #executionPhase("weightedRoundRobinExecutingAlgorithm",weightedRoundRobinExecutingAlgorithm,case4listOfColonies,listOfDoneTasksCase4,False,slot)

        ## controlling the loop
        slot+=1

    myFile.close() #close the file and write    


    #showing result of different cases
    #show_test(listOfDoneTasksCase0) # check src dst and response time of task in each test case
    #show_test(listOfDoneTasksCase2)
    #time.sleep(10000)
    print("*************** Result Checking ********************")
    #show_results("Case 1 ",case1listOfColonies)
    #print("////////*********")
    #show_results("Case 2 ",case2listOfColonies)
    listOfDones=[]
    names=[]
    
    #listOfDones.append(listOfDoneTasksCaseNew1)
    #names.append("New1 Proposed")

    listOfDones.append(listOfDoneTasksCaseNew2)
    names.append("New2 Proposed")
    
    #listOfDones.append(listOfDoneTasksCaseNew2)
    #names.append("New3 Proposed")

    #listOfDones.append(listOfDoneTasksCase0)
    #names.append("Propossed")
    #listOfDones.append(listOfDoneTasksCaseX)
    #names.append("X")
    #listOfDones.append(listOfDoneTasksCaseY)
    #names.append("Y")
    #listOfDones.append(listOfDoneTasksCase1)
    #names.append("MQP")
    #listOfDones.append(listOfDoneTasksCase2)
    #names.append("Critical First")
    #listOfDones.append(listOfDoneTasksCase3)
    #names.append("SRR")
    #listOfDones.append(listOfDoneTasksCase4)
    #names.append("WRR")


    ##showing in graphical way
    ##?? comparison of un done tasks of eah method
    #print("percentage of done tasks in proposed alg")
    #print(len(listOfDoneTasksCase0)/taskIndex)
    #print("percentage of done tasks in MQP alg")
    #print(len(listOfDoneTasksCase1)/taskIndex)
    #print("percentage of done tasks in CF alg")
    #print(len(listOfDoneTasksCase2)/taskIndex)
    #time.sleep(5)

    show_violation(listOfDones,names)
    show_responseTime(listOfDones,names)
    show_cpuUtilization2(listOfColoniesTestCases,names) 
    show_deadlineSatisfied2(listOfColoniesTestCases,names)
    show_throughput(listOfColoniesTestCases,names)    


def show_violation(listOfDones,names): # show violation of normal tasks in each scenario
    print("********** show_violation")
    resultsNormals=[]
    
    x1=[]
    for i, listOfDone in enumerate(listOfDones):
        print("---------------------------------------")
        print()
        resNormals=[]
        
        for task in listOfDone:
            if task.finished!=True:
                continue
            if task.type=="normal":
                violation=max(0,task.responseTime - (task.generationTime+task.deadline))
                resNormals.append (violation)
                                  
            print("selected task ",task.id)
            print("type ",task.type)
            #print("violation ",violation)
            #print("src ",(task.src).name)
            #print("dst ",(task.dst).name)
            #print("Resp ",task.responseTime)
        print("len of done normals ",len(resNormals))
        resultsNormals.append(deepcopy(max(resNormals))) 
        x1.append(i+1) 
        resNormals.clear()

    print("names ",names)
    print("x1 ",x1)
    print("resultsNormals ",resultsNormals)

    #box_plot(resultsNormals,names,x1,"Violation","Normal Tasks")


def remainedTasks(LOC): # check all nodes.buffer in each colonies: if all tasks has finished return true
    for node in (LOC[0].listOfComputingNodes):
        if(len(node.buffer)!=0):
            return True   
        if(len(node.queueOfNormals)!=0):
            return True 
        if(len(node.queueOfCriticals)!=0):
            return True 
        
    return False

def show_throughput(listOfColoniesTestCases,names):
    #result=[]
    resultNormals=[]
    resultCriticals=[]
    x1=[]
    for i,test in enumerate(listOfColoniesTestCases):
        #res=[]
        resNormals=[]
        resCriticals=[]
        for node in test[0].listOfComputingNodes:
            if node.time==0: ##error handling for when not all test cases have value
                continue
            if "cloud" in node.name:
                continue
            if node.name==((node.parent).nodeManager).name:#manager
                continue
            #totalDone=0
            totalNormalsDone=0
            totalCriticalsDone=0
            for task in node.listOfTasks:
                if ((task.finished==True) ): ## & (task.responseTime<=task.deadline)
                    if (task.type=="normal"):
                        totalNormalsDone+=1
                    else :
                        totalCriticalsDone+=1
            #res.append(int((totalDone/node.time)*1000)) ## ?? xx XX
            #res.append(round((totalDone/node.time)*timeSlot,2)) ## ?? xx XX
            resNormals.append(round((totalNormalsDone/node.time)*timeSlot,2))
            resCriticals.append(round((totalCriticalsDone/node.time)*timeSlot,2))
        #result.append(deepcopy(res))
        resultNormals.append(deepcopy(resNormals))
        resultCriticals.append(deepcopy(resCriticals))
        x1.append(i+1)
        #res.clear()
        resCriticals.clear()
        resNormals.clear()
    #box_plot(result,names,x1,"","Throughput(s)")
    box_plot(resultNormals,names,x1,"Throughput (task/timeslot) ","Normal Tasks")      
    box_plot(resultCriticals,names,x1,"Throughput (task/timeslot) ","Criticals Tasks") 

def show_deadlineSatisfied2(listOfColoniesTestCases,names):
    #result=[]
    resultNormals=[]
    resultCriticals=[]
    x1=[]
    for i,test in enumerate(listOfColoniesTestCases):
        print("---------changing test cases --------------")
        #res=[]
        resNormals=[]
        resCriticals=[]
        for node in test[0].listOfComputingNodes:
            print("------------changing node----------")
            print("node is ",node.name)
            if node.time==0: ##error handling for when not all test cases have value
                continue
            if "cloud" in node.name:
                continue
            if node.name==((node.parent).nodeManager).name:#manager
                continue
            #totalDone=0
            totalNormalsDone=0
            totalCriticalsDone=0
            #allCreatedTasks=0
            allCreatedTasksNormals=0
            allCreatedTasksCriticals=0
            for task in node.listOfTasks:
                #print("task is ",task.id)
                #print("type is ",task.type)
                #print("deadline is ",task.deadline)
                #print("resp is ",task.responseTime)
                if task.type=="normal":
                    allCreatedTasksNormals+=1
                    if ((task.finished==True) & (task.responseTime<=task.deadline)):
                        totalNormalsDone+=1
                        #resNormals.append((task.responseTime))
                else:
                    allCreatedTasksCriticals+=1
                    if ((task.finished==True) & (task.responseTime<=task.deadline)):
                        totalCriticalsDone+=1
                        #resCriticals.append((task.responseTime))
            
            #res.append(int((totalDone/allCreatedTasks)*100))
            resNormals.append(((totalNormalsDone/allCreatedTasksNormals)*100))
            resCriticals.append(((totalCriticalsDone/allCreatedTasksCriticals)*100))
            #print("node is ",node.name)
            #print("totalCriticalsDone ",totalCriticalsDone)
            #print("allCreatedTasksCriticals ",allCreatedTasksCriticals)
            print("done normal percentage ",((totalNormalsDone/allCreatedTasksNormals)*100))
            print("done critical percentage ",((totalCriticalsDone/allCreatedTasksCriticals)*100))
        #result.append(deepcopy(res))
        resultNormals.append(deepcopy(resNormals))
        resultCriticals.append(deepcopy(resCriticals))
        x1.append(i+1)
        #res.clear()
        resCriticals.clear()
        resNormals.clear()
    #box_plot(result,names,x1,"","Deadline Satisfied")  
    box_plot(resultNormals,names,x1,"Deadline Satisfiction","Normal Tasks")  
    box_plot(resultCriticals,names,x1,"Deadline Satisfiction","Criticals Tasks")  

def show_deadlineSatisfied(listOfColoniesTestCases,names):
    result=[]
    x1=[]
    for i,test in enumerate(listOfColoniesTestCases):
        res=[]
        #totalDone=0
        #allCreatedTasks=0
        for c in test:
            for node in c.listOfNodes:
                if node.name==((node.parent).nodeManager).name:
                    continue
                totalDone=0
                allCreatedTasks=0
                allCreatedTasks=(len(node.listOfTasks))
                for task in node.listOfTasks:
                    if ((task.finished==True) & (task.responseTime<=task.deadline)):
                        totalDone+=1
                res.append(int((totalDone/allCreatedTasks)*100)) 
        result.append(deepcopy(res))
        x1.append(i+1)
        res.clear()
    box_plot(result,names,x1,"","Deadline Satisfied")                     


def show_cpuUtilization2(listOfColoniesTestCases,names):
    result=[]
    x1=[]
    for i,test in enumerate(listOfColoniesTestCases):
        res=[]
        for node in test[0].listOfComputingNodes:
            if node.time==0: ##error handling for when not all test cases have value
                continue
            if "cloud" in node.name: ## ?? xx XX because we dont use cloud so we pass it 
                res.append(int((node.activeTime/(node.time))*100)) # paper : instead of below line
                #continue
            elif node.name==(((node.parent).nodeManager).name):
                continue
            else:
                res.append(int((node.activeTime/(node.time))*100))
        result.append(deepcopy(res))
        x1.append(i+1)
        res.clear()
    box_plot(result,names,x1,"","Cpu Utilization") 
    
def show_cpuUtilization(listOfColoniesTestCases,names): ## without cloud
    result=[]
    x1=[]
    for i,test in enumerate(listOfColoniesTestCases):
        res=[]
        for c in test:
            for node in c.listOfNodes:
                if node.name==(((node.parent).nodeManager).name):
                    continue
                res.append(int((node.activeTime/(node.time))*100))
        result.append(deepcopy(res))
        x1.append(i+1)
        res.clear()
    box_plot(result,names,x1,"","Cpu Utilization") 


def show_responseTime(listOfDones,names):
    print("**********RESP TIME OF TASKS")
    resultsNormals=[]
    resultsCriticals=[]
    x1=[]
    for i, listOfDone in enumerate(listOfDones):
        print("---------------------------------------")
        print()
        resNormals=[]
        resCriticals=[]
        for task in listOfDone:
            if task.finished!=True:
                continue
            if task.type=="normal":
                resNormals.append((task.responseTime))
            else:
                resCriticals.append((task.responseTime))
            print("selected task ",task.id)
            print("type ",task.type)
            #print("src ",(task.src).name)
            #print("dst ",(task.dst).name)
            #print("Resp ",task.responseTime)
        print("len of done normals ",len(resNormals))
        print("len of done criticals ",len(resCriticals))
        resultsNormals.append(deepcopy(resNormals))
        resultsCriticals.append(deepcopy(resCriticals))
        x1.append(i+1) 
        resCriticals.clear()
        resNormals.clear()
    box_plot(resultsNormals,names,x1,"Response Time","Normal Tasks")
    box_plot(resultsCriticals,names,x1,"Response Time","Criticals Tasks")


def show_test(testList):
    print("*-*-*-*-*--**-*-*-*")
    print(len(testList))
    total=0
    for task in testList:
        print("test is",end=" ")
        print(task.id,end=" src is ")
        print((task.src).id,end=" dst is ")
        print((task.dst).name,end=" resp is ")
        print(task.responseTime)
        total+=(task.responseTime)
    print("avg resp is ",total/(len(testList)))


def managers_copy(mList,lOC):
    mList.append((listOfManagers[0])) ##cloud
    for c in lOC:
        for n in c.listOfNodes:
            if n==c.nodeManager:
                mList.append(n)


def findComputingNodes(LOC,cloud): ## fill List Of Computing nodes for each colony
    listOfComputingNodes=[]
    listOfComputingNodes.append(cloud) ##adding cloud to computing nodes list
    for c in LOC:
        for node in c.listOfNodes:
            if node.name==(c.nodeManager).name:
                continue
            listOfComputingNodes.append(node)
    
    for c in LOC: ##the nodes in all colonies can be dst
        for computingNode in listOfComputingNodes:
            (c.listOfComputingNodes).append(computingNode)



def managerSchedulingPhase(name,NormalScheduling,listOfMan,colonyList,slot,BW_TransDelay1,Cost_ResponseTime1):
    print("managerSchedulingPhase",file=myFile)
    print("manager scheduling is based on ",name,file=myFile)
    print("",file=myFile)
    #time.sleep(2)
    managersList=[] ##list of managers that have task for schdule
    for manager in listOfMan:###here check managers for scheduling
        print("manager is ",manager.name,file=myFile)
        print("len of buffer is ",len(manager.buffer),file=myFile)
        if len(manager.buffer)!=0: ## if manager has task to schedule
            managersList.append(manager) 
    if len(managersList)!=0: #be task in manager buffer to schedule
        managerNormalSchduling(NormalScheduling,managersList,colonyList,slot,BW_TransDelay1,Cost_ResponseTime1) #set scheduled of task to True
    managersList.clear()
    #time.sleep(7)


def innerNodePhase(name,listOfC,slot,BW_TransDelay1,Cost_ResponseTime1,NormalSchdeulingAlg,CriticalSchedulingAlg):
    scheduled=[] ## list for critical scheduled task
    end=(slot+1)*subSlot
    now=slot*subSlot
    nowInSlotLook=int(slot/coSlot)*timeSlot
    print("Node Inner Phase",file=myFile)
    print("scheduling is based on ",name,file=myFile)
    #time.sleep(2)
    for colony in listOfC:#scheduling of tasks in the buffer of generator/src node and create task for each node
        for node in colony.listOfNodes:
            if node==colony.nodeManager:##cheching manager
                continue
            print("tasks In buffer ...",file=myFile)
            #time.sleep(2)
            for task in node.buffer:
                print("",file=myFile)
                #print("now ",now,file=myFile)
                print("now (node.time)",node.time,file=myFile)
                print("end ",end,file=myFile)
                print("task is ",task.id,file=myFile)
                print("task src ",(task.src).id,file=myFile)
                print("task type ",task.type,file=myFile)
                print("task generation ",task.generationTime,file=myFile)
                print("task.scheduled ",task.scheduled,file=myFile)
                print("task.generationTime<=now ",task.generationTime<=end,file=myFile)
                print("task.scheduled==False ",task.scheduled==False,file=myFile)
                print("(task.generationTime<=end)&(task.scheduled==False) ",(task.generationTime<=end)&(task.scheduled==False),file=myFile)
                #time.sleep(1)
                if ((task.generationTime<=end)&(task.scheduled==False)):
                    if task.type=="normal":
                        if ((NormalSchdeulingAlg==NormalProposedAlgorithm) | (NormalSchdeulingAlg==NormalProposedAlgorithm2)): #only for our propoesd alg
                            print("In proposed alg inner node. checking the condition for src as dst",file=myFile)
                            dst=task.src
                            manager=colony.nodeManager
                            # --> we comment change in Now because in forward func it will be managed based on task type in task type
                            #?? but I think this handling is not okay, because normal tasks in our proposed alg and others have a different scheduling time 
                            ##checkTheCondition(task,dst,manager,now,nowInSlotLook)
                            
                            ## below are belong to before; now we send the normal tasks to manager 
                            #checkTheCondition(task,dst,manager,task.generationTime,nowInSlotLook)
                            #if task.dst==-1: # when the task can not run at src we must add it to manager buffer
                                #print(name," in inner node process. but the task can not be assigend to src we have to send it to manager for scheduling in the next slot",file=myFile)
                                #((colony.nodeManager).buffer).append(task)
                            ((colony.nodeManager).buffer).append(task) ## replace of above commands

                        elif (NormalSchdeulingAlg==NormalAlgorithmLocal): ## in this alg, we do not need the manager to decide, we just add it to generator node for process
                            task.dst=NormalAlgorithmLocal(task,colony.nodeManager,listOfC,slot) ## check wheater it works or not
                            forward(task,task.generationTime,{},{}) ## before forward(task,now,{},{})
                            print("task assigend to its src",file=myFile)
                            print("total tasks (C & N) for executing soon",len(node.queueOfNormals+node.queueOfCriticals),file=myFile)
                        else: # for other algs. add the normal task to manager buffer for scheduling
                            ((colony.nodeManager).buffer).append(task)
                            print("N & len of manager's buffer after",len((colony.nodeManager).buffer),file=myFile)
                    elif task.type=="critical":
                        CriticalSchedulingAlg(task,node,now,BW_TransDelay1,Cost_ResponseTime1)
                        print("C & len of src's buffer after",len(node.queueOfCriticals),file=myFile)
                    #time.sleep(8)
                    scheduled.append(task)  
            for task in scheduled:#critical tasks goes to queueOfCriticals in thier node imidiately so must be removed from node buffer
                (node.buffer).remove(task)
            scheduled.clear()


def create_colony(num):
    for x in range(num): ##create colonies and their nodes
        listOfColonies.append(Colony())

def report():
    for colony in listOfColonies:#showing colonies and nodes in them and their properties and Tasks
        print("**********************************************")
        print("Colony Id is : ",colony.id)
        print("Number of Nodes: ",len(colony.listOfNodes))
        print("Manager is :",(colony.nodeManager).name)
        for node in colony.listOfNodes:
            print("Node Id is :",node.id)
            print("cpu ",node.cpu)
            print("mem ",node.mem)
            print("energyActive ",node.energyActive)
            print("energyIdle ",node.energyIdle)
            print("arrivalRateCritical ",node.arrivalRateCritical)
            print("arrivalRateNormal ",node.arrivalRateNormal)
            ####  printing a random task in each node buffer   #####
            if len(node.buffer)!=0:##show task if there is a task to show in the buffer
                print("showing a random task of this node ")
                randomTaskIndex=random.randint(0,(len(node.buffer))-1)
                randomTask=node.buffer[randomTaskIndex]
                print("Task id ",randomTask.id)
                print("cpu  ",randomTask.cpu)
                print("startTime  ",randomTask.startTime)
        colony.show_connection() ## Showing the Adjancy Matrix Of the Colony
        print("Routing Table of the Colony")
        for i in colony.listOfNodes:
            for j in colony.listOfNodes:
                if i==j:
                    continue
                print(i.id,end=" -> ")
                print(j.id,end=" : ")
                for n in colony.routingTable[(i,j)]:
                    print(n.id,end=" ")
                print()
        print("showing the BW and Propagation Delay betwwen nodes in the Colony")
        for i in colony.listOfNodes:
            for j in colony.listOfNodes:
                print(i.id,end=" -> ")
                print(j.id,end=" : ")
                print(colony.matrixBW_Propagation[(i,j)])
    print("showing the BW and Delay between the Managers")
    for i in range(len(listOfManagers)): ##showing the BW and Delay between the Managers
        for j in range(i+1,len(listOfManagers)):
            print(listOfManagers[i].name+" -> ",listOfManagers[j].name,end=": ")
            print(managersBW_TransDelay[(listOfManagers[j],listOfManagers[i])][0],end=", ")
            print(managersBW_TransDelay[(listOfManagers[j],listOfManagers[i])][1])
    print("showing the Cost and Average Response Time between the Managers")
    for i in range(len(listOfManagers)):#showing the Cost and Average Response Time between the Managers
        for j in range(i+1,len(listOfManagers)):
            print(listOfManagers[i].name+" -> ",listOfManagers[j].name,end=": ")
            print(managersCost_ResponseTime[(listOfManagers[j],listOfManagers[i])][0],end=", ")
            print(managersCost_ResponseTime[(listOfManagers[j],listOfManagers[i])][1])
    print("INNER TABLE") ##showing the INNER TABLE OF EACH MANAGER OF THE COLONIES(Here Just CPU)
    for colony in listOfColonies:
        manager=colony.nodeManager
        print("Colony is ",end=":")
        print(colony.id)
        print("Manager is ",end=":")
        print(manager.name,end=" Cpu of users in the colony are:")
        print()
        for node in colony.listOfNodes:
            if node==manager:
                continue
            print(node.name,end=", cpu is : ")
            print(manager.innerTable[node,"cpu"])

def connect_mannagers(): # create a tuple for every 2 managers with BW and Delay in Value
    global managersBW_TransDelay
    global managersCost_ResponseTime
    for i in range(len(listOfManagers)):
        for j in range(i+1,len(listOfManagers)):
            BW=random.randint(100,1000) #Mbps # create a BW for every two managers
            Delay=random.randint(2,10) #ms #create a Delay for every two managers
            Cost=random.randint(1,3) # create a Cost for every two managers
            Response=0               # create a Average Response time for every two managers
            managersBW_TransDelay[(listOfManagers[i].name,listOfManagers[j].name)]=BW,Delay
            managersBW_TransDelay[(listOfManagers[j].name,listOfManagers[i].name)]=BW,Delay
            managersCost_ResponseTime[(listOfManagers[i].name,listOfManagers[j].name)]=Cost,Response
            managersCost_ResponseTime[(listOfManagers[j].name,listOfManagers[i].name)]=Cost,Response
            
NO_PARENT = -1
def dijkstra(self,adjacency_matrix, node,listOfNodes): ##full adj matrix is here now
    
    n_vertices = len(listOfNodes)
    #print("n_vertices ",n_vertices)
    start_vertex=node
    #print("start_vertex ",start_vertex)

    added ={}
    shortest_distances={}
    parents={}
    for vertex in listOfNodes:
        shortest_distances[vertex] = sys.maxsize
        added[vertex] = False
        parents[vertex]=-1

    shortest_distances[start_vertex] = 0
    parents[start_vertex] = NO_PARENT

    for i in range(1, n_vertices):
        nearest_vertex = -1
        shortest_distance = sys.maxsize
        for vertex in listOfNodes:
            if not added[vertex] and shortest_distances[vertex] < shortest_distance:
                nearest_vertex = vertex
                shortest_distance = shortest_distances[vertex]

        added[nearest_vertex] = True
        for vertex in listOfNodes: 
            edge_distance = adjacency_matrix[(nearest_vertex.name,vertex.name)]
            if edge_distance > 0 and shortest_distance + edge_distance < shortest_distances[vertex]:
                parents[vertex] = nearest_vertex
                shortest_distances[vertex] = shortest_distance + edge_distance
    
    print_solution(self,start_vertex, shortest_distances, parents,listOfNodes)# list of node para is added recently
    



# A utility function to print
# the constructed distances
# array and shortest paths
def print_solution(self,start_vertex, distances, parents,listOfnodes):
    n_vertices = len(distances)
    routeList=[] ## for saving the route from start to other nodes
    #global step_distance_matrix

    #print("Vertex\t\t Distance\tPath")
    for vertex in listOfnodes:
        if vertex != start_vertex:
            #print("", start_vertex.id, "->", vertex.id, "\t", distances[vertex], "\t\t", end="")
            print_path(self,vertex, parents,routeList)
            #print("\n Current list is ",end=" : ")
            #for n in routeList:
                #print(n.id,end=" ")
            self.routingTable[(start_vertex.name,vertex.name)]=routeList[1:len(routeList)]
            routeList.clear()
            # papar: save the step between nodes in a matrix
            #step_distance_matrix[(start_vertex,vertex)]=len(self.routingTable[(start_vertex.name,vertex.name)])-1
            #step_distance_matrix[(vertex,start_vertex)]=len(self.routingTable[(start_vertex.name,vertex.name)])-1
            

# Function to print shortest path
# from source to current_vertex
# using parents array
def print_path(self,current_vertex, parents,routeList):
    if current_vertex == NO_PARENT:
        return
    print_path(self,parents[current_vertex], parents,routeList)
    #print(current_vertex.id, end=" ")
    routeList.append(current_vertex.name)


def traverse(node,visited,listOfNodes,neighbor):
        if node not in visited:
            visited.append(node)
            for i in listOfNodes:
                if neighbor[(node.name,i.name)]==1:
                    traverse(i,visited,listOfNodes,neighbor)

def taskCreatingPhase(LOTC,slot): # create same task for each node in each colony of given test cases
    print("In task creation phase ",file=myFile)
    #time.sleep(2)
    for colony in listOfColonies:
        for node in colony.listOfNodes:
            if node==colony.nodeManager:
                continue
            newTasks=node.create_tasks(slot)
            for test in LOTC:
                copyNewTasks=deepcopy(newTasks)
                for c in test:
                    if c.id==colony.id:
                        for n in c.listOfNodes:
                            if n.id==node.id:
                                #print("Test case ",test)
                                #print("node.id ",node.id)
                                #print("colony.id ",colony.id)
                                #print("c.id ",c.id)
                                #print("n.id ",n.id)
                                #time.sleep(4)
                                for task in copyNewTasks:
                                    #print("--------------------------")
                                    #print("task.id ",task.id)
                                    #print("Type ",task.type)
                                    #print("task.gen ",task.generationTime)
                                    #print("task.deadline ",task.deadline)
                                    #print("task.cpu ",task.cpu)
                                    #print("task.src ",n.id)
                                    #time.sleep(18)
                                    task.src=n
                                    (n.listOfTasks).append(task)# a source of all generated tasks in each colony to checking results
                                    (n.buffer).append(task)
                copyNewTasks.clear()
            newTasks.clear()
            

def executionPhase(name,executionStrategy,colonyLists,LOD,preemptive,slot): ##?? each node should have a time that shows better detail of execution of tasks that remains after end of time slot
    endTime=(slot+1)*subSlot
    print("In execution Phase",file=myFile)
    print("bases on ",name,file=myFile)
    #print("Exe started and the time is ",(listOfColonies[0].listOfNodes[0]).time,file=myFile)
    print("Exe started and the time is ",(colonyLists[0].listOfComputingNodes[0]).time,file=myFile) # the aboved showed 0 in some scenarios all time !
    print("",file=myFile)
    #time.sleep(3)
    for node in (colonyLists[0].listOfComputingNodes):   
        print("",file=myFile)
        print("************node is ",node.name,file=myFile)
        print("node.time",node.time,file=myFile)
        print("end is ",endTime,file=myFile)
        print("Cpu of node is ",node.cpu,file=myFile)
        print("Len of the qeues : ",len(node.queueOfNormals+node.queueOfCriticals),file=myFile)
        #time.sleep(5)
        #time.sleep(7)
        #node.criticalRun=0 ### for our poroposed alg
        #node.normalRun=0
        #node.freeCpu=0

        while node.time<endTime:
            print("node.time In the loop",node.time,file=myFile)
            #timeDoneBefore=0 ## for handling Queuing delay for preemptive task 
            selectedTask=executionStrategy(node,endTime)
            #time.sleep(5)
            if selectedTask=="no task": ##if no task is ready to be executed
                if (node.time<endTime): ##if tasks are done before end of slot, so update the time node to the begenning of next slot
                    print("no task & breaking the while",file=myFile)
                    ##this if added recently: to not push time of a time to endTime when all tasks are finished
                    if (slot<totalSlots): ##?? xx XX
                        node.time=endTime
                    #time.sleep(2)
                break
            else :# if there is a task and can be exe
                if(selectedTask.arrivalTime>node.time): #move the time of node to future and the moment task come to queue to be executed
                    node.time=selectedTask.arrivalTime

            requiredCpuRun=selectedTask.cpu-selectedTask.doneCPU
            print("remained cpu is ",requiredCpuRun,file=myFile)
            ## need to update task class and node time after execution
            if ((selectedTask.type=="normal") & (preemptive)):
                print("Selected task is normal & preemptive",selectedTask.id,file=myFile)
                remaining=(selectedTask.cpu-selectedTask.doneCPU) # remained job of task from last execution
                dynamicTS=DynamicTimeSlot(node,executionStrategy,endTime) ## calculate allowed cpu run for normal tasks based on execution strategy
                print("TS is ",dynamicTS,file=myFile)
                # XX xx ?? the dynamic sl sounds big and normal after done will not be removed
                #print("calculating Dynamic Sl is ",dynamicTS)
                #print("remaining ",remaining)
                if dynamicTS==-1:
                    print("dynamicTS==-1 ",file=myFile)
                    print("LETS FINISH THE PROGRAM ",file=myFile)
                    exit()
                if dynamicTS>=remaining:#if time slot be more than need of a task
                    dynamicTS=remaining
                    print("dynamicTS>=remaining and --> dynamicTS=remaining ",file=myFile)
                
                print("node.time before running the Dynamic TS ",node.time,file=myFile)
                print("Dynamic TS time is",(dynamicTS/node.cpu)*timeSlot,file=myFile)
                selectedTask.doneCPU+=dynamicTS
                node.normalRun+=(dynamicTS) # for completeing tables and proposed alg
                node.time+=((dynamicTS/node.cpu)*timeSlot) ## xx XX timeslot or 1000  recently added here this line, was in line 888 before
                remaining=(selectedTask.cpu-selectedTask.doneCPU) # remained job of task after this execution
                print("node.time after running the Dynamic TS ",node.time,file=myFile)
                print("selectedTask.cpu is ",selectedTask.cpu,file=myFile)
                print("selectedTask.doneCPU after running as TS ",selectedTask.doneCPU,file=myFile)

                if remaining>C: ## if task's reamining job is bigger than threshold we must run scheduling alg to find next task
                    print("Remining part of task is big so we need a pause to find a task for process",file=myFile)
                    continue
                else:# ((remaining<=C)):
                    if remaining==0: ## task is finished all ready
                        print("we finished the current task. let's, adapt the cahnges to system",file=myFile)
                    elif ((executionStrategy==MQPExecutingAlgorithm) | (executionStrategy==MQPExecutingAlgorithm2)): ## if the Alg is MQP and  remianed cpu after execution is less than the threshold
                        print("Remining part of task is small so we continue to finish it",file=myFile)
                        selectedTask.doneCPU+=remaining
                        node.time+=((remaining/node.cpu)*timeSlot)
                        node.normalRun+=(remaining)
                        print("we finished the small remained of the current task. let's, adapt the cahnges to system, time after finishing the small remaining is  ",node.time,file=myFile)
                    else : #(remaining<=C) but alg is not MQP
                        print("small part remained, but we have to make decision agian for next task",file=myFile)
                        continue
                        
                #print("timeDoneBefore ",timeDoneBefore,file=myFile) ## XX xx One tab is come front recently-
            else :# (when task is Critical or (Normal and Not Preepmtive))
                # update node.time and tables for node after execution of task;
                print("non preemptive part",file=myFile)
                print("node.time before new changes ",node.time,file=myFile)
                if selectedTask.type=="normal":
                    node.normalRun+=(requiredCpuRun)
                else:# selected.type=="critical"
                    node.criticalRun+=(requiredCpuRun)   
                node.time+=((requiredCpuRun/node.cpu)*timeSlot)
                selectedTask.doneCPU+=requiredCpuRun            
                
            executionTime=round(((selectedTask.cpu)/(node.cpu))*timeSlot,3)#xx XX
            selectedTask.executionDelay=executionTime
            selectedTask.queuingDelay=round(node.time-selectedTask.arrivalTime-executionTime,3) ## recently added
            selectedTask.schedulingDelay=selectedTask.schedulingTime-selectedTask.generationTime
            downloadingTime=download(selectedTask,node,selectedTask.src) #downloading of the result from the dst
            selectedTask.communicationDelay+=(downloadingTime) ##adding downloading time to com delay
            #selectedTask.finishTime=selectedTask.generationTime+selectedTask.schedulingDelay+selectedTask.communicationDelay+selectedTask.queuingDelay+selectedTask.executionDelay
            selectedTask.finishTime=node.time+ downloadingTime
            #selectedTask.responseTime=selectedTask.finishTime-selectedTask.generationTime
            selectedTask.responseTime=selectedTask.finishTime-selectedTask.generationTime
            print("selected task ",selectedTask.id,file=myFile)
            print("cpu ",selectedTask.cpu,file=myFile)
            print("Generation Time ",selectedTask.generationTime,file=myFile)
            print("Scheduling Time ",selectedTask.schedulingTime,file=myFile)
            print("Scheduling Delay ",selectedTask.schedulingDelay,file=myFile)
            print("Com delay ",selectedTask.communicationDelay,file=myFile)
            print("arrivalTime is ",selectedTask.arrivalTime,file=myFile)
            print("Queing delay ",selectedTask.queuingDelay,file=myFile)
            print("execution Delay ",selectedTask.executionDelay,file=myFile)
            print("finish Time ",selectedTask.finishTime,file=myFile)
            print("Deadline is ",selectedTask.deadline,file=myFile)
            print("resp Time ",selectedTask.responseTime,file=myFile)
            node.activeTime+=(executionTime) ##update node, active Time
            #time.sleep(5)
            #node.time=selectedTask.finishTime #xx XX ?? update node time after execution of the task !!! 
            if selectedTask in node.queueOfNormals: ##remove the task from queues after execution
                #print("len before ",len(node.queueOfNormals))
                (node.queueOfNormals).remove(selectedTask)
                #print("len after ",len(node.queueOfNormals))
            else:
                #print("len before ",len(node.queueOfCriticals))
                (node.queueOfCriticals).remove(selectedTask)
                #print("len after ",len(node.queueOfCriticals))
            node.queueOfDoneTasks.append(selectedTask) ##adding the finished task to the queue of done taks of the executing(dst) node
            LOD.append(selectedTask) ##additional for checking
            selectedTask.finished=True
            #time.sleep(20)
        print("time of node after doing tasks ",node.time,file=myFile)
        print("End",file=myFile)
        #time.sleep(30)
        updateNodeInnerTable(node) ## our proposed alg ## update manager.innerTable after each node done its tasks at the end of each slot


def download(task,src,dst):# src=the exucitive node, dst=the node that sent the task for exe
    print("*----------Download Phase-----------* ")
    print("comback from ",src.name)
    print(" to ",dst.name)
    com1=0 #if cloud is chosen as computer node before
    com2=0
    totalDelay=0
    if ("cloud" in (src.name)):
        BW,TransDelay=managersBW_TransDelay[((src).name,((dst.parent).nodeManager).name)] #delay from cloud to src colony manager
        com1=TransDelay+((task.outsize)/BW)
        com2=communicationDelayComputer(task,((dst.parent).nodeManager),dst,task.outsize) #send the task from src to manager
        totalDelay=com1+com2
        return totalDelay
    srcColony=src.parent
    dstColony=dst.parent
    if src.name==dst.name: ## no delay when the task is executed in generative node 
        return totalDelay
    elif srcColony.id==dstColony.id:
        com2=communicationDelayComputer(task,((dst.parent).nodeManager),dst,task.outsize) #send the task from src to manager
        totalDelay=com2
        return totalDelay
    else:
        com0=0 ##from src to its colony manager 
        com0=communicationDelayComputer(task,src,((srcColony).nodeManager),task.outsize) #send the task from src to manager
        BW,TransDelay=managersBW_TransDelay[((srcColony.nodeManager).name,((dstColony).nodeManager).name)] #delay from cloud to src colony manager
        com1=TransDelay+((task.outsize)/BW) ## manager to manager
        com2=communicationDelayComputer(task,(dstColony.nodeManager),dst,task.outsize) #send the task from src to manager
        totalDelay=com0+com1+com2
        return totalDelay


def proposedExecutingAlgorithm2(node,endTime): ## based on slack Time
    global slackEnabeled2
    global slackTime2
    print("#############proposedExecutingAlgorithm2###########",file=myFile)
    print("slackEnabeled2 ",slackEnabeled2,file=myFile)

    (node.queueOfCriticals).sort(key=lambda task: task.arrivalTime) #, reverse=True  sort the tasks in queueOfCriticals based on arrival time
    (node.queueOfNormals).sort(key=lambda task: task.arrivalTime) #sort the tasks in queueOfNormals based on arrival time
    

    lenC=len(node.queueOfCriticals)
    lenN=len(node.queueOfNormals)
    if ((lenC==0) & (lenN==0)):
        print("((lenC==0) & (lenN==0)): ",file=myFile)
        #print("((lenC==0) & (lenN==0)): ")
        print("return no task",file=myFile)
        slackEnabeled2=False
        return "no task"
    elif ((lenC!=0) & (lenN==0)): ## Q: if the current subslot we dont have critical, we choose a normal to be run. But till when ? (end or the next subslot when a critical task will come ?)
        taskCritical=(node.queueOfCriticals)[0]
        print("((lenC!=0) & (lenN==0)): ",file=myFile)
        #print("((lenC!=0) & (lenN==0)): ")
        print("return taskCritical",file=myFile)
        if taskCritical.arrivalTime<endTime:
            return taskCritical
    elif ((lenC==0) & (lenN!=0)): 
        taskNormal=(node.queueOfNormals)[0]
        print("((lenC==0) & (lenN!=0))",file=myFile)
        #print("((lenC==0) & (lenN!=0)): ")
        if taskNormal.arrivalTime<endTime:
            print("return taskNormal",file=myFile)
            return taskNormal
    elif ((lenC!=0) & (lenN!=0)):## here ??????
        #time.sleep(2)
        print("((lenC!=0) & (lenN!=0))",file=myFile)
        #print("((lenC!=0) & (lenN!=0)): ")
        taskCritical=(node.queueOfCriticals)[0]
        taskNormal=(node.queueOfNormals)[0]
        #slackCalculated=computeSlack(taskCritical,node)

        if (taskCritical.arrivalTime<=node.time) :
            print("taskCritical.arrivalTime<=node.time",file=myFile)
            #print("taskCritical.arrivalTime<=node.time")
            if taskCritical.arrivalTime<endTime:
                print("taskCritical.arrivalTime<endTime",file=myFile)
                #print("taskCritical.arrivalTime<endTime")
                slackCalculated=computeSlack(taskCritical,node,node.time)
                print("slackCalculated",slackCalculated,file=myFile)
                #print("slackCalculated for critical",slackCalculated)
                if slackCalculated>0:
                    print("slackCalculated>0",file=myFile)
                    #print("slackCalculated>0")
                    if taskNormal.arrivalTime<=node.time: # we have a normal task already in the buffer that came before node.time
                        print("taskNormal.arrivalTime<=node.time",file=myFile)
                        #print("taskNormal.arrivalTime<=node.time")
                        print("return taskNormal",file=myFile)
                        #print("return taskNormal")
                        slackEnabeled2=True
                        slackTime2=slackCalculated
                        return taskNormal
                    
                print("return taskCritical",file=myFile)
                #print("return taskCritical")
                return taskCritical

        elif taskNormal.arrivalTime<taskCritical.arrivalTime:
            print("taskNormal.arrivalTime<taskCritical.arrivalTime",file=myFile)
            #print("taskNormal.arrivalTime<taskCritical.arrivalTime")
            startOfNormal=taskNormal.arrivalTime
            if startOfNormal<=node.time:
                print("startOfNormal<=node.time",file=myFile)
                print("startOfNormal=node.time",file=myFile)
                startOfNormal=node.time
            slackCalculated=computeSlack(taskCritical,node,startOfNormal)
            print("slackCalculated ",slackCalculated,file=myFile)
            #print("slackCalculated ",slackCalculated)
            #timeTillNextCritical=taskCritical.arrivalTime-startOfNormal ## time from critical arrive till normal arrive
            #print("timeTillNextCritical ",timeTillNextCritical,file=myFile)

            #slackTime2=timeTillNextCritical+slackCalculated
            if slackCalculated>0:
                slackTime2=slackCalculated
                slackEnabeled2=True
                print("return taskNormall ",file=myFile)
                print("slackEnabeled2 ",slackEnabeled2,file=myFile)
                return taskNormal
            print("return taskNormal ",file=myFile)
            return taskCritical
            
        elif (taskCritical.arrivalTime<taskNormal.arrivalTime):
            #print("taskCritical.arrivalTime<taskNormal.arrivalTime")
            print("taskCritical.arrivalTime<taskNormal.arrivalTime",file=myFile)
            if taskNormal.arrivalTime<endTime:
                print("return taskCritical ",file=myFile)
                return taskCritical
               
    return "no task"


def proposedExecutingAlgorithm(node,endTime): ## based on slack Time
    global slackEnabeled
    global slackTime
    print("#############proposedExecutingAlgorithm2###########",file=myFile)
    print("slackEnabeled2 ",slackEnabeled,file=myFile)

    (node.queueOfCriticals).sort(key=lambda task: task.arrivalTime) #, reverse=True  sort the tasks in queueOfCriticals based on arrival time
    (node.queueOfNormals).sort(key=lambda task: task.arrivalTime) #sort the tasks in queueOfNormals based on arrival time
    
    lenC=len(node.queueOfCriticals)
    lenN=len(node.queueOfNormals)
    if ((lenC==0) & (lenN==0)):
        print("((lenC==0) & (lenN==0)): ",file=myFile)
        #print("((lenC==0) & (lenN==0)): ")
        print("return no task",file=myFile)
        slackEnabeled=False
        return "no task"
    elif ((lenC!=0) & (lenN==0)): ## Q: if the current subslot we dont have critical, we choose a normal to be run. But till when ? (end or the next subslot when a critical task will come ?)
        taskCritical=(node.queueOfCriticals)[0]
        print("((lenC!=0) & (lenN==0)): ",file=myFile)
        #print("((lenC!=0) & (lenN==0)): ")
        print("return taskCritical",file=myFile)
        if taskCritical.arrivalTime<endTime:
            return taskCritical
    elif ((lenC==0) & (lenN!=0)): 
        taskNormal=(node.queueOfNormals)[0]
        print("((lenC==0) & (lenN!=0))",file=myFile)
        #print("((lenC==0) & (lenN!=0)): ")
        if taskNormal.arrivalTime<endTime:
            print("return taskNormal",file=myFile)
            return taskNormal
    elif ((lenC!=0) & (lenN!=0)):## here ??????
        #time.sleep(2)
        print("((lenC!=0) & (lenN!=0))",file=myFile)
        #print("((lenC!=0) & (lenN!=0)): ")
        taskCritical=(node.queueOfCriticals)[0]
        taskNormal=(node.queueOfNormals)[0]
        #slackCalculated=computeSlack(taskCritical,node)

        if (taskCritical.arrivalTime<=node.time) :
            print("taskCritical.arrivalTime<=node.time",file=myFile)
            #print("taskCritical.arrivalTime<=node.time")
            if taskCritical.arrivalTime<endTime:
                print("taskCritical.arrivalTime<endTime",file=myFile)
                #print("taskCritical.arrivalTime<endTime")
                slackCalculated=computeSlack(taskCritical,node,node.time)
                print("slackCalculated",slackCalculated,file=myFile)
                #print("slackCalculated for critical",slackCalculated)
                if slackCalculated>0:
                    print("slackCalculated>0",file=myFile)
                    #print("slackCalculated>0")
                    if taskNormal.arrivalTime<=node.time: # we have a normal task already in the buffer that came before node.time
                        print("taskNormal.arrivalTime<=node.time",file=myFile)
                        #print("taskNormal.arrivalTime<=node.time")
                        print("return taskNormal",file=myFile)
                        #print("return taskNormal")
                        slackEnabeled=True
                        slackTime=slackCalculated
                        return taskNormal
                    
                print("return taskCritical",file=myFile)
                #print("return taskCritical")
                return taskCritical

        elif taskNormal.arrivalTime<taskCritical.arrivalTime:
            print("taskNormal.arrivalTime<taskCritical.arrivalTime",file=myFile)
            #print("taskNormal.arrivalTime<taskCritical.arrivalTime")
            startOfNormal=taskNormal.arrivalTime
            if startOfNormal<=node.time:
                print("startOfNormal<=node.time",file=myFile)
                print("startOfNormal=node.time",file=myFile)
                startOfNormal=node.time
            slackCalculated=computeSlack(taskCritical,node,startOfNormal)
            print("slackCalculated ",slackCalculated,file=myFile)
            #print("slackCalculated ",slackCalculated)
            #timeTillNextCritical=taskCritical.arrivalTime-startOfNormal ## time from critical arrive till normal arrive
            #print("timeTillNextCritical ",timeTillNextCritical,file=myFile)

            #slackTime2=timeTillNextCritical+slackCalculated
            if slackCalculated>0:
                slackTime=slackCalculated
                slackEnabeled=True
                print("return taskNormall ",file=myFile)
                print("slackEnabeled2 ",slackEnabeled2,file=myFile)
                return taskNormal
            print("return taskNormal ",file=myFile)
            return taskCritical
            
        elif (taskCritical.arrivalTime<taskNormal.arrivalTime):
            #print("taskCritical.arrivalTime<taskNormal.arrivalTime")
            print("taskCritical.arrivalTime<taskNormal.arrivalTime",file=myFile)
            if taskNormal.arrivalTime<endTime:
                print("return taskCritical ",file=myFile)
                return taskCritical
               
    return "no task"

def computeSlack(taskCritical,node,now):
    print("computeSlack ",file=myFile)
    print("now ",now,file=myFile)

    deadline=taskCritical.deadline
    execution=(taskCritical.cpu/node.cpu)*timeSlot
    generation=taskCritical.generationTime
    print("task is ",taskCritical.id,file=myFile)
    print("deadline ",deadline,file=myFile)
    print("execution ",execution,file=myFile)
    print("generation ",generation,file=myFile)
    ##now=node.time 
    #timeDif=now-generation
    timeDif=generation-now # critical task   
    print("timeDif ",timeDif,file=myFile)
    slackTime=timeDif+(deadline-execution)
    #slackTime=deadline-(timeDif-execution)
    print("slackTime ",slackTime,file=myFile)
    if slackTime>slackThr:
        print("slackTime>slackThr ",file=myFile)
        print("slackTime-slackThr ",slackTime-slackThr,file=myFile)
        return round(slackTime-slackThr,2)
    else :
        print("else ",file=myFile)
        return 0

## we have to update here for new 5 xx ?? XX

def DynamicTimeSlot(node,executionStrategy,endTime): # computing allowed cpu run for normal tasks in premption case for different strategies
    print("DynamicTimeSlot calculator ",file=myFile)

    TS=0
    if ((executionStrategy==MQPExecutingAlgorithm) |(executionStrategy==MQPExecutingAlgorithm2)):
        TS=dynamicTSForMQP(node,endTime)
    elif (executionStrategy==criticalFirstExecutingAlgorithm):
        TS=dynamicTSForCF(node,endTime)
    elif (executionStrategy==New1proposedExecutingAlgorithm):
        TS=dynamicTSForNew1Proposed(node,endTime)
    elif (executionStrategy==New2proposedExecutingAlgorithm):
        TS=dynamicTSForNew2Proposed(node,endTime)
    elif (executionStrategy==New3proposedExecutingAlgorithm):
        TS=dynamicTSForNew3Proposed(node,endTime)
    elif (executionStrategy==proposedExecutingAlgorithm):
        print("proposedExecutingAlgorithm ",file=myFile)
        TS=dynamicTSForProposed(node,endTime)
    elif (executionStrategy==proposedExecutingAlgorithm2): # for CaseY
        print("proposedExecutingAlgorithm2 ",file=myFile)
        TS=dynamicTSForProposed2(node,endTime)
    else :
        TS= "Error"
    return TS

# got commented last edit: when debauging the CF algorithm

'''def dynamicTSForProposed(node,endTime): #XX xx /slot or /subslot
    slack=-1
    
    if slackEnabeled==True:
        slack= slackTime
    else :#task normal and no opcoming critical task in this current slot:  
        slack= endTime-node.time #we continue processing the normal task till end of the slot, then we decide later
    
    return (slack/timeSlot)*node.cpu #?? xx
'''

def dynamicTSForProposed(node,endTime): 
    global slackEnabeled

    slack=-1
    timeTillEndOfSlot=endTime-node.time 

    print("dynamicTSForProposed ",file=myFile)
    print("slackEnabeled ",slackEnabeled,file=myFile)
    if slackEnabeled==True:
        slackEnabeled=False
        slack= slackTime
        print("slackEnabeled==True ",file=myFile)
        print("slack ",slackTime,file=myFile)
        
        if slack>timeTillEndOfSlot:
            slack=timeTillEndOfSlot
            print("slack>timeTillEndOfSlot ",file=myFile)
            print("slack=timeTillEndOfSlot :" ,slack,file=myFile)
    
    else :#task normal and no opcoming critical task in this current slot:  
        print("else",file=myFile)
        slack=timeTillEndOfSlot #we continue processing the normal task till end of the slot, then we decide later
        print("slack ",slack,file=myFile)
        
    return (slack/timeSlot)*node.cpu #?? xx



def dynamicTSForProposed2(node,endTime): #for CaseY
    global slackEnabeled2

    slack=-1
    timeTillEndOfSlot=endTime-node.time 

    print("dynamicTSForProposed2 ",file=myFile)
    print("slackEnabeled2 ",slackEnabeled2,file=myFile)
    if slackEnabeled2==True:
        slackEnabeled2=False
        slack= slackTime2
        print("slackEnabeled2==True ",file=myFile)
        print("slack ",slackTime2,file=myFile)
        
        if slack>timeTillEndOfSlot:
            slack=timeTillEndOfSlot
            print("slack>timeTillEndOfSlot ",file=myFile)
            print("slack=timeTillEndOfSlot :" ,slack,file=myFile)
    
    else :#task normal and no opcoming critical task in this current slot:  
        print("else",file=myFile)
        slack=timeTillEndOfSlot #we continue processing the normal task till end of the slot, then we decide later
        print("slack ",slack,file=myFile)
        
    return (slack/timeSlot)*node.cpu #?? xx


def dynamicTSForNew1Proposed(node,endTime):
    slack=-1
    print("dynamicTSForNew1Proposed ",file=myFile)
    if new1Time!=-1:
        print("new1Time!=-1 ",file=myFile)
        print("new1Time ",new1Time,file=myFile)
        slack= new1Time
    else : #new1Time==-1 --> task normal and no opcoming critical task in this current slot:
        print("new1Time==-1 task normal and no opcoming critical task in this timeslot",file=myFile)
        slack=endTime-node.time#continue processing the normal task till end of the slot, then we decide later
        print("slack calculated ",slack,file=myFile)
    print("returning (slack/timeSlot)*node.cpu ",(slack/timeSlot)*node.cpu,file=myFile)
    return (slack/timeSlot)*node.cpu #?? xx

def dynamicTSForNew2Proposed(node,endTime):
    slack=-1
    print("dynamicTSForNew2Proposed ",file=myFile)
    if new2Time!=-1:
        print("new2Time!=-1 ",file=myFile)
        print("new2Time ",new1Time,file=myFile)
        slack= new2Time
    else : #new2Time==-1 --> task normal and no opcoming critical task in this current slot:
        print("new2Time==-1 task normal and no opcoming critical task in this timeslot",file=myFile)
        slack=endTime-node.time#continue processing the normal task till end of the slot, then we decide later
        print("slack calculated ",slack,file=myFile)
    print("returning (slack/timeSlot)*node.cpu ",(slack/timeSlot)*node.cpu,file=myFile)
    return (slack/timeSlot)*node.cpu #?? xx

def dynamicTSForNew3Proposed(node,endTime):
    slack=-1
    print("dynamicTSForNew2Proposed ",file=myFile)
    if new3Time!=-1:
        print("new3Time!=-1 ",file=myFile)
        print("new3Time ",new1Time,file=myFile)
        slack= new3Time
    else : #new2Time==-1 --> task normal and no opcoming critical task in this current slot:
        print("new3Time==-1 task normal and no opcoming critical task in this timeslot",file=myFile)
        slack=endTime-node.time#continue processing the normal task till end of the slot, then we decide later
        print("slack calculated ",slack,file=myFile)
    print("returning (slack/timeSlot)*node.cpu ",(slack/timeSlot)*node.cpu,file=myFile)
    return (slack/timeSlot)*node.cpu #?? xx

def dynamicTSForCF(node,endTime):
    slack=-1
    print("dynamicTSForCF ",file=myFile)
    if CFTime!=-1:
        print("CFTime!=-1 ",file=myFile)
        print("CFTime ",CFTime,file=myFile)
        slack= CFTime
    else :#task normal and no opcoming critical task in this current slot:
        print("CFTime==-1 task normal and no opcoming critical task in this timeslot",file=myFile)
        slack=endTime-node.time#continue processing the normal task till end of the slot, then we decide later
        print("slack calculated ",slack,file=myFile)
    print("returning (slack/timeSlot)*node.cpu ",(slack/timeSlot)*node.cpu,file=myFile)
    return (slack/timeSlot)*node.cpu #?? xx


def dynamicTSForMQP(node,endTime): ## xx XX return cpu or timeslot
    total=0
    for task in node.queueOfNormals:
        total+=(task.cpu-task.doneCPU)
    avg=total/(len(node.queueOfNormals))
    DynamicTimeSlot=(P/100)*avg
    #remained=((endTime-node.time)/timeSlot)*node.cpu

    ##if DynamicTimeSlot>remained:
        ##DynamicTimeSlot=remained

    return DynamicTimeSlot


def MQPExecutingAlgorithm(node,endTime):
    print("#############MQPExecutingAlgorithm###########",file=myFile)
    (node.queueOfCriticals).sort(key=lambda task: task.cpu) ##sort based on burst time
    (node.queueOfNormals).sort(key=lambda task: (task.cpu-task.doneCPU))
    lenC=len(node.queueOfCriticals)
    lenN=len(node.queueOfNormals)
    global m
    if ((lenC==0) & (lenN==0)):
        return "no task"
    elif ((lenC!=0) & (lenN!=0)):
        taskCritical=min(node.queueOfCriticals, key=lambda x: x.arrivalTime)
        taskNormal=min(node.queueOfNormals, key=lambda x: x.arrivalTime)
        if m<M:## turn of critical
            for task in node.queueOfCriticals:
                if task.arrivalTime<=node.time:
                    m+=1
                    return task
            if taskCritical.arrivalTime<=taskNormal.arrivalTime:
                m+=1
                return taskCritical
            else: #taskCritical.arrivalTime>taskNormal.arrivalTime:
                m=0
                return taskNormal
        else: ## turn of normals
            for task in node.queueOfNormals:
                if task.arrivalTime<=node.time:
                    m=0
                    return task
            if taskNormal.arrivalTime<=taskCritical.arrivalTime:
                m=0
                return taskNormal
            else:#taskCritical.arrivalTime<taskNormal.arrivalTime:
                m+=1
                return taskCritical
    elif (lenC!=0): ## & (lenN==0)
        taskCritical=min(node.queueOfCriticals, key=lambda x: x.arrivalTime)
        m+=1
        for task in node.queueOfCriticals:
            if task.arrivalTime<=node.time:
                return task
        return taskCritical
    elif (lenN!=0): ## & (lenC==0)
        taskNormal=min(node.queueOfNormals, key=lambda x: x.arrivalTime)
        m=0
        for task in node.queueOfNormals:
            if task.arrivalTime<=node.time:
                return task
        return taskNormal
    else:
        return "no task"
    

def MQPExecutingAlgorithm2(node,endTime):
    print("#############MQPExecutingAlgorithm###########",file=myFile)
    (node.queueOfCriticals).sort(key=lambda task: task.cpu) ##sort based on burst time
    (node.queueOfNormals).sort(key=lambda task: (task.cpu-task.doneCPU))
    lenC=len(node.queueOfCriticals)
    lenN=len(node.queueOfNormals)
    global m2
    if ((lenC==0) & (lenN==0)):
        return "no task"
    elif ((lenC!=0) & (lenN!=0)):
        taskCritical=min(node.queueOfCriticals, key=lambda x: x.arrivalTime)
        taskNormal=min(node.queueOfNormals, key=lambda x: x.arrivalTime)
        if m2<M2:## turn of critical
            for task in node.queueOfCriticals:
                if task.arrivalTime<=node.time:
                    m2+=1
                    return task
            if taskCritical.arrivalTime<=taskNormal.arrivalTime:
                m2+=1
                return taskCritical
            else: #taskCritical.arrivalTime>taskNormal.arrivalTime:
                m2=0
                return taskNormal
        else: ## turn of normals
            for task in node.queueOfNormals:
                if task.arrivalTime<=node.time:
                    m2=0
                    return task
            if taskNormal.arrivalTime<=taskCritical.arrivalTime:
                m2=0
                return taskNormal
            else:#taskCritical.arrivalTime<taskNormal.arrivalTime:
                m2+=1
                return taskCritical
    elif (lenC!=0): ## & (lenN==0)
        taskCritical=min(node.queueOfCriticals, key=lambda x: x.arrivalTime)
        m2+=1
        for task in node.queueOfCriticals:
            if task.arrivalTime<=node.time:
                return task
        return taskCritical
    elif (lenN!=0): ## & (lenC==0)
        taskNormal=min(node.queueOfNormals, key=lambda x: x.arrivalTime)
        m2=0
        for task in node.queueOfNormals:
            if task.arrivalTime<=node.time:
                return task
        return taskNormal
    else:
        return "no task"


def New1proposedExecutingAlgorithm(node,endTime):# it works like critical First

    ## first order normal and critical task based on arrival time :FCFS
    ## executing strategy is like Critical First

    print("#############New1proposedExecutingAlgorithm###########",file=myFile)
    (node.queueOfCriticals).sort(key=lambda task: task.arrivalTime) #, reverse=True  sort the tasks in queueOfCriticals based on arrival time
    (node.queueOfNormals).sort(key=lambda task: task.arrivalTime) #sort the tasks in queueOfNormals based on arrival Time
    
    lenC=len(node.queueOfCriticals)
    lenN=len(node.queueOfNormals)
    #CFTime=-1 before

    global new1Time
    new1Time=-1

    print("(lenC==0) ",lenC,file=myFile)
    print("(lenN==0) ",lenN,file=myFile)

    if ((lenC==0) & (lenN==0)):
        print("(lenC==0) & (lenN==0)",file=myFile)
        return "no task"
    elif ((lenC!=0) & (lenN==0)):
        print("(lenC!=0) & (lenN==0)",file=myFile)
        taskCritical=(node.queueOfCriticals)[0]
        if taskCritical.arrivalTime<endTime:
            print("taskCritical.arrivalTime<endTime",file=myFile)
            return taskCritical
    elif ((lenC==0) & (lenN!=0)):
        print("(lenC==0) & (lenN!=0)",file=myFile)
        taskNormal=(node.queueOfNormals)[0]
        if taskNormal.arrivalTime<endTime:
            print("taskNormal.arrivalTime<endTime",file=myFile)
            return taskNormal
    elif ((lenC!=0) & (lenN!=0)):
        print("(lenC!=0) & (lenN!=0)",file=myFile)
        taskCritical=(node.queueOfCriticals)[0]
        taskNormal=(node.queueOfNormals)[0]

        #new (if it does not work , here must be removed )
        if (taskNormal.arrivalTime<taskCritical.arrivalTime) & (taskCritical.arrivalTime>node.time):
            print("(taskNormal.arrivalTime<taskCritical.arrivalTime) & (taskCritical.arrivalTime>node.time)",file=myFile)
            if taskNormal.arrivalTime<endTime: ## we need change here :for how much time normal be run?
                print("taskNormal.arrivalTime<endTime",file=myFile)
                startOfNormal=taskNormal.arrivalTime
                if taskNormal.arrivalTime<=node.time:
                    print("taskNormal.arrivalTime<=node.time",file=myFile)
                    startOfNormal=node.time

                timeTillNextCritical=taskCritical.arrivalTime-startOfNormal
                new1Time=timeTillNextCritical
                return taskNormal
            
        return taskCritical
        
        
    return "no task"


def New2proposedExecutingAlgorithm(node,endTime): ## CF + normals sorted based on manager scheduling priority
    ## first order criticals based on arrival time
    ## order normals based on manager shceduling preference : remaned_deadline/job_remained
    ## executing strategy is like Critical First 
    ## (if first a low priority normal task comes in queue and no other task is available, cpu waits for higher priority task)

    print("#############New2proposedExecutingAlgorithm###########",file=myFile)
    (node.queueOfCriticals).sort(key=lambda task: task.arrivalTime) #, reverse=True  sort the tasks in queueOfCriticals based on arrival time
    (node.queueOfNormals).sort(key=lambda task: (task.cpu-task.doneCPU)/(task.deadline-(node.time-task.generationTime)), reverse=True) #sort the tasks in queueOfNormals based on remained deadline and remained job

    lenC=len(node.queueOfCriticals)
    lenN=len(node.queueOfNormals)

    global new2Time
    new2Time=-1


    print("(lenC==0) ",lenC,file=myFile)
    print("(lenN==0) ",lenN,file=myFile)

    if ((lenC==0) & (lenN==0)):
        print("(lenC==0) & (lenN==0)",file=myFile)
        return "no task"
    elif ((lenC!=0) & (lenN==0)):
        print("(lenC!=0) & (lenN==0)",file=myFile)
        taskCritical=(node.queueOfCriticals)[0]
        if taskCritical.arrivalTime<endTime:
            print("taskCritical.arrivalTime<endTime",file=myFile)
            return taskCritical
    elif ((lenC==0) & (lenN!=0)):
        print("(lenC==0) & (lenN!=0)",file=myFile)
        taskNormal=(node.queueOfNormals)[0]
        if taskNormal.arrivalTime<endTime:
            print("taskNormal.arrivalTime<endTime",file=myFile)
            return taskNormal
    elif ((lenC!=0) & (lenN!=0)):
        print("(lenC!=0) & (lenN!=0)",file=myFile)
        taskCritical=(node.queueOfCriticals)[0]
        taskNormal=(node.queueOfNormals)[0]

        #new (if it does not work , here must be removed )
        if (taskNormal.arrivalTime<taskCritical.arrivalTime) & (taskCritical.arrivalTime>node.time):
            print("(taskNormal.arrivalTime<taskCritical.arrivalTime) & (taskCritical.arrivalTime>node.time)",file=myFile)
            if taskNormal.arrivalTime<endTime: ## we need change here :for how much time normal be run?
                print("taskNormal.arrivalTime<endTime",file=myFile)
                startOfNormal=taskNormal.arrivalTime
                if taskNormal.arrivalTime<=node.time:
                    print("taskNormal.arrivalTime<=node.time",file=myFile)
                    startOfNormal=node.time

                timeTillNextCritical=taskCritical.arrivalTime-startOfNormal
                new2Time=timeTillNextCritical
                return taskNormal
            
        return taskCritical
        
        
    return "no task"


def weighted_score(tasks,task, cpu_weight, deadline_weight):# for new3 proposed execution algorithm
    
    #calculate score of a task based on a wighted score of it's remained deadline and remained job

    # Find the maximum CPU and deadline values for normalization
    max_cpu = max(task.cpu for task in tasks)
    max_deadline = max(task.deadline for task in tasks)


    # Normalize CPU and deadline (assuming lower values are better for both)
    normalized_cpu = task.cpu / max_cpu  # Normalize CPU by the maximum CPU value in the list
    normalized_deadline = task.deadline / max_deadline  # Normalize deadline by the maximum deadline in the list

    # Calculate the weighted score
    score = (cpu_weight * normalized_cpu) + (deadline_weight * normalized_deadline)
    task.score=score
    return score


def New3proposedExecutingAlgorithm(node,endTime): # alg between c & n : CF, between normals: deadline_remained + sjf 
    ## first order normal task based on remained deadline 
    ## executing strategy is like Critical First
    
    print("#############New3proposedExecutingAlgorithm###########",file=myFile)


    (node.queueOfCriticals).sort(key=lambda task: task.arrivalTime) #, reverse=True  sort the tasks in queueOfCriticals based on arrival time
    (node.queueOfNormals).sort(key=lambda task: task.deadline+task.generationTime) #sort the tasks in queueOfNormals based on remained deadline
    

    normalTasks=node.queueOfNormals

    # Define weights for CPU and deadline
    cpu_weight = 0.6  # Higher weight for CPU
    deadline_weight = 0.4  # Lower weight for deadline

    # Sort the tasks based on the weighted score
    sortedByScore = sorted(node.queueOfNormals, key=lambda task: weighted_score(normalTasks,task, cpu_weight, deadline_weight))

    lenC=len(node.queueOfCriticals)
    lenN=len(node.queueOfNormals)
    

    global new3Time
    new3Time=-1

    print("(lenC) ",lenC,file=myFile)
    print("(lenN) ",lenN,file=myFile)

    if ((lenN==0)):
        print("(lenN==0)",file=myFile)
        if (lenC==0):
            print("(lenC==0) & (lenN==0)",file=myFile)
            return "no task"
        elif ((lenC!=0) & (lenN==0)):
            print("(lenC!=0) & (lenN==0)",file=myFile)
            firstCritical=(node.queueOfCriticals)[0] ## among Criticals: FCFS
            if firstCritical.arrivalTime<endTime:
                print("firstCritical.arrivalTime<endTime",file=myFile)
                return firstCritical
            
    else: # ==((lenN!=0)):

        print("(lenN!=0)",file=myFile)
        firstNormal=(node.queueOfNormals)[0]# first arriving normal
    
        if ((lenC==0)):
            print("(lenC==0)",file=myFile)
            for highScore in sortedByScore:# a high score normal is in queue already: it has higher priority than all other 
                if highScore.arrivalTime<=node.time:
                    print("highScore.arrivalTime<=node.time",file=myFile)
                    return highScore
                
            for highScore in sortedByScore: #iterate itrough list of task ordered in high priority
                
                if((weighted_score(normalTasks,highScore, cpu_weight, deadline_weight)<weighted_score(normalTasks,firstNormal, cpu_weight, deadline_weight)) & (highScore.arrivalTime<endTime)):
                    print("(weighted_score(normalTasks,highScore, cpu_weight, deadline_weight)<=weighted_score(normalTasks,firstNormal, cpu_weight, deadline_weight)) & (highScore.arrivalTime<endTime)",file=myFile)
                    timeTillNextHighScore=highScore.arrivalTime-firstNormal.arrivalTime
                    new3Time=timeTillNextHighScore
                    #print("timeTillNextHighScore", timeTillNextHighScore)
                    print("timeTillNextHighScore",file=myFile)
                    print("new3Time",file=myFile)
                    return firstNormal
                
        else:#( (lenN!=0) & lenC!=0)

            print("(lenC!=0)",file=myFile) 
            firstCritical=(node.queueOfCriticals)[0]
            if ((firstCritical.arrivalTime<=node.time) | (firstCritical.arrivalTime<=firstNormal.arrivalTime)):
                print("(firstCritical.arrivalTime<=node.time) | (firstCritical.arrivalTime<=firstNormal.arrivalTime)",file=myFile) 
                
                return firstCritical
            
            else: # for sure a normal must be choosed first
                print("Here")    
                for highScore in sortedByScore:# a high score normal is in queue already: it has higher priority than all other 
                    if highScore.arrivalTime<=node.time:
                        print("highScore.arrivalTime<=node.time",file=myFile)
                        timeTillNextCritical=firstCritical.arrivalTime-highScore.arrivalTime
                        new3Time=timeTillNextCritical 
                        print("Here2")
                        return highScore
                    
                # all normal and critical come after node.time (no task in queue before)
                # also the next task is normal (but critical comes soon and also possible normal with higher priority)
                for highScore in sortedByScore:
                    if((weighted_score(normalTasks,highScore, cpu_weight, deadline_weight)<weighted_score(normalTasks,firstNormal, cpu_weight, deadline_weight)) & (highScore.arrivalTime<endTime)):
                        print("(weighted_score(normalTasks,highScore, cpu_weight, deadline_weight)<=weighted_score(normalTasks,firstNormal, cpu_weight, deadline_weight)) & (highScore.arrivalTime<endTime)",file=myFile)
                        timeTillNextHighScore=highScore.arrivalTime-firstNormal.arrivalTime
                        timeTillNextCritical=firstCritical.arrivalTime-firstNormal.arrivalTime
                        new3Time=min(timeTillNextHighScore,timeTillNextCritical) #wheather next task is a critical or a higher priority normal
                        print("Here3")
                        print("new3Time",file=myFile)
                        print("timeTillNextHighScore",file=myFile)
                        return firstNormal

                # first the highest proiority normal then a critical
                timeTillNextCritical=firstCritical.arrivalTime-firstNormal.arrivalTime
                new3Time=timeTillNextCritical
                
                return firstNormal

        
        
    return "no task"



def criticalFirstExecutingAlgorithm(node,endTime):
    print("#############criticalFirstExecutingAlgorithm###########",file=myFile)
    (node.queueOfCriticals).sort(key=lambda task: task.arrivalTime) #, reverse=True  sort the tasks in queueOfCriticals based on arrival time
    (node.queueOfNormals).sort(key=lambda task: task.arrivalTime) #sort the tasks in queueOfNormals based on arrival time
    lenC=len(node.queueOfCriticals)
    lenN=len(node.queueOfNormals)
    #CFTime=-1 before

    global CFTime
    CFTime=-1

    print("(lenC==0) ",lenC,file=myFile)
    print("(lenN==0) ",lenN,file=myFile)

    if ((lenC==0) & (lenN==0)):
        print("(lenC==0) & (lenN==0)",file=myFile)
        return "no task"
    elif ((lenC!=0) & (lenN==0)):
        print("(lenC!=0) & (lenN==0)",file=myFile)
        taskCritical=(node.queueOfCriticals)[0]
        if taskCritical.arrivalTime<endTime:
            print("taskCritical.arrivalTime<endTime",file=myFile)
            return taskCritical
    elif ((lenC==0) & (lenN!=0)):
        print("(lenC==0) & (lenN!=0)",file=myFile)
        taskNormal=(node.queueOfNormals)[0]
        if taskNormal.arrivalTime<endTime:
            print("taskNormal.arrivalTime<endTime",file=myFile)
            return taskNormal
    elif ((lenC!=0) & (lenN!=0)):
        print("(lenC!=0) & (lenN!=0)",file=myFile)
        taskCritical=(node.queueOfCriticals)[0]
        taskNormal=(node.queueOfNormals)[0]

        #new (if it does not work , here must be removed )
        if (taskNormal.arrivalTime<taskCritical.arrivalTime) & (taskCritical.arrivalTime>node.time):
            print("(taskNormal.arrivalTime<taskCritical.arrivalTime) & (taskCritical.arrivalTime>node.time)",file=myFile)
            if taskNormal.arrivalTime<endTime: ## we need change here :for how much time normal be run?
                print("taskNormal.arrivalTime<endTime",file=myFile)
                startOfNormal=taskNormal.arrivalTime
                if taskNormal.arrivalTime<=node.time:
                    print("taskNormal.arrivalTime<=node.time",file=myFile)
                    startOfNormal=node.time

                timeTillNextCritical=taskCritical.arrivalTime-startOfNormal
                CFTime=timeTillNextCritical
                return taskNormal
            
        return taskCritical
        
        
    return "no task"
    
def simpleRoundRobinExecutingAlgorithm(node,endTime):
    (node.queueOfCriticals).sort(key=lambda task: task.arrivalTime) #, reverse=True  sort the tasks in queueOfCriticals based on arrival time
    (node.queueOfNormals).sort(key=lambda task: task.arrivalTime) #sort the tasks in queueOfNormals based on arrival time
    lenC=len(node.queueOfCriticals)
    lenN=len(node.queueOfNormals)
    global simpleRRTurnF
    if ((lenC==0) & (lenN==0)):
        return "no task"
    elif ((lenC!=0) & (lenN==0)):
        taskCritical=(node.queueOfCriticals)[0]
        if taskCritical.arrivalTime<endTime:
            simpleRRTurnF=="normal"
            return taskCritical
    elif ((lenC==0) & (lenN!=0)):
        taskNormal=(node.queueOfNormals)[0]
        if taskNormal.arrivalTime<endTime:
            simpleRRTurnF=="critical"
            return taskNormal
    elif ((lenC!=0) & (lenN!=0)):
        taskCritical=(node.queueOfCriticals)[0]
        taskNormal=(node.queueOfNormals)[0]
        if simpleRRTurnF=="critical": ##its time to execute critical Task
            if taskCritical.arrivalTime<=node.time:
                simpleRRTurnF=="normal"
                return taskCritical
            elif taskNormal.arrivalTime<=node.time:
                simpleRRTurnF=="critical"
                return taskNormal
            else:
                if taskCritical.arrivalTime<=taskNormal.arrivalTime:
                   simpleRRTurnF=="normal"
                   return taskCritical
                else:
                    simpleRRTurnF=="critical"
                    return taskNormal
        elif simpleRRTurnF=="normal": ##its time to execute normal Task 
            if taskNormal.arrivalTime<=node.time:
                simpleRRTurnF=="critical"
                return taskNormal
            elif taskCritical.arrivalTime<=node.time:
                simpleRRTurnF=="normal"
                return taskCritical
            else:
                if taskNormal.arrivalTime<=taskCritical.arrivalTime:
                    simpleRRTurnF=="critical"
                    return taskNormal
                else:
                    simpleRRTurnF=="normal"
                    return taskCritical
    print ("++++ no task outside of condotions")
    return "no task"

def weightedRoundRobinExecutingAlgorithm(node,endTime):
    (node.queueOfCriticals).sort(key=lambda task: task.arrivalTime) #, reverse=True  sort the tasks in queueOfCriticals based on arrival time
    (node.queueOfNormals).sort(key=lambda task: task.arrivalTime) #sort the tasks in queueOfNormals based on arrival time
    lenC=len(node.queueOfCriticals)
    lenN=len(node.queueOfNormals)
    global criticalUnit ##this 2 need to be changed during simulations
    global normalUnit
    print(" criticalUnit ",criticalUnit)
    print(" normalUnit ",normalUnit)
    if ((lenC==0) & (lenN==0)):
        return "no task"
    elif ((lenC!=0) & (lenN==0)):
        taskCritical=(node.queueOfCriticals)[0]
        if taskCritical.arrivalTime<endTime:
            criticalUnit=weigthedRRUnit ##turn critical unit into base unit
            normalUnit+=(aNormal*weigthedRRUnit)
            return taskCritical
    elif ((lenC==0) & (lenN!=0)):
        taskNormal=(node.queueOfNormals)[0]
        if taskNormal.arrivalTime<endTime:
            normalUnit=weigthedRRUnit*aNormal ##turn normal unit into base unit
            criticalUnit+=weigthedRRUnit
            return taskNormal   
    elif ((lenC!=0) & (lenN!=0)): 
        taskCritical=(node.queueOfCriticals)[0]
        taskNormal=(node.queueOfNormals)[0]
        criticalcpu=taskCritical.cpu
        normalcpu=taskNormal.cpu
        if ((taskCritical.arrivalTime >= node.time) & (taskNormal.arrivalTime >taskCritical.arrivalTime)):
            criticalUnit=weigthedRRUnit ##turn critical unit into base unit
            normalUnit+=(aNormal*weigthedRRUnit)
            return taskCritical
        elif ((taskNormal.arrivalTime >= node.time) & (taskNormal.arrivalTime <taskCritical.arrivalTime)):
            normalUnit=weigthedRRUnit*aNormal ##turn normal unit into base unit
            criticalUnit+=weigthedRRUnit
            return taskNormal
        else:  
            while True:
                if normalcpu<=normalUnit:
                    normalUnit=aNormal*weigthedRRUnit
                    return taskNormal
                elif criticalcpu<=criticalUnit:
                    criticalUnit=weigthedRRUnit ##change to base amount
                    return taskCritical
                normalUnit+=(aNormal*weigthedRRUnit)
                criticalUnit+=weigthedRRUnit
    return "no task"

def NormalAlgorithmLocal(task,manager,colonyList,slot):
    dst=task.src
    return dst

def NormalRandomAlgorithmNeighbor(task,manager,colonyList,slot):## return a random node in its colony
    srcNode=task.src
    srcColony=srcNode.parent
    dst=srcColony.nodeManager
    while dst.id==(srcColony.nodeManager).id:
        options=srcColony.listOfNodes
        dstIndex=random.randint(0,len(options)-1)
        dst=options[dstIndex]
    return dst


def updateNodeOutterTable(listOfMans,LOC): ##in each manager outterTable we have a key that is another manager with 3 different value of critical & Normal total used of cpu of last slot and free cpu capacity(that is total of nodes in their's colony)
    for manager in listOfMans:
        totalFreeCapacity=0
        totalNormalCpuUsed=0
        totalCriticalUsed=0
        totalNormalCapacity=0
        for key in manager.innerTable:
            if key[1]=="freeCapacity":
                totalFreeCapacity+=(manager.innerTable[key])
            elif key[1]=="normalCpuUsed":
                totalNormalCpuUsed+=(manager.innerTable[key])
            elif key[1]=="criticalCpuUsed":
                totalCriticalUsed+=(manager.innerTable[key])
            elif key[1]=="normalCapacity":
                totalNormalCapacity+=(manager.innerTable[key])
        for man in listOfMans:
            if man==manager:
                continue
            man.outterTable[manager,"freeCapacity"]=totalFreeCapacity
            man.outterTable[manager,"normalCpuUsed"]=totalNormalCpuUsed
            man.outterTable[manager,"criticalCpuUsed"]=totalCriticalUsed
            man.outterTable[manager,"normalCapacity"]=totalNormalCapacity

    resetUsedCPU(LOC)
        
def resetUsedCPU(LOC):## reset cpu used in last subslot
    for node in (LOC[0].listOfComputingNodes):
        node.criticalRun=0 
        node.normalRun=0
        node.freeCpu=0  
        node.normalCapacity=0


def updateNodeInnerTable(node):
    if "cloud" in node.name: ##??? if dst has chooden as cloud ...
        manager=node #?? xx XX here we are updating managers of each colony but cloud has no manager !! 
        return # XX ?? xx
    else:
        manager=(node.parent).nodeManager

    #totalCpu=node.cpu # its for whole timeslot not subslot
    totalCpu=math.ceil(node.cpu* Coefficient) ## for paper # calculate total cpu in a subslot

    criticalUsed=node.criticalRun
    normalUsed=node.normalRun
    #freeCpu=totalCpu-criticalUsed-normalUsed
    freeCpu=totalCpu-criticalUsed # for paper

    thr=node.cpu*freeCpuThreshold # threshold 
    normalCapacity=totalCpu-criticalUsed-thr ## for propossed Alg

    if normalCapacity<=0:
        normalCapacity=0

    manager.innerTable[node,"criticalCpuUsed"]=criticalUsed
    manager.innerTable[node,"normalCpuUsed"]=normalUsed
    #manager.innerTable[node,"freeCapacity"]+=freeCpu
    manager.innerTable[node,"freeCapacity"]=freeCpu # paper

    manager.innerTable[node,"normalCapacity"]+=normalCapacity # it is used for the proposed alg


def deadlineSatisfaction(manager,task,now):
    listOfNodesInOurColony=[]
    ##find computing nodes in our colony
    for item in manager.innerTable.keys():
        listOfNodesInOurColony.append(item[0])
    listOfNodesInOurColony = list(dict.fromkeys(listOfNodesInOurColony)) ## delete repatitive nodes in the list     
    listOfNodesInOurColony.sort(key=lambda task: task.cpu) ## sort nodes in colony based on cpu power
    src=task.src
    exeDelay=0
    comDealy=0
    queuDelay=0
    taskArrivalTime=0
    print("/// in our colony")
    for node in listOfNodesInOurColony: ##need to be sort based on cpu power ????
        if node==task.src: ## error handling:we checked the src as dst, in the previous step
            continue
        comDealy=communicationDelayComputer(task,src,node,task.insize)
        exeDelay=(task.cpu/node.cpu)*1000 ## xx XX
        taskArrivalTime=comDealy+now
        queuDelay=(futureQueueComputer(node,taskArrivalTime,now)*1000)
        totalDelay=comDealy+queuDelay+exeDelay
        print("testing for dst ...",node.name)
        print("comDealy ",comDealy)
        print("queuDelay ",queuDelay)
        print("exeDelay ",exeDelay)
        print("now+totalDelay ",now+totalDelay)
        print("task.generationTime+task.deadline ",task.generationTime+task.deadline)
        if (task.generationTime+task.deadline)>=(now+totalDelay): ##can be done before deadline
            print("CHOOOOOSE")
            return node
    return -1 ## no node meet the deadline



def findDstInOurColony(manager,task,now):##choose the alg that we want to find dst in our colony
    #dst=WeightedRandomInOurColony(manager,task)
    dst=deadlineSatisfaction(manager,task,now)
    return dst


def WeightedRandomInOurColony(manager,task): ##to find the best node in colony to offload
    ## weighted random based on free cpu capacity of last slot
    myOptions=[] # save those nodes that have free capacity more than task requirement
    freeCapacity=[]
    for item in manager.innerTable.keys():
        if item[1]=="freeCapacity":
            if manager.innerTable[item]>=task.cpu:
                myOptions.append(item[0])
                freeCapacity.append(manager.innerTable[item])
    if len(myOptions)==0: ##if no node in colony could process the normal task
        dst="null"
        return dst
    #print(myOptions)
    #print(probilality)
    total=np.sum(freeCapacity)
    probibality=[x/total for x in freeCapacity] ## each node gets a percetage of probibality based on free capacity
    #print(probibality)
    #print(np.sum(probibality))
    sampleList = myOptions
    dst = choice(
    sampleList, 1, p=probibality)
    #print(dst)
    return dst

def findDstInOtherColonies(manager,task):
    ...

def currentQueueComputer(queue,now):## compute the current workload of a node before assigning new tasks since curent subslot
    print("*****in currentQueueComputer",file=myFile)
    print("tasks in the queue waiting for process ... ",file=myFile)
    totalCpu=0

    for task in queue:# +dst.queueOfCriticals
        #if task.finished==False:
        if ( (task.schedulingTime<now) & (task.finished==False)):
            totalCpu+=(task.cpu-task.doneCPU)
            print("in the queue normal task ",task.id,file=myFile)
            print("scheduling Time ",task.schedulingTime,file=myFile)
            print("arriving time in the normal queue ",task.arrivalTime,file=myFile)

    currentQueueLoad=totalCpu
    return currentQueueLoad


def futureQueueComputer(queue,now): ## compute the workload of a node with new tasks that have been assigned to that node, in the current timeslot
    totalCpu=0
    print("*****in futureQueueComputer",file=myFile)
    print("tasks in the queue waiting for process ... ",file=myFile)
    for task in queue:# +dst.queueOfCriticals
        if ( task.schedulingTime>=now):
            #totalCpu+=(task.cpu) xx XX before was this
            totalCpu+=(task.cpu-task.doneCPU)
            print(task.id,file=myFile)
            print("scheduling Time ",task.schedulingTime,file=myFile)
            #print("task cpu added ",task.id)
            #print("gen ",task.generationTime)
            #time.sleep(16)
    #print("*****")
    futureQueueLoad=totalCpu
    return futureQueueLoad

def checkTheConditionNew1(task,dst,manager,now,nowInSlotLook):## check allocating a task to a node wheather is possible or not (check task.cpu+node.load <=node.freeCPU)
    print("In Check Condition New1",file=myFile)
    print("",file=myFile) 

    src=task.src
    freeCPU=manager.innerTable[dst,"freeCapacity"] # free Cpu from the last slot =(cpu-criticalused)
    ## we need a normal capacity from now till end of time slot
    
    taskCpuRequired=task.cpu 
    futureQueueLoad=futureQueueComputer(dst.queueOfNormals,now) # paper:nowInSlotLook before was based on MI ##?? check here !!
    currentQueueLoad=currentQueueComputer(dst.queueOfNormals,now)
    
    #print("Now is ",now,file=myFile)
    print("task.id ",task.id,file=myFile)
    print("src ",src.name,file=myFile)
    print("task gen is ",task.generationTime,file=myFile)
    print("checking possible dst ",dst.name,file=myFile)
    print("freeCPU ",freeCPU,file=myFile)
    print("futureQueueLoad ",futureQueueLoad,file=myFile)
    print("currentQueueLoad ",currentQueueLoad,file=myFile)
    print("taskCpuRequired ",taskCpuRequired,file=myFile)
    print("-----------------------------")
    #time.sleep(17)
    print("freeCPU>=futureQueueLoad+currentQueueLoad ",freeCPU>=futureQueueLoad+currentQueueLoad,file=myFile)

    if ((freeCPU>=futureQueueLoad+currentQueueLoad)): # in this approach we dont consider task.cpu 
        print("dst choosed  ",dst.name,file=myFile)
        task.dst=dst
        forward(task,now,{},{}) ## before forward(task,now,{},{})

def checkTheConditionNew3(task,dst,manager,now,nowInSlotLook):## check allocating a task to a node wheather is possible or not (check task.cpu+node.load <=node.freeCPU)
    print("In Check Condition New3",file=myFile)
    print("",file=myFile) 

    src=task.src
    freeCPU=manager.innerTable[dst,"freeCapacity"] # free Cpu from the last slot =(cpu-criticalused)
    ## we need a normal capacity from now till end of time slot
    
    taskCpuRequired=task.cpu 
    futureQueueLoad=futureQueueComputer(dst.queueOfNormals,now) # paper:nowInSlotLook before was based on MI ##?? check here !!
    currentQueueLoad=currentQueueComputer(dst.queueOfNormals,now)
    
    #print("Now is ",now,file=myFile)
    print("task.id ",task.id,file=myFile)
    print("src ",src.name,file=myFile)
    print("task gen is ",task.generationTime,file=myFile)
    print("checking possible dst ",dst.name,file=myFile)
    print("freeCPU ",freeCPU,file=myFile)
    print("futureQueueLoad ",futureQueueLoad,file=myFile)
    print("currentQueueLoad ",currentQueueLoad,file=myFile)
    print("taskCpuRequired ",taskCpuRequired,file=myFile)
    print("-----------------------------")
    #time.sleep(17)
    print("freeCPU>=futureQueueLoad+currentQueueLoad ",freeCPU>=futureQueueLoad+currentQueueLoad,file=myFile)

    if ((freeCPU>=futureQueueLoad+currentQueueLoad)): # in this approach we dont consider task.cpu 
        print("dst choosed  ",dst.name,file=myFile)
        task.dst=dst
        forward(task,now,{},{}) ## before forward(task,now,{},{})


def checkTheConditionNew2(task,dst,manager,now,nowInSlotLook):## new proposed alg
    #checking wheather a task can be completed within next 2 sub slot or not (with a deadline violation limit)
    #considering task.cpu, node.cpu*2, criticalCpuUsed*2 + Load of normals
    print("In Check Condition New2",file=myFile)
    print("",file=myFile) 

    src=task.src
    taskCpuRequired=task.cpu 

    totalCpu=(dst.cpu*Coefficient)*2  # amout of total dst.cpu  for next 2 slot
    criticalNeed=manager.innerTable[dst,"criticalCpuUsed"]*2 # predicted cpu usage of critical tasks till for next 2 subslot

    
    ## load of normal tasks in the queue already
    futureQueueLoad=futureQueueComputer(dst.queueOfNormals,now) 
    currentQueueLoad=currentQueueComputer(dst.queueOfNormals,now)
    totalNormalsLoad=currentQueueLoad+futureQueueLoad #total work load in node.queueOfNormals

    freeCpu=totalCpu-criticalNeed-totalNormalsLoad #cpu availibility for next 2 subslot

    transDelay=0
    propDelay=0
    networkDelay=0
    step=0


    if task.src!=dst: # when src and dst are different the network delay is not 0 anymore
        #network delay
        col=manager.parent
        step=len(col.routingTable[((task.src).name,dst.name)])
        #step=(step_distance_matrix [(task.src,dst)])
        print("(task.insize+task.outsize) ", (task.insize+task.outsize))
        print("avg_bw ", avg_bw)
        transDelay=( step* ( (((task.insize+task.outsize)*8)*pow(10,3))/(avg_bw*pow(10,6)) )) *1000 #ms
        propDelay=step * avg_propDelay #ms
        networkDelay=transDelay+propDelay
    
    '''if transDelay>0:
        print("task.id ",task.id)
        print("src ",src.name)
        print("checking possible dst ",dst.name)
        print("step ",step)
        print("transDelay ",transDelay)
        time.sleep(12)  '''  

    finishTime=((task.cpu/freeCpu)*subSlot)+networkDelay # ms

    violation_limit=1.2 # maximum violation limit

    remained_deadline=task.deadline-(now-task.generationTime)

    
    #print("Now is ",now,file=myFile)
    print("task.id ",task.id,file=myFile)
    print("src ",src.name,file=myFile)
    print("checking possible dst ",dst.name,file=myFile)
    print("freeCpu ",freeCpu,file=myFile)
    print("task gen is ",task.generationTime,file=myFile)
    print("criticalNeed ",criticalNeed,file=myFile)
    print("futureQueueLoad ",futureQueueLoad,file=myFile)
    print("currentQueueLoad ",currentQueueLoad,file=myFile)
    print("taskCpuRequired ",taskCpuRequired,file=myFile)
    print("transDelay ",transDelay,file=myFile)
    print("networkDelay ",networkDelay,file=myFile)
    print("finishTime ",finishTime,file=myFile)
    print("deadline ",task.deadline,file=myFile)
    print("remained_deadline ",remained_deadline,file=myFile)


    

    
    #time.sleep(17)
    

    if ((finishTime<=remained_deadline*violation_limit)): # in this approach we dont consider task.cpu 
        print("dst choosed  ",dst.name,file=myFile)
        task.dst=dst
        forward(task,now,{},{}) ## before forward(task,now,{},{})


def checkTheCondition(task,dst,manager,now,nowInSlotLook):
    print("In Check Condition",file=myFile)
    print("",file=myFile) 
    #time.sleep(2)
    src=task.src
    UN=manager.innerTable[dst,"normalCpuUsed"] # Cpu utilization for normal tasks in the last slot
    UC=manager.innerTable[dst,"criticalCpuUsed"]
    normalCapacity=manager.innerTable[dst,"normalCapacity"] ## ?? its for a complete time slot

    freeCPU=manager.innerTable[dst,"freeCapacity"] 
    ## we need a normal capacity from now till end of time slot
    taskCpuRequired=task.cpu 
    QueueLoad=futureQueueComputer(dst,nowInSlotLook) #based on MI ##?? check here !!
    #threshold=((dst.cpu*freeCpuThreshold)/100)*coSlot ## its a global variable that is 20 percent now
    #total=UN+UC+taskCpuRequired+QueueLoad+threshold

    
    #print("Now is ",now,file=myFile)
    print("src ",src.name,file=myFile)
    print("task.id ",task.id,file=myFile)
    print("task gen is ",task.generationTime,file=myFile)
    print("dst ",dst.name,file=myFile)
    print("UN ",UN,file=myFile)
    print("UC ",UC,file=myFile)
    print("QueueLoad ",QueueLoad,file=myFile)
    print("taskCpuRequired ",taskCpuRequired,file=myFile)
    print("normalCapacity ",normalCapacity,file=myFile)
    #print("threshold ",threshold,file=myFile)
    #print("total ",total,file=myFile)
    #print(total<=dst.cpu*coSlot,file=myFile)
    print("-----------------------------")
    #time.sleep(17)
    if ((normalCapacity>=taskCpuRequired+QueueLoad)): 
        task.dst=dst
        forward(task,now,{},{}) ## before forward(task,now,{},{})


def new1FirstPhase(tasks,manager,now,nowInSlotLook):##check wheather normal tasks can be done in thier src 
    print("*********new1 First phase .....",file=myFile)
    print("now ",now,file=myFile)
    print("",file=myFile)
    #time.sleep(2)
    for task in tasks:
        print("task.id ",task.id,file=myFile)
        print("task.generationTime",task.generationTime,file=myFile)
        print("task.scheduled ",task.scheduled,file=myFile)
        #time.sleep(3)
        if ((task.generationTime<=now) & (task.scheduled==False) & (task.dst==-1)):
            dst=task.src
            checkTheConditionNew1(task,dst,manager,now,nowInSlotLook)## check the Eq of our New1 proposed Alg


def new2FirstPhase(tasks,manager,now,nowInSlotLook):##check wheather normal tasks can be done in thier src 
    print("*********new2 First phase .....",file=myFile)
    print("now ",now,file=myFile)
    print("",file=myFile)
    #time.sleep(2)
    for task in tasks:
        print("task.id ",task.id,file=myFile)
        print("task.generationTime",task.generationTime,file=myFile)
        print("task.scheduled ",task.scheduled,file=myFile)
        #time.sleep(3)
        if ((task.generationTime<=now) & (task.scheduled==False) & (task.dst==-1)):
            dst=task.src
            checkTheConditionNew2(task,dst,manager,now,nowInSlotLook)## check the Eq of our New2 proposed Alg


def firstPhase(tasks,manager,now,nowInSlotLook):##check wheather normal tasks can be done in thier src 
    print("*********First phase .....",file=myFile)
    print("now ",now,file=myFile)
    print("",file=myFile)
    #time.sleep(2)
    for task in tasks:
        print("task.id ",task.id,file=myFile)
        print("task.generationTime",task.generationTime,file=myFile)
        print("task.scheduled ",task.scheduled,file=myFile)
        #time.sleep(3)
        if ((task.generationTime<=now) & (task.scheduled==False) &(task.dst==-1)):
            dst=task.src
            checkTheCondition(task,dst,manager,now,nowInSlotLook)## check the Eq of our proposed Alg


def new1SecondPhase(tasks,manager,now,nowInSlotLook):# find a dst for each task in its colony (or cloud) by considering: 
    # first order  nodes based on free cpu
    # sort tasks based on remained deadline
    # assign tasks to those node who has a free cpu
    # if a task has no option among nodes (none of them has free cpu) -> it will be send to the cloud

    scheduled=[] # for storing scheduled normal tasks by manager (In phase1&phase2)

    print("new1 SECOND PHASE",file=myFile)
    #time.sleep(2)
    srcColony=manager.parent
    
    nodes = set(node for (node, attr) in manager.innerTable.keys() if attr == "freeCapacity")
    node_freeCpu_pairs = [(node, manager.innerTable[(node, "freeCapacity")]) for node in nodes]

    # Sort nodes based on cpu values (ascending order)
    sorted_nodes = sorted(node_freeCpu_pairs, key=lambda x: x[1], reverse=True)  #(node, freeCapacity is in this list as a tuple)

    # sort nodes based on free cpu and save them in a list (only name of nodes is saved)
    sorted_nodes_only = [node for (node, cpu) in sorted_nodes] 

    # sort tasks in manager's buffer based on genTime+deadline
    sorted_tasks = sorted(tasks, key=lambda task: task.generationTime + task.deadline)

    for task in sorted_tasks:
        if (task.dst !=-1):# if in first phase dst is choosen already, pass this steps
            scheduled.append(task) ##paper: scheduled normal tasks in phase 1
            continue 
        #print("now ",now,file=myFile)
        print("task.id ",task.id,file=myFile)
        print("task.generationTime ",task.generationTime,file=myFile)
        print("task.deadline ",task.deadline,file=myFile)
        print("task.generationTime + task.deadline ",task.generationTime + task.deadline,file=myFile)
        print("(task.generationTime<=now) & (task.scheduled==False) ",(task.generationTime<=now) & (task.scheduled==False),file=myFile)
        
        if ((task.generationTime<=now) & (task.scheduled==False)):
            #if (task.dst !=-1):# if in first phase dst is choosen already, pass this steps
                #continue : paper
            for node in sorted_nodes_only: # iterate into sorted list of nodes based on free Cpu (the highest capacity is in the top)
                #if task.dst==-1:## xx XX this condition added recently (for not sending a normal to all nodes that are a good fit)
                    #checkTheConditionNew1(task,node,manager,now,nowInSlotLook)## check the Eq of our New1 proposed Alg
                #paper: above 2 lines replaced with below
                if(task.dst ==-1): # avoid checking when the first candidate node is suitable
                    checkTheConditionNew1(task,node,manager,now,nowInSlotLook)## check the Eq of our New1 proposed Alg

            if task.dst==-1: ## if a task can not be scheduled with last phases will be send to cloud
                task.dst=srcColony.listOfComputingNodes[0] ## to cloud
                print("the only option remained is ",(task.dst).name,file=myFile)
                print("sending the task to cloud ...",file=myFile)
                forward(task,now,{},{})
                print("task arrives in cloud at ",task.arrivalTime,file=myFile)

            print("task.id ",task.id,file=myFile)
            print("src ",(task.src).name,file=myFile)
            print("dst ",(task.dst).name,file=myFile)

            scheduled.append(task) #paper: scheduled normal tasks in phase 2

    for scheduledTasks in scheduled: #paper: delete scheduled tasks from manager's buffer 
        (manager.buffer).remove(scheduledTasks)        
    scheduled.clear()

def new2SecondPhase(tasks,manager,now,nowInSlotLook):# find a dst for each task in its colony (or cloud) by considering: 
    # first order  nodes based on free cpu
    # sort tasks based on size/remained deadline (decreasing)
    # assign tasks to those node who has a free cpu
    # if a task has no option among nodes (none of them has free cpu) -> it will be send to the cloud

    scheduled=[] # for storing scheduled normal tasks by manager (In phase1&phase2)

    print("new2 SECOND PHASE",file=myFile)
    
    #time.sleep(2)
    srcColony=manager.parent
    
    nodes = set(node for (node, attr) in manager.innerTable.keys() if attr == "freeCapacity")
    node_freeCpu_pairs = [(node, manager.innerTable[(node, "freeCapacity")]) for node in nodes]

    # Sort nodes based on cpu values (ascending order)
    sorted_nodes = sorted(node_freeCpu_pairs, key=lambda x: x[1], reverse=True)  #(node, freeCapacity is in this list as a tuple)

    # sort nodes based on free cpu and save them in a list (only name of nodes is saved)
    sorted_nodes_only = [node for (node, cpu) in sorted_nodes] 

    # sort tasks in manager's buffer based on size/remained deadline
    sorted_tasks = sorted(tasks, key=lambda task: (task.cpu-task.doneCPU)/(task.deadline-(now-task.generationTime)), reverse=True)


    for task in sorted_tasks:
        if (task.dst !=-1):# if in first phase dst is choosen already, pass this steps
            scheduled.append(task) ##paper: scheduled normal tasks in phase 1
            continue 
        #print("now ",now,file=myFile)
        print("task.id ",task.id,file=myFile)
        print("task.generationTime ",task.generationTime,file=myFile)
        print("task.deadline ",task.deadline,file=myFile)
        print("task.generationTime + task.deadline ",task.generationTime + task.deadline,file=myFile)
        print("(task.generationTime<=now) & (task.scheduled==False) ",(task.generationTime<=now) & (task.scheduled==False),file=myFile)
        
        if ((task.generationTime<=now) & (task.scheduled==False)):
            #if (task.dst !=-1):# if in first phase dst is choosen already, pass this steps
                #continue : paper
            for node in sorted_nodes_only: # iterate into sorted list of nodes based on free Cpu (the highest capacity is in the top)
                if node==task.src: # since we check the src at first phase, now we ignore task's src as dst
                    continue
                #paper: above 2 lines replaced with below
                if(task.dst ==-1): # avoid checking when the first candidate node is suitable
                    print("src ",(task.src).name)
                    print("dst ",node.name)
                    checkTheConditionNew2(task,node,manager,now,nowInSlotLook)## check the Eq of our New1 proposed Alg

            if task.dst==-1: ## if a task can not be scheduled with last phases will be send to cloud
                task.dst=srcColony.listOfComputingNodes[0] ## to cloud
                print("the only option remained is ",(task.dst).name,file=myFile)
                print("sending the task to cloud ...",file=myFile)
                forward(task,now,{},{})
                print("task arrives in cloud at ",task.arrivalTime,file=myFile)

            print("task.id ",task.id,file=myFile)
            print("src ",(task.src).name,file=myFile)
            print("dst ",(task.dst).name,file=myFile)

            scheduled.append(task) #paper: scheduled normal tasks in phase 2

    for scheduledTasks in scheduled: #paper: delete scheduled tasks from manager's buffer 
        (manager.buffer).remove(scheduledTasks)        
    scheduled.clear()

def new3SecondPhase(tasks,manager,now,nowInSlotLook):# find a dst for each task in its colony (or cloud) by considering: 
    # first order  nodes based on free cpu
    # sort tasks based on remained deadline
    # assign tasks to those node who has a free cpu
    # if a task has no option among nodes (none of them has free cpu) -> it will be send to the cloud

    scheduled=[] # for storing scheduled normal tasks by manager (In phase1&phase2)

    print("new3 SECOND PHASE",file=myFile)
    #time.sleep(2)
    srcColony=manager.parent
    
    nodes = set(node for (node, attr) in manager.innerTable.keys() if attr == "freeCapacity")
    node_freeCpu_pairs = [(node, manager.innerTable[(node, "freeCapacity")]) for node in nodes]

    # Sort nodes based on cpu values (ascending order)
    sorted_nodes = sorted(node_freeCpu_pairs, key=lambda x: x[1], reverse=True)  #(node, freeCapacity is in this list as a tuple)

    # sort nodes based on free cpu and save them in a list (only name of nodes is saved)
    sorted_nodes_only = [node for (node, cpu) in sorted_nodes] 

    # sort tasks in manager's buffer based on genTime+deadline
    sorted_tasks = sorted(tasks, key=lambda task: task.generationTime + task.deadline)

    for task in sorted_tasks:
        if (task.dst !=-1):# if in first phase dst is choosen already, pass this steps
            scheduled.append(task) ##paper: scheduled normal tasks in phase 1
            continue 
        #print("now ",now,file=myFile)
        print("task.id ",task.id,file=myFile)
        print("task.generationTime ",task.generationTime,file=myFile)
        print("task.deadline ",task.deadline,file=myFile)
        print("task.generationTime + task.deadline ",task.generationTime + task.deadline,file=myFile)
        print("(task.generationTime<=now) & (task.scheduled==False) ",(task.generationTime<=now) & (task.scheduled==False),file=myFile)
        
        if ((task.generationTime<=now) & (task.scheduled==False)):
            #if (task.dst !=-1):# if in first phase dst is choosen already, pass this steps
                #continue : paper
            for node in sorted_nodes_only: # iterate into sorted list of nodes based on free Cpu (the highest capacity is in the top)
                #if task.dst==-1:## xx XX this condition added recently (for not sending a normal to all nodes that are a good fit)
                    #checkTheConditionNew1(task,node,manager,now,nowInSlotLook)## check the Eq of our New1 proposed Alg
                #paper: above 2 lines replaced with below
                if(task.dst ==-1): # avoid checking when the first candidate node is suitable
                    checkTheConditionNew3(task,node,manager,now,nowInSlotLook)## check the Eq of our New1 proposed Alg

            if task.dst==-1: ## if a task can not be scheduled with last phases will be send to cloud
                task.dst=srcColony.listOfComputingNodes[0] ## to cloud
                print("the only option remained is ",(task.dst).name,file=myFile)
                print("sending the task to cloud ...",file=myFile)
                forward(task,now,{},{})
                print("task arrives in cloud at ",task.arrivalTime,file=myFile)


            print("task.id ",task.id,file=myFile)
            print("src ",(task.src).name,file=myFile)
            print("dst ",(task.dst).name,file=myFile)

            scheduled.append(task) #paper: scheduled normal tasks in phase 2

    for scheduledTasks in scheduled: #paper: delete scheduled tasks from manager's buffer 
        (manager.buffer).remove(scheduledTasks)        
    scheduled.clear()


def secondPhase(tasks,manager,now,nowInSlotLook):# find a dst for each task in its colony (or cloud)by considering: 
    # first order nodes with minimum delay with src
    # check the DEFINED CONDITION for each task on those nodes
    print("SECOND PHASE",file=myFile)
    #time.sleep(2)
    listOfNodesInOurColony=[]
    srcColony=manager.parent
    #srcColony.routingTable[(start_vertex.name,vertex.name)]
    ##find computing nodes in our colony
    for item in manager.innerTable.keys():
        listOfNodesInOurColony.append(item[0])
    listOfNodesInOurColony = list(dict.fromkeys(listOfNodesInOurColony)) ## delete repatitive nodes in the list 
    listOfNodesInOurColony.sort(key=lambda node: node.cpu) ## sort nodes in colony based on 

    for task in tasks:
        print("now ",now,file=myFile)
        print("task.id ",task.id,file=myFile)
        print("task.generationTime ",task.generationTime,file=myFile)
        print("task.scheduled ",task.scheduled,file=myFile)
        #time.sleep(5)
        if ((task.generationTime<=now) & (task.scheduled==False)):
            if (task.dst !=-1):# if in first phase dst is choosen already, pass this steps
                continue 

            for node in listOfNodesInOurColony:
                if task.dst==-1:## xx XX this condition added recently (for not sending a normal to all nodes that are a good fit)
                    checkTheCondition(task,node,manager,now,nowInSlotLook)## check the Eq of our proposed Alg
            print("task.id ",task.id,file=myFile)
            print("src ",(task.src).name,file=myFile)
            if task.dst==-1: ## if a task can not be scheduled with last phases will be send to src
                #task.dst=srcColony.listOfComputingNodes[0] ## to cloud
                #task.dst=task.src # xx XX ?? 
                task.dst=last_chance(manager) #?? xx XX here must be checked in the function
                #print("sending the task to cloud ...",file=myFile)
                forward(task,now,{},{})
            print("dst ",(task.dst).name,file=myFile)
            #time.sleep(1)


def last_chance(manager):#?? xx XX when in first and second phase dst is not assigned
    
    possibbleDSTs={}
    for key in manager.innerTable:
        if key[1]=="normalCapacity":
            possibbleDSTs[key[0]] = manager.innerTable[key]

    b = sorted(possibbleDSTs.items(), key=lambda x: x[1], reverse=True)  # sort neighbors baed on free cpu
    
    n=(b)[0][0] ## the node with the largest "normal capacity "
    normalCap=(b)[0][1] ## the amount of normal capacity

    return n

def New1NormalProposedAlgorithm(tasks,manager,slot):##find a suitable dst for normal task by the manager based on status of inner colony nodes
    print("In New1 NormalProposedAlgorithm",file=myFile)
    print("len of managers'buffer (normal tasks waiting for scheduling) ",len(tasks),file=myFile)
    print("",file=myFile)
    now=slot*subSlot
    nowInSlotLook=int(slot/coSlot)*timeSlot #?? for computing queue from begening of main slot
    print("nowInSlotLook(= int(slot/coSlot)*timeSlot) :",nowInSlotLook,file=myFile)

    new1FirstPhase(tasks,manager,now,nowInSlotLook) ##check src as dst ?
    new1SecondPhase(tasks,manager,now,nowInSlotLook) # check normal tasks in thier colony then cloud

def New3NormalProposedAlgorithm(tasks,manager,slot): ##same as new1
    print("In New3 NormalProposedAlgorithm",file=myFile)
    print("len of managers'buffer (normal tasks waiting for scheduling) ",len(tasks),file=myFile)
    print("",file=myFile)
    now=slot*subSlot
    nowInSlotLook=int(slot/coSlot)*timeSlot #?? for computing queue from begening of main slot
    print("nowInSlotLook(= int(slot/coSlot)*timeSlot) :",nowInSlotLook,file=myFile)

    new3FirstPhase(tasks,manager,now,nowInSlotLook) ##check src as dst ?
    new3SecondPhase(tasks,manager,now,nowInSlotLook) # check normal tasks in thier colony then cloud


def new3FirstPhase(tasks,manager,now,nowInSlotLook):##check wheather normal tasks can be done in thier src 
    print("*********new3 First phase .....",file=myFile)
    print("now ",now,file=myFile)
    print("",file=myFile)
    #time.sleep(2)
    for task in tasks:
        print("task.id ",task.id,file=myFile)
        print("task.generationTime",task.generationTime,file=myFile)
        print("task.scheduled ",task.scheduled,file=myFile)
        #time.sleep(3)
        if ((task.generationTime<=now) & (task.scheduled==False) & (task.dst==-1)):
            dst=task.src
            checkTheConditionNew3(task,dst,manager,now,nowInSlotLook)## check the Eq of our New1 proposed Alg


def New2NormalProposedAlgorithm(tasks,manager,slot): ## check the violation based on free cpu capacity (availability) for next 2 subslot
    print("In New2 NormalProposedAlgorithm",file=myFile)
    print("len of managers'buffer (normal tasks waiting for scheduling) ",len(tasks),file=myFile)
    print("",file=myFile)
    now=slot*subSlot
    nowInSlotLook=int(slot/coSlot)*timeSlot #?? for computing queue from begening of main slot
    print("nowInSlotLook(= int(slot/coSlot)*timeSlot) :",nowInSlotLook,file=myFile)

    new2FirstPhase(tasks,manager,now,nowInSlotLook) ##check src as dst 
    new2SecondPhase(tasks,manager,now,nowInSlotLook) # check normal tasks in thier colony then cloud


def NormalProposedAlgorithm2(tasks,manager,slot): ##find a suitable dst for normal task by the manager based on status of inner colony nodes
    print("In NormalProposedAlgorithm",file=myFile)
    print("len of normal queue ",len(tasks),file=myFile)
    print("",file=myFile)
    #time.sleep(2)
    now=slot*subSlot
    nowInSlotLook=int(slot/coSlot)*timeSlot # for computing queue from begening of main slot
    firstPhase(tasks,manager,now,nowInSlotLook) ##?? here can be deleted? because we check src as dst before? ##check normal tasks in thier src node
    secondPhase(tasks,manager,now,nowInSlotLook) # check normal tasks in thier colony then cloud

   
			
def NormalProposedAlgorithm(tasks,manager,slot): ##find a suitable dst for normal task by the manager based on status of inner colony nodes
    print("In NormalProposedAlgorithm",file=myFile)
    print("len of normal queue ",len(tasks),file=myFile)
    print("",file=myFile)
    #time.sleep(2)
    now=slot*subSlot
    nowInSlotLook=int(slot/coSlot)*timeSlot # for computing queue from begening of main slot
    firstPhase(tasks,manager,now,nowInSlotLook) ##?? here can be deleted? because we check src as dst before? ##check normal tasks in thier src node
    secondPhase(tasks,manager,now,nowInSlotLook) # check normal tasks in thier colony then cloud

    #for task in tasks: ## here just a routin function for each scheduling alg that we have to send/forward tasks into thier dst
        #forward(task,slot,{},{})
  

def NormalRandomAlgorithmGlobal(task,manager,colonyList,slot): ## return a random node in list of nodes in all colonies except cloud and managers
    srcNode=task.src
    srcColony=srcNode.parent
    options=srcColony.listOfComputingNodes
    dstIndex=random.randint(0,len(options)-1)
    dst=options[dstIndex]
    while dst==cloud:
        dstIndex=random.randint(0,len(options)-1)
        dst=options[dstIndex]
    return dst

def nodeCriticalScheduling(task,node,now,BW_TransDelay1,Cost_ResponseTime1): ##schedule critical task in a generator/src node
    task.dst=node ##need a func later to find suitable dst node
    forward(task,now,BW_TransDelay1,Cost_ResponseTime1)
    #task.schedulingDelay=task.schedulingTime-task.generationTime
    task.scheduled=True

def nodeCriticalSchedulingProposed(task,node,now,BW_TransDelay1,Cost_ResponseTime1):## our scheduling critical tasks :
    # when a task is created, while giving it to the source check wheather it finish on deadline or not
    # reject is pissible 
    futureLoad=futureQueueComputer(node.queueOfCriticals,now)
    currentLoad=currentQueueComputer(node.queueOfCriticals,now)

    load=((currentLoad+futureLoad)/node.cpu)*1000 #ms

    exeTime=(task.cpu/node.cpu)*1000 #ms

    finishTime=exeTime+load # total amount of time the critical task need to be finished

    if (finishTime<=task.deadline): # task can be accepted
        task.dst=node
        forward(task,now,BW_TransDelay1,Cost_ResponseTime1)
        task.scheduled=True
    else:
        print(" task is rejected ",file=myFile)
        print(" task.id ",task.id,file=myFile)
        print(" load ",load,file=myFile)
        task.rejected=True # task is rejected


def managerNormalSchduling(NormalScheduling,managersList,colonyList,slot,BW_TransDelay1,Cost_ResponseTime1):#Forward Task. and set scheduled to True, and the dst of task to the node that is schduled by Algorithm
    print(" In managerNormalSchduling",file=myFile)
    print("",file=myFile)
    now=slot*subSlot ## added later
    if ((NormalScheduling==New3NormalProposedAlgorithm) |(NormalScheduling==New2NormalProposedAlgorithm) | (NormalScheduling==New1NormalProposedAlgorithm) | (NormalScheduling==NormalProposedAlgorithm) | (NormalScheduling==NormalProposedAlgorithm2)): ## because currently our proposed Alg mechansim for choosing a dst for normal tasks is different with other strategy:
        ## Our proposed alg has two phases:
        # first for check tasks in thier src
        # then when all node check thier own task, check other node's tasks
        for manager in managersList:
            NormalScheduling(manager.buffer,manager,slot)
            #(manager.buffer).clear() #clear the buffer of manager after scheduling
    else: ## for other normal scheduling algorithms
        for manager in managersList:
            for task in manager.buffer:
                if ((task.generationTime<=now) & (task.scheduled==False)):
                    dst=NormalScheduling(task,manager,colonyList,slot) 
                    task.dst=dst
                    print("task src is ",task.src.name)
                    print("dst is ",task.dst.name)
                    forward(task,now,BW_TransDelay1,Cost_ResponseTime1) #XX xx slot changed to now because in proposed alg we send the now time not slot
        #(manager.buffer).clear() #clear the buffer of manager after scheduling


def forward(task,now,BW_TransDelay1,Cost_ResponseTime1):#append task based on its type to sutable queue of dst
    #and update scheduling time,arrival time and com delay of the task
    src=task.src
    dst=task.dst
    srcColony=src.parent
    dstColony=srcColony # will change later
    #print("src ",src.name)
    #print("colony of src is ",srcColony.id)
    #print("src colony node manager ",(srcColony.nodeManager).name)
    #print("dst ",dst.name)
    communicationDelay=0
    #cloud=listOfManagers[0] #!!!!!!
    #here is special case :when cloud is chosen as a dst
    if dst.name!=cloud.name: ## error handling, if dst is cloud so it has no parent
        #print("HERE1")
        dstColony=(dst).parent
        #print("colony ",dstColony.id)
    ## checking dst from here 
    if task.dst==-1:
        #print("HERE2")
        print("for task ",end=" ")
        print(task.id,end=" ")
        print("Using forward before scheduling")
        return
    elif dst.name==src.name:#when task is scheduled on the generator node
        #print("HERE3")
        communicationDelay=0
    elif dst.name==cloud.name:
        #print("HERE4")
        com1=0 #Delay from src to src manager
        com1=communicationDelayComputer(task,src,srcColony.nodeManager,task.insize) #send the task from src to manager
        com2=0 #Delay from src.manager to cloud
        #BW,TransDelay=srcColony.matrixBW_Propagation[((srcColony.nodeManager).name,cloud.name)]
        BW,TransDelay=managersBW_TransDelay[((srcColony.nodeManager).name,cloud.name)]
        com2=TransDelay+((task.insize)/BW)
        communicationDelay=com1+com2
    elif srcColony.id==dstColony.id: #when dst and src are in a same Colony
        #print("HERE5")
        communicationDelay=communicationDelayComputer(task,src,dst,task.insize)
    else:## dst and src are not in a Colony
        #print("HERE6")
        com1=0 #Delay from src to src manager
        com1=communicationDelayComputer(task,src,srcColony.nodeManager,task.insize) #send the task from src to manager
        com2=0 #Delay from src manger to dst manager
        #print("dst colony node manager ",(dstColony.nodeManager).name)
        print((dstColony.nodeManager))
        #print("com 1:from src to manager ",com1)
        print()
        BW,TransDelay=managersBW_TransDelay[((srcColony.nodeManager).name,(dstColony.nodeManager).name)]
        com2=TransDelay+((task.insize)/BW)
        com3=0 #Delay from dst manger to dst
        com3=communicationDelayComputer(task,dstColony.nodeManager,dst,task.insize) #send the task from dst.manager to dst
        communicationDelay=com1+com2+com3

    # based on task type add to the suitable queuedd
    tasktype=task.type
    if tasktype=="normal":
        dst.queueOfNormals.append(task) ## add task to queueOfNormals of the DST
        task.schedulingTime=now ## becouse it schedule in thr beggening of slot
    elif tasktype=="critical":
        dst.queueOfCriticals.append(task) ## add task to queueOfCriticals of the DST
        task.schedulingTime=task.generationTime ## becouse scheduling of critical task is realtime at the moment of creation
    task.arrivalTime=(task.schedulingTime)+communicationDelay #update arrivalTime of the task at dst node
    task.communicationDelay=communicationDelay #update communicationDelay of the task from src to dst
    task.schedulingDelay=task.schedulingTime-task.generationTime
    task.scheduled=True ##update scheduled of task to TRUE

def communicationDelayComputer(task,src,dst,fileSize): ##compute a com delay from src to dst by the given route
    srcColony=src.parent
    path=srcColony.routingTable[((src).name,(dst).name)] ##path from src to dst
    communicationDelay=0
    current=src.name
    for node in path: ##compute the com delay from src to dst
        nextNode=node
        BW,PropagationDelay=srcColony.matrixBW_Propagation[(current,nextNode)]
        TransDealay=(fileSize)/BW
        communicationDelay=communicationDelay+PropagationDelay+TransDealay
        current=node
    return communicationDelay


def manager_table_copy_proceed(ListOfC,BW_TransDelay,Cost_ResponseTime):
    for k in managersBW_TransDelay.keys(): 
        n1=k[0]
        n2=k[1]
        node1=-1
        node2=-1
        for c in ListOfC:
            for node in c.listOfNodes:
                if n1.name==node.name:
                    node1=node
                if n2.name==node.name:
                    node2=node
        if node1==-1:
            node1=cloud
        if node2==-1:
            node2=cloud
        FN1=cloud
        FN2=cloud
        for c in ListOfC:
            for n in c.listOfNodes:
                if node1!=cloud:
                    if(n.id==node1.id):
                        FN1=n
                if node2!=cloud:
                    if(n.id==node2.id):
                        FN1=n
        BW=managersBW_TransDelay[(n1,n2)][0] 
        Trans=managersBW_TransDelay[(n1,n2)][1]
        Cost=managersCost_ResponseTime[(n1,n2)][0]
        Resp=managersCost_ResponseTime[(n1,n2)][1]
        BW_TransDelay[(FN1.name,FN2.name)]=BW,Trans
        BW_TransDelay[(FN2.name,FN1.name)]=BW,Trans
        Cost_ResponseTime[(FN1.name,FN2.name)]=Cost,Resp
        Cost_ResponseTime[(FN2.name,FN1.name)]=Cost,Resp  

def box_plot(data,names,x1,xlabel,ylabel):
    #data=[rspOfNormals,rspOfCriticals]
    fig = plt.figure(figsize =(10, 7))
     
    # Creating plot
    plt.boxplot(data)
    print("data ",data)
    #plt.xticks([1, 2], ['Normals', 'Criticals'])
    print("x1  ",x1)
    print("names ",names)
    plt.xticks(x1, names)


    #axis label 
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    # show plot
    plt.show()


def show_results(name,LOC):
    print("******Name of Strategy is ",name)
    totalNumberOfTasks=0
    avgResp=0
    totalResp=0
    listOfNotFinished=[]
    for colony in LOC:
        print("colony is ",colony.id)
        for node in colony.listOfNodes:
            print("node is ",node.id)
            if(node==colony.nodeManager):
                print(" is a manager")
                continue
            totalNumberOfTasks+=(len(node.listOfTasks))
            for task in node.listOfTasks:
                print("task is ",end="  ")
                print(task.id,end="  ")
                if (task.finished!=True) | (task.scheduled!=True):
                    listOfNotFinished.append(task)
                    print("Unfinnished")
                    continue
                totalResp+=(task.responseTime)
                print("src is ",end="  ")
                print((task.src).name,end="  ")
                print("dst is ",end="  ")
                print((task.dst).name,end="  ")
                print(task.type,end="  ")
                print("response time  is ",end="  ")
                print(int(task.responseTime))
    print("******Total number Of generated tasks = ",totalNumberOfTasks)      
    print("Total number Of Not finished tasks = ",len(listOfNotFinished))
    print("percentage of NOT FINISHED tasks: ",int(len(listOfNotFinished)/totalNumberOfTasks*100))    
    print("Avg responseTime ",end=": ")
    print((totalResp/(totalNumberOfTasks-len(listOfNotFinished))))
    listOfNotFinished.clear()


FogCloud()

#box_plot()
#show_status()
#report()
