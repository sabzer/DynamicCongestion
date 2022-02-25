##############################IMPORTS##############################
import numpy as np
import random
import copy

##############################INITIALIZATIONS##############################
random.seed()
learningRate=0.1
discountFactor=0.9
agents=5
adjMtx = np.genfromtxt('adjMtx.txt', delimiter=' ')
agentLocMtx=np.zeros(np.shape(adjMtx))
stateTable=np.zeros((agents)) #all initialized at source
actionTable=np.zeros((agents))
iterCount=1000

doraTheExplorer=0.1

totalStates=(np.shape(adjMtx))[0]
totalActions=int(sum(sum(adjMtx)))

edgeList={
    }

enumEdge={
    }
stateBase=max(totalStates,agents)
actionBase=max(totalActions,agents)+1

congestion=np.zeros(totalStates)

qMatrix=np.zeros((stateBase**agents,actionBase**agents))




##############################FUNCTIONS FOR MATRIX MATH##############################

def defineEdgeList():
    count=0
    for i in range(totalStates):
        for j in range(totalStates):
            if(adjMtx[i,j]==1):
                edgeList[count]=(i,j)
                enumEdge[(i,j)]=count
                count+=1

def possibleNextStates(v): #returns the 'degree' of the current vertex (possible actions)
    return int(np.sum(adjMtx[v,:]))

def vtxPos(x,v): #figure out where xth vertex is (with v is row)
    count=0
    for i in range((np.shape(adjMtx))[0]):
        if(adjMtx[int(v),i]==1):
            count+=1
        if(count==x):
            return i
    return 0

def convertIntoBaseAction(a):
    retVal=0
    for i in range(agents):
        retVal+=(actionBase**i)*a[i]
    return int(retVal)

def convertIntoBaseState(a):
    retVal=0
    for i in range(agents):
        retVal+=(stateBase**i)*a[i]
    return int(retVal)

def convertOutofBaseAction(x):
    retVal=np.zeros(agents)
    for i in range(agents):
        retVal[agents-i-1]= int(x/actionBase**(agents-i-1))
        x=x%actionBase**(agents-i-1)
    return retVal

def convertOutofBaseState(x):
    retVal=np.zeros(agents)
    for i in range(agents):
        retVal[agents-i-1]= int(x/stateBase**(agents-i-1))
        x=x%stateBase**(agents-i-1)
    return retVal    

def rewardCalc(): 
    return np.square(congestion)/agents
    #cost of edge
    
def allPossibleNextActions(statTable):#wish to return list of list of edges that can be taken next for each agent
    ret=[]
    for i in range(agents):
        curList=[]
        curVtx=statTable[i]
       # print("the current vtx is ")
       # print(curVtx)
        for j in range(totalStates):
            if(adjMtx[int(curVtx)][j]==1):
                edge=(int(curVtx),j)
                curList.append(enumEdge[edge])
        if(curList==[]):
            curList.append(actionBase-1)
        ret.append(curList)
        
    return ret

#currently are awarding the congestion while needing to deduct it??
def congestionCalc():
    for i in range(totalActions):
        tu=edgeList[i]
        row=tu[0]
        col=tu[1]
        for j in range(agents):
            comp=edgeList[actionTable[j]]
            compRow=comp[0]
            compCol=comp[1]
            if(row==compRow and col==compCol):
                congestion[i]=congestion[i]+1
    return congestion
                


#choices_list - a list (n long) of choices. Each choice is a list of options that need to be chosen
def combinator(choices_list=[]):
    if len(choices_list) == 0:
        return [[]] #no options need selection
    #the list of options for the choice we need to make this recursive execution
    options = choices_list.pop(-1) #grab the last choice in the list

    combs = combinator(choices_list) #generate a list of combinations of choices for everything but the last choice
    new_combs = []
    
    for option_selection in options:
        new_comb = copy.deepcopy(combs)
        for comb_list in new_comb:
            comb_list.append(option_selection)
        new_combs.extend(new_comb)
    return new_combs


for i in range(agents):
    z=possibleNextStates(int(stateTable[i]))
    x=random.randrange(z)
    stateTable[i]=vtxPos(x,stateTable[i])

'''
stateTable[0]=0
stateTable[1]=1
stateTable[2]=2
stateTable[3]=3
stateTable[4]=4

a=convertIntoBaseState()
print(a)
b=convertOutofBaseState(a)
print(b)

defineEdgeList()
print(edgeList)
print(enumEdge)

a=allPossibleNextActions()
print(a)
print("now combining aciton")
a=a[0:5]
print(combinator(a))
'''


##############################FUNCTIONS FOR RL##############################


def qUpdate():
    for i in range(iterCount):
        done=False
        stateTable=np.zeros((agents))
        
        while(not done):
            a=allPossibleNextActions(stateTable)
            actionCombos=combinator(a)
            x=random.random()
            
            if(x<doraTheExplorer):#we are exploring!
                z=random.randrange(len(actionCombos))
                actionTable=actionCombos[z]
                
            else:
                #This is kinda a stupid way to go around this but since it inits to 0, need to make sure
                #that there is equal opportunity among 0 Q value S,A pairs
                sameCheck=np.zeros((len(actionCombos)))
                maxVal=-np.inf
                bestAction=np.zeros((agents))
                for j in range(len(actionCombos)):
                    curCombo=actionCombos[j]
                    convStateToBase=convertIntoBaseState(stateTable)
                    specActionToBase=convertIntoBaseAction(curCombo)
                    val=qMatrix[convStateToBase,specActionToBase]
                    if(val==0):
                        sameCheck[j]=1
                    if(val>maxVal):
                        maxVal=val
                        bestAction=curCombo
                if(maxVal==0):
                    valToPick=random.randrange(sum(sameCheck))
                    count=0
                    for k in range(len(sameCheck)):
                        if(count==valToPick and sameCheck[k]==1):
                            bestAction=actionCombos[k]
                        if(sameCheck[k]==1):
                            count+=1
                actionTable=bestAction
            #NEED TO DO CALCULATING OF NEXT STATE
            nextStateTable=stateTable
            
            for a in range(agents):
                nextStateTable[a]=stateTable[a]
                edgeVal=actionTable[a]
                if(edgeVal==actionBase-1):
                    nextStateTable[a]=stateTable[a]
                else:
                    edge=edgeList[actionTable[a]]
                    nextStateTable[a]=edge[1]
            
            
            ########REVIEW, IS THIS EVEN FEASIBLE????? REPLACE ARGS OF FUNCTIONS
            a=allPossibleNextActions(nextStateTable)
            actionCombos=combinator(a)
            sameCheck=np.zeros((len(actionCombos)))
            maxVal=-np.inf
            newBestAction=np.zeros((agents))
            for j in range(len(actionCombos)):
                curCombo=actionCombos[j]
                convStateToBase=convertIntoBaseState(nextStateTable)
                specActionToBase=convertIntoBaseAction(curCombo)
                val=qMatrix[convStateToBase,specActionToBase]
                if(val==0):
                    sameCheck[j]=1
                if(val>maxVal):
                    maxVal=val
                    newBestAction=curCombo
            if(maxVal==0):
                valToPick=random.randrange(sum(sameCheck))
                count=0
                for k in range(len(sameCheck)):
                    if(count==valToPick):
                        newBestAction=actionCombos[k]
                    if(sameCheck[k]==1):
                        count+=1
            mostBestAction=newBestAction
            
            #Q value update
            congestionCalc()
            reward=np.sum(rewardCalc())
            
            stateTableBase=convertIntoBaseState(stateTable)
            actionTableBase=convertIntoBaseAction(actionTable)
            
            bestStateTableBase=convertIntoBaseState(nextStateTable)
            bestActionTableBase=convertIntoBaseAction(mostBestAction)
            qMatrix[stateTableBase,actionTableBase]=qMatrix[stateTableBase,actionTableBase]+learningRate*(reward+discountFactor*qMatrix[bestStateTableBase,bestActionTableBase]-qMatrix[stateTableBase,actionTableBase])
            #print("finished an update")
            if(np.sum(actionTable)==(actionBase-1)*agents):
                done=True
    return qMatrix


def findBestPath(qMatrix):
    res=[]
    stateTable=np.zeros((agents))    
    a=allPossibleNextActions(stateTable)
    actionCombos=combinator(a)
    z=random.randrange(len(actionCombos))
    actionTable=actionCombos[z]
    res.append(stateTable)
    #for initialization
    #print(actionTable)
    while(np.sum(actionTable)!=(actionBase-1)*agents):
        nextStateTable=np.zeros((agents))
        for a in range(agents):
            nextStateTable[a]=stateTable[a]
            edgeVal=actionTable[a]
            if(edgeVal==actionBase-1):
                nextStateTable[a]=stateTable[a]
            else:
                edge=edgeList[actionTable[a]]
                nextStateTable[a]=edge[1]
       # print(nextStateTable)
        maxVal=0
        maxAction=0
        a=allPossibleNextActions(nextStateTable)
       # print(a)
        a=combinator(a)
       # print(a)
        nextStateInBase=convertIntoBaseState(nextStateTable)
        for i in a:
            #print("current action is")
            #print(i)
            curAction=convertIntoBaseAction(i)   
            curVal=qMatrix[nextStateInBase,curAction]
            if(curVal>=maxVal):
                maxVal=curVal
                maxAction=curAction
        print(maxAction)
        print(maxVal)
        stateTable=nextStateTable
        actionTable=convertOutofBaseAction(maxAction)
        res.append(stateTable)        
        #print(res)
    return res
            
            
        

##############################RUNNING THE SIM!##############################


example=[0, 0, 1, 1, 1]
#print(example)
base=convertIntoBaseAction(example)
#print(base)
exampleOutBase=convertOutofBaseAction(base)
#print(exampleOutBase)
defineEdgeList()
qUpdate()
res=findBestPath(qMatrix)
print(res)





