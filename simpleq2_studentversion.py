import matplotlib.pyplot as plt

import numpy as np

import matplotlib
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

class simpleprob1():
  # all actions into one single state, the keystate, give a high reward

  def __init__(self,numh,numw, keystate):
  
    self.numh=numh
    self.numw=numw

    if (keystate[0]<0) or (keystate[0]>=self.numh):
      print('illegal')
      exit()
    if (keystate[1]<0) or (keystate[1]>=self.numw):
      print('illegal')
      exit()

    #state space: set of tuples (h,w) 0<=h<=numh, 0<=w<=numw
    self.statespace=[ (h,w) for h in range(self.numh) for w in range(self.numw) ]
 
    self.statespace2index=dict()
    for i,s in enumerate(self.statespace):
      self.statespace2index[s]=i



    self.actions=['stay','left','down','right','up']
    self.actdict2index=dict()
    for i,a in enumerate(self.actions):
      self.actdict2index[a]=i


    self.highrewardstate=keystate
    self.rewardtogothere=10.

    #print(self.statespace2index)
    #only for RL
    #self.state=[np.random.randint(0,self.numh),np.random.randint(0,self.numw)]
    self.reset()

  def transition_deterministic(self,oldstate_index,action):
    #P(s'|s,a) is 1 for one specific s'

    if action not in self.actions:
      print('illegal')
      exit()


    oldstate=self.statespace[oldstate_index]
    
    # all deterministic

    if self.actdict2index[action]==0:
      newstate=list(oldstate)

    elif self.actdict2index[action]==1:
      newstate=list(oldstate)
      newstate[1]=min(self.numw-1,newstate[1]+1)


    elif self.actdict2index[action]==2:
      newstate=list(oldstate)
      newstate[0]=min(self.numh-1,newstate[0]+1)


    elif self.actdict2index[action]==3:
      newstate=list(oldstate)
      newstate[1]=max(0,newstate[1]-1)


    elif self.actdict2index[action]==4:
      newstate=list(oldstate)
      newstate[0]=max(0,newstate[0]-1)

    #can return probs or set of new states and probabilities

    done=False # can play forever
    return self.statespace2index[tuple(newstate)]
  

  def reward(self,oldstate_index,action,newstate_index):
    #P(R|s,a)
    onlygoalcounts=True

    if False==onlygoalcounts: #one gets  a reward when one jumps into the golden state or stays there
      r=self.tmpreward1(oldstate_index, action, newstate_index)
    else: #one gets only a reward when one stays in the golden state
      r=self.tmpreward2(oldstate_index, action, newstate_index) 

    return r
  
  def tmpreward1(self,oldstate_index,action,newstate_index):

    newstate=self.statespace[newstate_index]
    if (newstate[0]==self.highrewardstate[0]) and (newstate[1]==self.highrewardstate[1]):
      return self.rewardtogothere
    else:
      return 0

  def tmpreward2(self,oldstate_index,action,newstate_index):

    newstate=self.statespace[newstate_index]
    if (newstate[0]==self.highrewardstate[0]) and (newstate[1]==self.highrewardstate[1]) and (action=='stay'):
      return self.rewardtogothere
    else:
      return 0



  ##################################
  # for RL
  #####################################
  def reset(self):
    #randomly set start point
    self.state=np.random.randint(0,len(self.statespace))
    return self.state

  def getstate(self):
    return self.state

  def step(self,action):
    #print(self.state,action)
    done=False
    tmpstateind=self.transition_deterministic(self.state,action)
    reward=self.reward(self.state, action, tmpstateind)

    self.state=tmpstateind
    
    if self.state == self.statespace2index[tuple(self.highrewardstate)]:
      done = True

    return self.state, reward, done



def plotqvalstable(qvals, simpleprob_instance, block):
  # input is numpy of shape (5,h,w)  
  #plotted into 3x3 + boundary  qvals[c,h,w] c=center,l,d,r,up
  plt.ion()

  offsets=[ [1,1],[1,2],[2,1],[1,0],[0,1] ]  
  symbols=[ 'o','->','\ ','<-','^' ]  

  mh=simpleprob_instance.numh
  mw=simpleprob_instance.numw

  plotvals=-np.ones((3*mh,3*mw))

  for i in range(len(simpleprob_instance.statespace)):
    h=simpleprob_instance.statespace[i][0]
    w=simpleprob_instance.statespace[i][1]

    for c in range( len(simpleprob_instance.actions)):
        plotvals[3*h + offsets[c][0] ,3*w+ offsets[c][1]]=qvals[ i,c]

  plotvals = np.ma.masked_where(plotvals<0,plotvals)

  fig, (ax0) = plt.subplots(1, 1, figsize=(10, 10))

  ax0.imshow(plotvals, cmap=plt.get_cmap('summer'),interpolation='nearest')



  #c = ax0.pcolor(plotvals, edgecolors='white', linewidths=1)
  ax0.patch.set(hatch='xx', edgecolor='red')

  for i in range(len(simpleprob_instance.statespace)):
    h=simpleprob_instance.statespace[i][0]
    w=simpleprob_instance.statespace[i][1]

    for c in range( len(simpleprob_instance.actions)):
      if c==0:
        printstr= "{:.2f}".format(qvals[ i,c]) #str(qvals[c,h,w])
      elif c==1:
        printstr="{:.2f}".format(qvals[ i,c])+symbols[c]
      elif c==2:
        printstr=symbols[c]+"{:.2f}".format(qvals[ i,c])
      elif c==3:
        printstr=symbols[c]+ "{:.2f}".format(qvals[ i,c])
      elif c==4:
        printstr=symbols[c]+"{:.2f}".format(qvals[ i,c])
      
              
      ax0.text( 3*w+ offsets[c][1], 3*h + offsets[c][0],printstr,
                     ha="center", va="center", color="k")

  plt.draw()
  plt.pause(0.01)
  plt.savefig('./q.png')
  if True==block:
    input("Press [enter] to continue.")




def plotonlyvalstable2(qvals, simpleprob_instance,  block):
  # input is numpy of shape (5,h,w)  
  #plotted into 3x3 + boundary  qvals[c,h,w] c=center,l,d,r,up
  plt.ion()

  mh=simpleprob_instance.numh
  mw=simpleprob_instance.numw


  plotvals=-np.ones((mh,mw))
  for i in range(len(simpleprob_instance.statespace)):
    h=simpleprob_instance.statespace[i][0]
    w=simpleprob_instance.statespace[i][1]
    for c in range( len(simpleprob_instance.actions)):
      plotvals[h,w]=np.max(qvals[ i,:])

  #print(qvals)
  plotvals = np.ma.masked_where(plotvals<0,plotvals)

  fig, (ax0) = plt.subplots(1, 1)

  ax0.imshow(plotvals, cmap=plt.get_cmap('summer'),interpolation='nearest')

  #c = ax0.pcolor(plotvals, edgecolors='white', linewidths=1)
  #ax0.patch.set(hatch='xx', edgecolor='black', color='red')

  for h in range(mh):
    for w in range(mw):
      printstr= "{:.2f}".format(plotvals[h,w])   
      ax0.text( w, h ,printstr,ha="center", va="center", color="k")

  plt.draw()
  plt.pause(0.01)
  if True==block:
    #pass
    input("Press [enter] to continue.")



def plotonlyvalstable2b(qvals, simpleprob_instance,  block):
  # input is numpy of shape (5,h,w)  
  #plotted into 3x3 + boundary  qvals[c,h,w] c=center,l,d,r,up
  plt.ion()

  mh=simpleprob_instance.numh
  mw=simpleprob_instance.numw


  plotvals=-np.ones((mh,mw))
  for i in range(len(simpleprob_instance.statespace)):
    h=simpleprob_instance.statespace[i][0]
    w=simpleprob_instance.statespace[i][1]
    for c in range( len(simpleprob_instance.actions)):
      plotvals[h,w]=np.max(qvals[ i,:])

  #print(qvals)
  plotvals = np.ma.masked_where(plotvals<0,plotvals)

  #plt.savefig('./.png')
  
  fig= plt.figure(1)
  plt.clf()
  #fig, (ax0) = plt.subplots(1, 1)
  ax0=plt.axes()
  ax0.imshow(plotvals, cmap=plt.get_cmap('summer'),interpolation='nearest')

  #c = ax0.pcolor(plotvals, edgecolors='white', linewidths=1)
  #ax0.patch.set(hatch='xx', edgecolor='black', color='red')

  for h in range(mh):
    for w in range(mw):
      printstr= "{:.2f}".format(plotvals[h,w])   
      ax0.text( w, h ,printstr,ha="center", va="center", color="k")


  plt.draw()
  plt.pause(0.001)
  if True==block:
    #pass
    input("Press [enter] to continue.")


def plotmoves(statesseq, simpleprob_instance,  block):
  fig= plt.figure(5)
  ax0=plt.axes()

  plt.clf()
  mh=simpleprob_instance.numh
  mw=simpleprob_instance.numw

  for s in statesseq:

    #plt.clf()

    h=simpleprob_instance.statespace[s][0]
    w=simpleprob_instance.statespace[s][1]
    plotvals=np.zeros((mh,mw))
    plotvals[h,w]=1
    print(h,w)
    ax0.imshow(plotvals, cmap=plt.get_cmap('summer'),interpolation='nearest')

    plt.draw()

    plt.pause(0.05)  # pause a bit so that plots are updated
    
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
  plt.savefig('./moves.png')
def valueiter_mdp_q2(problemclass,gamma,  delta , showeveryiteration):

  pass



def plot_rewards2(episode_rewards,means100):
    plt.figure(3)
    plt.clf()

    plt.title('training or testing...')
    plt.xlabel('Episode')
    plt.ylabel('averaged reward')
    plt.plot( np.asarray( episode_rewards ))
    # Take 100 episode averages and plot them too
    if len(episode_rewards) >= 100:
        #means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        #means = torch.cat((torch.zeros(99), means))
        #plt.plot(means.numpy())
        mn=np.mean(episode_rewards[-100:])
    else:
        mn=np.mean(episode_rewards)
    means100.append(mn)
    plt.plot(means100)
        #print('100 mean:')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

class agent_Qlearn_thattable( ):
  def __init__(self, simpleprob_instance):

    self.gamma=0.9
    self.softupdate_alpha=0.2

    self.delta=1e-3 # Q-convergence

    self.numactions=len(simpleprob_instance.actions)
    self.statespace= [i for i in range(len(simpleprob_instance.statespace))]
    self.Q=np.zeros(( len(self.statespace) , self.numactions  ))

    #how should the eps for exploration vs exploitation decay?
    self.epsforgreediness_start=0.9
    self.epsforgreediness_end=0.01
    self.epsforgreediness_maxdecaytime=100




  def currenteps(self,episode_index):
    
    if episode_index<0:
      v=self.epsforgreediness_end
    else:
      v=self.epsforgreediness_end + (self.epsforgreediness_start-self.epsforgreediness_end)* max(0,self.epsforgreediness_maxdecaytime-episode_index)/ float(self.epsforgreediness_maxdecaytime)

    return v

  def actionfromQ(self,state_index,episode_index):
    #episode_index for decay
    eps=self.currenteps(episode_index)
    random_action = np.random.choice(2, p = [1-eps, eps])
    if random_action == 1:
        action = np.random.randint(5)
    else:
        policy = self.Q[state_index]
        action = np.argmax(policy)
    #
    # YOU
    #
    return action

  def train(self, simpleprob_instance):

    numepisodes=120 #250
    maxstepsperepisode=100
    

    episode_rewards=[]
    means100=[]

    keystate = tuple(simpleprob_instance.highrewardstate)
    
    for ep in range(numepisodes):
      avgreward = 0
      count = 0
      for step in range(maxstepsperepisode):
        old_state = simpleprob_instance.getstate()
        action_num = self.actionfromQ(old_state, ep)
        action = simpleprob_instance.actions[action_num]
        new_state, reward, done = simpleprob_instance.step(action)
        prev_q = self.Q[old_state,action_num]
        self.Q[old_state,action_num] += self.softupdate_alpha*(reward + self.gamma*np.max(self.Q[new_state] - prev_q))
        avgreward += reward
        count += 1
        #print(simpleprob_instance.statespace2index[keystate], new_state, done)
        if done:
          print('Obj reached, early termination')
          break
      
      #
      # YOU
      #

      avgreward /= count
      # outside of playing one episode
      episode_rewards.append(avgreward)
      plot_rewards2(episode_rewards,means100)

      print('episode',ep,'averaged reward',avgreward)

      if ep%10==0:
        plotonlyvalstable2b(self.Q, simpleprob_instance,  block=False)
        
      simpleprob_instance.reset()
    plotqvalstable(self.Q, simpleprob_instance,  block=False)

  def runagent(self, simpleprob_instance):
    maxstepsperepisode=20

    state_index=simpleprob_instance.reset()

    episode_rewards=[]
    means100=[]
    statesseq=[state_index]

    avgreward=0
    count = 0
    for playstep in range(maxstepsperepisode):
    
      old_state = simpleprob_instance.getstate()
      policy = self.Q[old_state]
      action_num = np.argmax(policy)
      action = simpleprob_instance.actions[action_num]
      new_state, reward, done = simpleprob_instance.step(action)
      avgreward += reward
      count += 1
      statesseq.append(new_state)
      #
      # YOU
      #
    avgreward /= count
    # outside of playing one episode
    episode_rewards.append(avgreward)
    #plot_rewards2(episode_rewards,means100)
    print('post training run averaged reward',avgreward)
    plotmoves(statesseq, simpleprob_instance,  block=False)

def runmdp():

  plotbig=True
  showeveryiteration=True

  mdp=simpleprob1(5,6,keystate=[1,4])
  values,qsa=valueiter_mdp_q2(mdp,gamma=0.5, delta=3e-2, showeveryiteration=showeveryiteration)

  
  if False==plotbig:
    if False==showeveryiteration:
      plotonlyvalstable(qsa,  mdp, block=False)
  else:
    plotqvalstable(qsa, mdp,block=False)

  for i in range(3):
    print('FINISHED')
  input("Press [enter] to continue.")


def trainsth():

  plotbig=True
  showeveryiteration=True

  problem=simpleprob1(5,6,keystate=[1,4])

  ag=agent_Qlearn_thattable(problem)
  ag.train(problem)

  '''
  if False==plotbig:
    if False==showeveryiteration:
      plotonlyvalstable(ag.Q,  problem, block=False)
  else:
    plotqvalstable(ag.Q, problem,block=False)
  '''

  for i in range(3):
    print('FINISHED')

  ag.runagent( problem)

  for i in range(3):
    print('FINISHED')
  input("Press [enter] to continue.")


if __name__=='__main__':
  #tester()
  #runmdp()
  trainsth()



