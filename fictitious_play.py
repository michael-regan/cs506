import csv, random, operator
import numpy as np

# Fictitious play

# Modeling two players rock-paper-scissors match through fictitious play. Uses MWU update for actual play.

"""Parameters"""

learningRate_p1 = 0.001
learningRate_p2 = 0.001

num_rounds = 1200000

########################

objects=['rock', 'paper', 'scissors']

weights_p1 = [1/3] * len(objects)

weights_p2 = [1/3] * len(objects)

########################

def draw(myWeights):
    choice = random.uniform(0, sum(myWeights))
    choiceIndex = 0

    for weight in myWeights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1
        

def reward(move, player):
    
    """ outcome is a tuple (player1move, player2move)
        player is in {-1, 1}
        where player1=1, player2=-1
    """
    
    if move[0]==move[1]:
        return 0
        
    elif move[0]=='rock' and move[1]=='paper':
        return -0.1*player
        
    elif move[0]=='rock' and move[1]=='scissors':
        return 1*player
        
    elif move[0]=='paper' and move[1]=='rock':
        return 0.1*player
        
    elif move[0]=='paper' and move[1]=='scissors':
        return -1*player
        
    elif move[0]=='scissors' and move[1]=='rock':
        return -1*player
        
    elif move[0]=='scissors' and move[1]=='paper':
        return 1*player
    
    
def fictitious_play(player, myWeights):
    
    rock=[0, -0.1, 1]
    paper=[0.1, 0, -1]
    scissors=[-1, 1, 0]
    
    l=len(myWeights)
    
    exp_rock=np.dot(rock, myWeights)/l
    exp_paper=np.dot(paper, myWeights)/l
    exp_scissors=np.dot(scissors, myWeights)/l

    indices_with_exps=enumerate([exp_rock, exp_paper, exp_scissors])
    
    index, value = max(indices_with_exps, key=operator.itemgetter(1))
    
    return index
    
    
def payoff(index):
    
    payoff_when_opp_plays_rock=[0, 0.1, -1]
    payoff_when_opp_plays_paper=[-0.1, 0, 1]
    payoff_when_opp_plays_scissors=[1, -1, 0]
    
    if index==0:
        indx, value = max(enumerate(payoff_when_opp_plays_rock), key=operator.itemgetter(1))
    elif index==1:
        indx, value = max(enumerate(payoff_when_opp_plays_paper), key=operator.itemgetter(1))
    else:
        indx, value = max(enumerate(payoff_when_opp_plays_scissors), key=operator.itemgetter(1))
        
    return indx, value
        

def update_weight_paranoid(myWeights, player, index, best_payoff):
    
    if player==1:
        myWeights[index] *= (1 + learningRate_p1 * weights_p1[index]*best_payoff)
    else:
        myWeights[index] *= (1 + learningRate_p2 * weights_p2[index]*best_payoff)
    
    return normalize(myWeights)
    


def update_weight(myWeights, move, player):

    """for every expert i who predicts wrongly, decrease his weight for the next round by multiplying it by a factor of (1+learning rate*reward)"""
    
    if player==1:
        indx = objects.index(move[0])
    else:
        indx = objects.index(move[1])
        
    if player==1:
        myWeights[indx] *= (1 + learningRate_p1 * reward(move, player))
    else:
        myWeights[indx] *= (1 + learningRate_p2 * reward(move, player))
        
    return normalize(myWeights)
    

def normalize(myWeights):
    
    total=0
    for i in range(len(myWeights)):
        total+=myWeights[i]
        
    return [x/total for x in myWeights]
    
    

def make_move():
    
    # column you expect player 2 to play
    exp_col = fictitious_play(1, weights_p1)
    index_p1, best_payoff_p1 = payoff(exp_col)
    

    # row you expect player 1 to play
    exp_row = fictitious_play(-1, weights_p2)
    index_p2, best_payoff_p2 = payoff(exp_row)
    
    
    w1=update_weight_paranoid(weights_p1, 1, index_p1, best_payoff_p1)
    w2=update_weight_paranoid(weights_p2, -1, index_p2, best_payoff_p2)
    
    move_p1=objects[draw(w1)]
    move_p2=objects[draw(w2)]
    
    move=(move_p1, move_p2)

    w1=update_weight(w1, move, 1)
    w2=update_weight(w2, move, -1)
    
    return w1, w2
    


def to_csv(myValues):
    
    path='/Users/Michael/Desktop/mwu_plot.csv'
    
    with open (path, 'w') as g:
        
        writer=csv.writer(g)
        writer.writerow(('date', 'rock_p1', 'paper_p1', 'scissor_p1', 'rock_p2', 'paper_p2', 'scissor_p2'))
        for i in myValues:
            writer.writerow(i)
        
        

if __name__ == "__main__":

    plot_weights1_rock=[]
    plot_weights1_paper=[]
    plot_weights1_scissors=[]
    plot_weights2_rock=[]
    plot_weights2_paper=[]
    plot_weights2_scissors=[]
    plot_times=[]

    for i in range(num_rounds):
        weights_p1, weights_p2 = make_move()

        if i%10000==0:
            w1=[round(x, 4) for x in weights_p1]
            w2=[round(x, 4) for x in weights_p2]
            #print(i, w1, w2)
            if w1[0]!='':
                plot_weights1_rock.append(w1[0])
                plot_weights1_paper.append(w1[1])
                plot_weights1_scissors.append(w1[2])
                plot_weights2_rock.append(w2[0])
                plot_weights2_paper.append(w2[1])
                plot_weights2_scissors.append(w2[2])
                plot_times.append(i)
                
    to_csv(zip(plot_times, plot_weights1_rock, plot_weights1_paper, plot_weights1_scissors, plot_weights2_rock, plot_weights2_paper, plot_weights2_scissors))
