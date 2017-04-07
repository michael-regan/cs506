import random

learningRate = 0.01

num_rounds = 10000

objects=['rock', 'paper', 'scissors']

weights_p1 = [1/3] * len(objects)

weights_p2 = [1/3] * len(objects)

# Fictitious play/MWU

# Modeling two players rock-paper-scissors match, with each player making move based on uniform random distribution of weights 
# (AKA experts) that are updated using an arbitrarily chosen set of costs that are incurred for making a losing choice 
# (in this implementation, the cost of rock losing against paper is decreased)

# Task: Show that a Nash equilibrium exists

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
    
    
        
def update_weight(myWeights, move, player):
    
    """for every expert i who predicts wrongly, decrease his weight for the next round by multiplying it by a factor of (1+learning rate*reward)"""
    
        if player==1:
            indx = objects.index(move[0])
        else:
            indx = objects.index(move[1])
    
        myWeights[indx] *= (1 + learningRate * reward(move, player))
        
        return normalize(myWeights)



def normalize(myWeights):
    
    total=0
    for i in range(len(myWeights)):
        total+=myWeights[i]
        
    return [x/total for x in myWeights]
    
    

def make_move():
    
    p1move=objects[draw(weights_p1)]

    p2move=objects[draw(weights_p2)]

    move = (p1move, p2move)
    
    w1=update_weight(weights_p1, move, 1)
    
    w2=update_weight(weights_p2, move, -1)
    
    return w1, w2
    

   

if __name__ == "__main__":
    

    for i in range(num_rounds):
        weights_p1, weights_p2 = make_move()
        if i%100==0:
            print(i, weights_p1, weights_p2)
