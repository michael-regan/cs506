import numpy as np
import math

# a natural breaking point: ValueError emerges when all of the different weights converge to their optimal point (TODO: fix ValueError)

# TODO: in fact, fix everything; having an expert for each point is not the standard implementation

# test run of Winnow algorithm

features = [[-1/2, 1/3], [3/4, -1/4], [1/2, -1/3], [-3/4, 1/4]]

labels = [1, 1, -1, -1]

weights={}  # dictionary to store all updates to weights; record of weights

n = len(features[0])

weights[1] = [([1/n]*n)]*len(features)
    
epsilon = 0.05
rho = 1
eta = 0.025

t=1
    

def mwu_update(feature, label, feature_ind, current_t):
    
    # feature is the point of form (x_i, y_i)
    # weight is of the form (w_0, w_1)
    
    current_weights_all = weights[current_t]
    
    current_weight = current_weights_all[feature_ind]
    
    weight_0 = current_weight[0]
    weight_1 = current_weight[1]
    print("weight_0", weight_0)
    print("weight_1", weight_1)
    
    feature_0 = feature[0]
    feature_1 = feature[1]
    
    # update happens here
    
    new_weight_0 = weight_0 * math.exp(eta*label*feature_0)
    new_weight_1 = weight_1 * math.exp(eta*label*feature_1)

    print("label", label)
    print("exponent_0", eta*label*feature_0)
    print("exponent_1", eta*label*feature_1)

    print("new_weight_0", math.exp(eta*feature_0))
    print("new_weight_1", math.exp(eta*feature_1))
        
    z = new_weight_0 + new_weight_1
    
    new_weight_0 = new_weight_0/z
    new_weight_1 = new_weight_1/z
    
    current_weights_all[feature_ind]=[new_weight_0,new_weight_1]
    print("new weights", [new_weight_0,new_weight_1])
    
    weights[current_t+1] = current_weights_all
    print("updated", weights[current_t+1])
    print()
    
    
    
def predict(t):
    
    num_rounds = 100

    while t < num_rounds:
        
        for ind, feature in enumerate(features):
        
            #print(weights)
        
            w_t = weights[t][ind]
    
            label = labels[ind]
        
            print(t, w_t, feature, label)
            
            dot = np.dot(w_t, feature)
            print(dot)
    
            if dot >= epsilon:
                print("NOTHING CHANGES!!")
                
                current_weights_all = weights[t]
                current_weights_all[ind] = weights[t][ind]
                print("weight_updated", weights[t][ind])
                print()
                
                # if t+1 not in weights:
                #     weights[t+1]=[[], [], [], []]
                # else:
                weights[t+1] = current_weights_all[ind]
            else:
                mwu_update(feature, label, ind, t)
            
            
        # in the end, the present manifestation converged in <100 steps, so this was not necessary
        # if t % 100 == 0:
        #     output_progress(t)
            
        #weights[t+1]=weights[t]
        
        t+=1
            
                
def output_progress(current_t):
    
    print("Step:", current_t, " Weights:", weights[current_t])
            
        
def main():
    
    predict(t) 




if __name__ == "__main__":
    main()
    
    
    
    
    
    
    

    



