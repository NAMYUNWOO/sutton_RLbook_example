import numpy as np
import math
import pandas as pd

r_profit = 10
r_cost = -2
max_move = 5
gamma = 0.9
max_capa1 = 20
max_capa2 = 20
probposs = lambda l,n : math.pow(math.e,-l)*math.pow(l,n)/math.factorial(n)
prob_req1 = lambda n : probposs(3,n)
prob_req2 = lambda n : probposs(4,n)
prob_ret1 = lambda n : probposs(3,n)
prob_ret2 = lambda n : probposs(2,n)
actions = list(range(-max_move,max_move+1))
def state_iter(max_capa1,max_capa2):
    for n in range(max_capa1+1):
        for m in range(max_capa2+1):
            yield (n,m)
            
def policy_evaluation(values,policies):
    values_ = np.copy(values)
    for state in state_iter(max_capa1 , max_capa2):
        s_1,s_2 = state
        values_at_state = 0
        values_[s_1][s_2] = values_at_state
        a = policies[s_1][s_2]
        s_1_a,s_2_a = s_1 + a, s_2 - a 
        carout_1,carout_2 = max_capa1 - s_1 , max_capa2 -s_2
        if s_1_a < 0 or s_1_a > max_capa1 or s_2_a < 0 or s_2_a > max_capa2:
            continue
        for req_1 in range(s_1_a+1):
            for req_2 in range(s_2_a+1):
                for ret_1 in range(carout_1+1):
                    for ret_2 in range(carout_2 + 1):
                        ns_1 = s_1_a + ret_1 - req_1
                        ns_2 = s_2_a + ret_2 - req_2
                        over_capa1 = ns_1 - min(max_capa1,ns_1)
                        over_capa2 = ns_2 - min(max_capa2,ns_2)
                        ns_1 += over_capa2
                        ns_2 += over_capa1
                        #print("state",state);print("action",a) ;print(ret_1,req_1,ret_2,req_2);print("next state",(ns_1,ns_2));print("----------------------")
                        reward = abs(a)*r_cost+(req_1+req_2)*r_profit
                        values_[s_1][s_2] +=  prob_ret1(ret_1)*prob_req1(req_1)*prob_ret2(ret_2)*prob_req2(req_2)*(reward + gamma*values[ns_1][ns_2])

    return values_

def policy_improvement(values,policies):
    for state in state_iter(max_capa1 , max_capa2):
        s_1,s_2 = state
        action_values = np.ones(len(actions))*-np.Infinity
        for idx,a in enumerate(actions):
            value_at_action = 0
            s_1_a,s_2_a = s_1 + a, s_2 - a 
            carout_1,carout_2 = max_capa1 - s_1 , max_capa2 -s_2
            if s_1_a < 0 or s_1_a > max_capa1 or s_2_a < 0 or s_2_a > max_capa2:
                continue
            for req_1 in range(s_1_a+1):
                for req_2 in range(s_2_a+1):
                    for ret_1 in range(carout_1 + 1):
                        for ret_2 in range(carout_2 + 1):
                            ns_1 = s_1_a + ret_1 - req_1
                            ns_2 = s_2_a + ret_2 - req_2
                            over_capa1 = ns_1 - min(max_capa1,ns_1)
                            over_capa2 = ns_2 - min(max_capa2,ns_2)
                            ns_1 += over_capa2
                            ns_2 += over_capa1
                            #print("state",state);print("action",a) ;print("next state",next_state);print(ret_1,req_1,ret_2,req_2);print("----------------------")
                            reward = abs(a)*r_cost+(req_1+req_2)*r_profit
                            value_at_action +=  prob_ret1(ret_1)*prob_req1(req_1)*prob_ret2(ret_2)*prob_req2(req_2)*(reward + gamma*values[ns_1][ns_2])
            action_values[idx] = value_at_action
        policies[s_1][s_2] = actions[np.argmax(action_values)]
    return policies

def main():
    values = np.zeros([max_capa1+1,max_capa2+1])
    policies = np.zeros([max_capa1+1,max_capa2+1],dtype=int)
    for i in range(10):
        print("iter %d"%i)
        np.savetxt("value_%d.csv"%i, values, delimiter=",")
        np.savetxt("policy_%d.csv"%i, policies.astype(int), delimiter=",")
        for _ in range(4):
            values = policy_evaluation(values,policies)
        policies = policy_improvement(values,policies)
    

if __name__ == "__main__":
    main()

