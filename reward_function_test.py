import math

def f1 (dist_goal, min_dist_obs):

    dist_reward=1-dist_goal**0.4
    dist_discount=1-max(min_dist_obs,1)**(1/max(min_dist_obs,0.2))
    
    return dist_discount*dist_reward

test1= f1(10,5)
test2=f1(10,2)
test1= f1(10,1.5)
test2=f1(10,0.8)
test3=f1(10,0.5)
test4=f1(10,0.3)
test5=f1(10,0.2)


print(test1)
print(test2)
print(test3)
print(test4)
print(test5)