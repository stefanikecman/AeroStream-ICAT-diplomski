import numpy as np

from agent import Agent #ovo ako testriamo stare obs
#from agent2 import Agent #ako testiramo nove obs
#from ddqn_agent import DDQN_Agent
import time

#ja radim samo sa agent!!!
if __name__ == "__main__":
    goal = np.array([50, 50, -50])
    agent = Agent(useGPU=True, useDepth=True, goal=goal)
    N = 10000
    state,_ = agent.env.reset()
    for i in range(N):  
        print("Step {0}".format(i))
        # select action with policy
        action = agent.act_test(state)
        start_time = time.time()
        state, reward, done, _ = agent.env.step(action)
        final_time = time.time() - start_time
        print("ms: ", final_time * 1000)
        print(state[0,:])
        if done:
            break
    #agent.train()
    #ddqn_agent = DDQN_Agent(useDepth=False)
    #ddqn_agent.train()
