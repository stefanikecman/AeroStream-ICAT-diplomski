#from agent import Agent #ovo ako testriamo stare obs
from agent2 import Agent #ako testiramo nove obs
#from ddqn_agent import DDQN_Agent

#ja radim samo sa agent!!!
if __name__ == "__main__":
    agent = Agent(useGPU=True, useDepth=True)
    agent.train()
    #ddqn_agent = DDQN_Agent(useDepth=False)
    #ddqn_agent.train()
