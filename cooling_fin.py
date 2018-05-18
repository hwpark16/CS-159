
# coding: utf-8

# In[11]:

"""
Simulates steady-state heat transfer in a 1D cooling fin.
Actions are updating geometry to find optimal design
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CoolingFinEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        #set action and observation (state) spaces
        self.action_space = spaces.Box(low=-self.r_max, high=self.r_max, shape=(self.N,))
        self.observation_space = spaces.Box(low=-self.r_max, high=self.T0, shape=(2*self.N+2,))
        
        #set up parameters
        self.h = 20
        self.k = 400
        self.L = 1
        self.r0 = 0.25
        self.T0 = 500
        self.Tinf = 200
        #self.tau = 0.02  # seconds between state updates
        self.r_max = 0.6
        self.r_min = 0.015
        #Make grid
        self.N = 100
        self.x = np.linspace(0,self.L,self.N)
        self.dx = self.x[1]-self.x[0]
        self.tol = r_min#if norm of action goes below this we have converged
        #Randomize initial state
        self.randState()
        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        #high = np.array([
        #    self.x_threshold * 2,
        #    np.finfo(np.float32).max,
        #    self.theta_threshold_radians * 2,
        #    np.finfo(np.float32).max])

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = bool(self.tol >= np.linalg.norm(action))
        #apply action to get new radius
        self.r = self.r + action
        #make sure new radius is within bounds
        for i in range(0,self.N):
            if(self.r[i]>self.r_max):
                self.r[i]=self.r_max
            elif(self.r[i]<self.r_min):
                self.r[i]=self.r_min
        #compute new state after action
        self.completeState()
        #compute reward function
        reward = -5000*self.V-20*np.log(1+np.exp((self.Qmin-self.Q)/10))

        return np.array(self.state), reward, done, {}

    def completeState(self):
        #Solve the heat equation
        M = np.zeros((N,N))
        f = np.zeros((N,1))
        M[0,0] = 1
        f[0,0] = self.T0
        M[N-1,N-1] = 3/(2*self.dx)+self.h/self.k
        M[N-1,N-2] = -2/dx
        M[N-1,N-3] = 1/(2*dx)
        f[N-1,0] = Tinf*h/k
        for i in range(1,N-1):
            M[i,i+1] = 0.5*r[i]/(dx**2)+0.25*(r[i+1]-r[i-1])/(dx**2);
            M[i,i] = -r[i]/(dx**2)-h/k;
            M[i,i-1] = 0.5*r[i]/(dx**2)-0.25*(r[i+1]-r[i-1])/(dx**2);
            f[i,0] = -h*Tinf/k;
        self.T = np.linalg.solve(M,f)
        #Post process to find V and Q
        A = np.pi*r**2;
        self.Q = -k*A[0]*(-1.5*T[0]+2*T[1]-0.5*T[2])/dx;
        self.V = np.trapz(A,x)
        self.state = [self.r,self.T,Q,V]
    
    def randState(self):
        #Randomize the state
        n = np.random.randint(2,10)

        xp = np.linspace(0,1,n)
        rp = np.random.rand(n)*(0.5*self.r_max-2*self.r_min)+2*self.r_min
        x = np.linspace(0,1,self.N)
        rs = sp.interpolate.CubicSpline(xp,rp)
        RR = rs(x)
        for i in range(0,N):
            if(RR[i]>(self.r_max)):
                RR[i]=self.r_max
            elif(RR[i]<(self.r_min)):
                RR[i]=self.r_min
        self.r = RR
        self.completeState()