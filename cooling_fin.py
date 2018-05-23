
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
import scipy as sp
import matplotlib.pyplot as plt
#%matplotlib inline

class CoolingFinEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):        
        #set up parameters
        self.counter = 0
        self.counter2 = 0
        self.h = 20
        self.k = 400
        self.L = 1
        self.r0 = 0.25
        self.T0 = 500
        self.Tinf = 200
        self.r_max = 0.1
        self.r_min = 0.02
        self.Qmin = 5000
        #Make grid
        self.ml_N = 10
        self.ml_x = np.linspace(0,self.L,self.ml_N)
        self.ml_dx = self.ml_x[1]-self.ml_x[0]
        #Make grid
        self.pde_N = 500
        self.pde_x = np.linspace(0,self.L,self.pde_N)
        self.pde_dx = self.pde_x[1]-self.pde_x[0]
        self.tol = self.r_min/100#if norm of action goes below this we have converged
        #set action and observation (state) spaces
        self.action_space = spaces.Box(low=-self.r_max, high=self.r_max, shape=(self.ml_N,))
        self.observation_space = spaces.Box(low=0, high=100000, shape=(2*self.ml_N+2,))#increase high
        #Randomize initial state
        self.randState()
        self.rwd = np.zeros(1)
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
        for i in range(0,self.ml_N):
            if(self.r[i]>self.r_max):
                self.r[i]=self.r_max
            elif(self.r[i]<self.r_min):
                self.r[i]=self.r_min
        r_interp = sp.interpolate.CubicSpline(self.ml_x,self.r)
        self.pde_r = r_interp(self.pde_x)
        for i in range(0,self.pde_N):
            if(self.pde_r[i]>self.r_max):
                self.pde_r[i]=self.r_max
            elif(self.pde_r[i]<self.r_min):
                self.pde_r[i]=self.r_min
        #compute new state after action
        self.completeState()
        #compute reward function
        #reward = -5000*self.V-200*np.log(1+np.exp((self.Qmin-self.Q)/100))
        #really dont gain anything from the complicated function approximation
        #just have an if statement...
        #maybe leaky ReLU is the best choice of reward function
        reward = -500*self.V+self.Q
        #reward = self.Q
        #reward = self.V
        self.rwd = np.append(self.rwd,reward)
        self.counter2 = self.counter2 + 1
        self.counter = self.counter +1
        if self.counter == 10000:
            print(self.rwd)
            self.counter = 0
            plt.figure()
            plt.plot(self.pde_x,self.pde_r)
            #plt.title('reward: '+str(reward))
            plt.xlabel('x')
            plt.ylabel('Fin Radius')
            plt.ylim((0,self.r_max*1.1))
            plt.figure()
            plt.plot(self.pde_x,self.pde_T)
            #plt.title('reward: '+str(reward))
            plt.xlabel('x')
            plt.ylabel('Temperature')
            plt.figure()
            plt.plot(self.rwd)
            plt.xlabel('Training Step')
            plt.ylabel('Reward')
        return np.array(self.state), reward, done, {}

    def completeState(self):
        #Solve the heat equation
        M = np.zeros((self.pde_N,self.pde_N))
        f = np.zeros((self.pde_N,1))
        M[0,0] = 1
        f[0,0] = self.T0
        M[self.pde_N-1,self.pde_N-1] = 3/(2*self.pde_dx)+self.h/self.k
        M[self.pde_N-1,self.pde_N-2] = -2/self.pde_dx
        M[self.pde_N-1,self.pde_N-3] = 1/(2*self.pde_dx)
        f[self.pde_N-1,0] = self.Tinf*self.h/self.k
        for i in range(1,self.pde_N-1):
            M[i,i+1] = 0.5*self.pde_r[i]/(self.pde_dx**2)+0.25*(self.pde_r[i+1]-self.pde_r[i-1])/(self.pde_dx**2);
            M[i,i] = -self.pde_r[i]/(self.pde_dx**2)-self.h/self.k;
            M[i,i-1] = 0.5*self.pde_r[i]/(self.pde_dx**2)-0.25*(self.pde_r[i+1]-self.pde_r[i-1])/(self.pde_dx**2);
            f[i,0] = -self.h*self.Tinf/self.k;
        self.pde_T = np.linalg.solve(M,f)
        self.pde_T=self.pde_T[:,0]
        T_interp = sp.interpolate.CubicSpline(self.pde_x,self.pde_T)
        self.T = T_interp(self.ml_x)
        #Post process to find V and Q
        A = np.pi*self.pde_r**2;
        #willPrint = np.random.randint(0,100)

        #    print(self.r)
        self.Q = np.trapz(2*np.pi*self.h*np.multiply(self.pde_r,(self.pde_T-self.Tinf)),self.pde_x)
        #self.Q = -self.k*A[0]*(-1.5*self.pde_T[0]+2*self.pde_T[1]-0.5*self.pde_T[2])/self.pde_dx;
        self.V = np.trapz(A,self.pde_x)
        self.state = np.concatenate((self.r,self.T),axis=0)
        self.state = np.append(self.state,self.Q)
        self.state = np.append(self.state,self.V)
        
    def reset(self):
        self.randState()
        return np.array(self.state)
    
    def randState(self):
        #Randomize the state
        n = np.random.randint(2,10)

        xp = np.linspace(0,1,n)
        rp = np.random.rand(n)*(0.5*self.r_max-2*self.r_min)+2*self.r_min
        rs = sp.interpolate.CubicSpline(xp,rp)
        RR = rs(self.ml_x)
        for i in range(0,self.ml_N):
            if(RR[i]>(self.r_max)):
                RR[i]=self.r_max
            elif(RR[i]<(self.r_min)):
                RR[i]=self.r_min
        self.r = RR
        Rp = rs(self.pde_x)
        for i in range(0,self.pde_N):
            if(Rp[i]>(self.r_max)):
                Rp[i]=self.r_max
            elif(Rp[i]<(self.r_min)):
                Rp[i]=self.r_min
        self.pde_r = Rp
        self.completeState()
        
        
    def render(self, mode='human'):
        """n = 83
        #n = np.random.randint(0,100)
        if n == 83:
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(500,500)
                self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
                #rod = rendering.make_capsule(1, .2)
                #rod.set_color(.8, .3, .3)
                #self.pole_transform = rendering.Transform()
                #rod.add_attr(self.pole_transform)
                #self.viewer.add_geom(rod)
                #axle = rendering.make_circle(.05)
                #axle.set_color(0,0,0)
                #self.viewer.add_geom(axle)
                V = np.column_stack((self.ml_x,self.r))
                fin = rendering.draw_polygon(V,filled=True)
                self.viewer.add_geom(fin)
                fname = path.join(path.dirname(__file__), "assets/clockwise.png")
                self.img = rendering.Image(fname, 1., 1.)
                self.imgtrans = rendering.Transform()
                self.img.add_attr(self.imgtrans)
                
            self.viewer.add_onetime(self.img)
            self.pole_transform.set_rotation(self.state[0] + np.pi/2)
            
            if self.last_u:
                self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        """
    def close(self):
        if self.viewer: self.viewer.close()