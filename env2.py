#! /usr/bin/env python

import csv
import math
import random
import pprint
import time
import msgpack
import torch
import msgpackrpc
from PIL import Image
import numpy as np
import airsim
#import setup_path

SPEEDUP = 2
MOVEMENT_INTERVAL = 0.5 / SPEEDUP

class DroneEnv(object):
    """Drone environment class using AirSim python API"""

    def __init__(self, useDepth=False, goal=np.array([20, 20, -7])):
        self.client = airsim.MultirotorClient()
        self.goal = goal 

        self.last_dist = self.get_distance(self.client.getMultirotorState().kinematics_estimated.position)
        self.quad_offset = (0, 0, 0)
        self.useDepth = useDepth
        self.n_first = 10
        self.goal_eps = 1e1

    def step(self, action):
        """Step"""
        #print("new step ------------------------------")

        self.quad_offset = self.interpret_action(action) #ovo vraća offset quada, dobra se akcija prosljeđuje

        #print("quad_offset: ", self.quad_offset)

        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(
            quad_vel.x_val + self.quad_offset[0],
            quad_vel.y_val + self.quad_offset[1],
            quad_vel.z_val + self.quad_offset[2],
            MOVEMENT_INTERVAL
        ).join()
        collision = self.client.simGetCollisionInfo().has_collided

        time.sleep(0.5 / SPEEDUP)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        
        #if quad_state.z_val < - 7.3:
        #    self.client.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 1).join()

        #result, done = self.compute_reward(quad_state, quad_vel, collision)
        result, done = self.compute_reward2(quad_state, quad_vel, collision)
        state, image = self.get_obs(quad_state)

        return state, result, done, image 

    def reset(self, soft_reset=False):
        self.client.reset()
        self.last_dist = self.get_distance(self.client.getMultirotorState().kinematics_estimated.position)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        if soft_reset:
            x_val, y_val = random.randint(0, self.goal[0]), random.randint(0, self.goal[1])
            quad_state.x_val = x_val
            quad_state.y_val = y_val
        self.client.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 1).join()
        obs, image = self.get_obs(quad_state)
        return obs, image

    def get_obs(self, quad_state):
        if self.useDepth:
            # get depth image
            responses = self.client.simGetImages(
                [airsim.ImageRequest(1, airsim.ImageType.DepthPlanar, pixels_as_float=True)])
            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float)
            img1d = img1d * 3.5 + 30
            img1d[img1d > 255] = 255
            image = np.reshape(img1d, (responses[0].height, responses[0].width)) #ovo je rgb slika koju mi ne koristimo
            
            image_array = Image.fromarray(image).convert("L") 
            
        else:
            # Get rgb image
            responses = self.client.simGetImages(
                [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)] #1 je ime kamere, to je depth kamera
            )
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            image = img1d.reshape(response.height, response.width, 3) 
            image_array = Image.fromarray(image).convert("L")
        
        obs = np.array(image_array)
        obs_dist_angles = self.get_angles(obs)
        q = np.array([quad_state.x_val, quad_state.y_val, quad_state.z_val] )
        obs_dist_angles = np.vstack((q, obs_dist_angles))
        #print(dist_angles)
        return obs_dist_angles, image


    def get_angles (self, image_array):
        n = self.n_first
        f = self.client.simGetFocalLength("1")
        height, width = image_array.shape 
        n_points = height * width
        dist_angles_array=np.zeros((n_points,3)) #forma [d,alpha,beta]
        k = 0
        for i in range(height):
            for j in range (width):
                d = image_array[i,j]
                alpha = math.atan((j-(width/2))/f)
                beta = math.atan((i-(height/2))/f)
                dist_angles_array[k,0] = d
                dist_angles_array[k,1] = alpha
                dist_angles_array[k,2] = beta
                k+=1
       
        sorted_dist_angles_array = dist_angles_array[dist_angles_array[:,0].argsort()]
        first_n_dist_angles=sorted_dist_angles_array[:n,:]

        return first_n_dist_angles
        


    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        #pts = np.array([3, -76, -7]) #ovo se nece hardcodirat nego ce se citati iz yaml fajla
        pts = self.goal
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - pts)
        return dist

    def compute_reward(self, quad_state, quad_vel, collision):
        """Compute reward"""
        reward = -1

        if collision:
            reward = -50
        else:
            dist = self.get_distance(quad_state)
            diff = self.last_dist - dist

            if dist < 10:
                reward = 500
            else:
                reward += diff


            self.last_dist = dist

        done = 0
        if reward <= -10:
            done = 1
            time.sleep(1)
        elif reward > 499:
            done = 1
            time.sleep(1)

        return reward, done
    
    def compute_reward2(self, quad_state, quad_vel, collision):
        ksi = 1
        FI_m = 20
        eta = 4
        R = 0.5 # Robot radius
        rho_0 = 1
        d = 1e9 
        point = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        D = self.get_distance(quad_state)
        FI_g = 1./2 * ksi * D
        
        obstacles, img = self.get_obs(quad_state)

        for i in range(1, len(obstacles)):
            d = min(d, obstacles[i,0])

        FI_o = 0 if d > rho_0 else FI_m * (1 - np.exp(-D**2/R**2)) * ((rho_0 - d) / rho_0) ** eta
        reward = -5 * FI_o - FI_g
        
        done = False

        if (self.get_distance(quad_state) < self.goal_eps):
            done = True
            reward += 1e2

        if (self.quad_offset[2] < -100 or collision): #da sprijecimo odlazak u visinu u nedogled
             reward -= 1e2
        
        if collision:
            done = True

        return reward, done



    def interpret_action(self, action):
        """Interprete action"""
        scaling_factor = 3

        if action == 0:
            self.quad_offset = (scaling_factor, 0, 0)
        elif action == 1:
            self.quad_offset = (-scaling_factor, 0, 0)
        elif action == 2:
            self.quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            self.quad_offset = (0, -scaling_factor, 0)
        elif action == 4:
            self.quad_offset = (scaling_factor, scaling_factor, 0)
        elif action == 5:
            self.quad_offset = (scaling_factor, -scaling_factor, 0)
        elif action == 6:
            self.quad_offset = (-scaling_factor, scaling_factor, 0)
        elif action == 7:
            self.quad_offset = (-scaling_factor, -scaling_factor, 0)
        
        elif action == 8:
            self.quad_offset = (0, 0, scaling_factor)
        elif action == 9:
            self.quad_offset = (0, 0, -scaling_factor)
        elif action == 10:
            self.quad_offset = (scaling_factor, 0, scaling_factor)
        elif action == 11:
            self.quad_offset = (-scaling_factor, 0, scaling_factor)
        elif action == 12:
            self.quad_offset = (0, scaling_factor, scaling_factor)
        elif action == 13:
            self.quad_offset = (0, -scaling_factor, scaling_factor)
        elif action == 14:
            self.quad_offset = (scaling_factor, scaling_factor, scaling_factor)
        elif action == 15:
            self.quad_offset = (scaling_factor, -scaling_factor, scaling_factor)
        
        elif action == 16:
            self.quad_offset = (-scaling_factor, -scaling_factor, scaling_factor)
        elif action == 17:
            self.quad_offset = (-scaling_factor, scaling_factor, scaling_factor)
        elif action == 18:
            self.quad_offset = (scaling_factor, 0, -scaling_factor)
        elif action == 19:
            self.quad_offset = (-scaling_factor, 0, -scaling_factor)
        elif action == 20:
            self.quad_offset = (0, scaling_factor, -scaling_factor)
        elif action == 21:
            self.quad_offset = (0, -scaling_factor, -scaling_factor)
        elif action == 22:
            self.quad_offset = (scaling_factor, scaling_factor, -scaling_factor)
        elif action == 23:
            self.quad_offset = (scaling_factor, -scaling_factor, -scaling_factor)
        
        elif action == 24:
            self.quad_offset = (-scaling_factor, scaling_factor, -scaling_factor)
        elif action == 25:
            self.quad_offset = (-scaling_factor, -scaling_factor, -scaling_factor)

        return self.quad_offset
