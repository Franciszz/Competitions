# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 08:38:55 2018

@author: Franc
"""
import numpy as np
from PIL import Image
imgdir = 'C:/Users/Franc/Desktop/Ref/# 机器学习小组/树模型/Slides/graphics'
captain = Image.open(imgdir+'/team_blue.jpg')
captain_rev = Image.open(imgdir+'/team_blue2.jpg')
iron = Image.open(imgdir+'/team_red22.jpg')
iron = Image.open(imgdir+'/team_red.jpg')
widow = Image.open(imgdir+'/team_green2.jpg')
thor = Image.open(imgdir+'/team_purple.jpg')
witch = Image.open(imgdir+'/team_yellow.jpg')
thor_iron = Image.fromarray(np.concatenate([np.array(iron),np.array(thor)],1))
thor_iron.save(imgdir+'/thor_iron.jpg')

widow_witch = Image.fromarray(np.concatenate([np.array(witch),np.array(widow)],1))
widow_witch.save(imgdir+'/widow_witch.jpg')


cap_rion = Image.fromarray(np.concatenate([np.array(iron),np.array(captain_rev)],1))
cap_rion.save(imgdir+'/cap_rion.jpg')
