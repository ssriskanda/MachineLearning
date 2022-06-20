#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:55:26 2020

@author: ssriskanda
"""


q13 = .5
q17 = -.4
q18 = -.6
q28 = .7
count = 0 

while count < 100000:
    #episode 1
    q17 = q17 + .01*(1 + 0 - q17)
    
    #episode 2
    q18 = q18 + .01*(-1 + 0 - q18)
    
    #episode 3
    q13 = q13 + .01*(0 + q28 - q13)
    q28 = q28 + .01*(0 + 0 - q28)
    
    count += 1