# -*- coding: utf-8 -*-
"""
Created on Wed May 18 07:58:59 2022

@author: Administrator
"""

import numpy as np
import os
import shutil 


for i in [0,1,2,3,4,5,6,7,8,9]:
    

    shutil.copytree("directory_to_be_copied_for_different_seeds", str(i)) 

