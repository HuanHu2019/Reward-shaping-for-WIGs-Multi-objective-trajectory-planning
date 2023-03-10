import numpy as np
import os
import shutil 
import os
import re

import sys

import time

path=os.path.dirname(sys.argv[0])

fff = os.path.basename(path)

print('fff',fff)


cur_file  = os.path.abspath(__file__)
cur_dir = cur_file.rsplit("/", 1)[0] 
print('cur_dir',cur_dir)

cur_file_name = __file__[:-2]

print('cur_file_name',cur_file_name)



seed_int = [int(fff)]

allguys = []


he1 = [[0,0,0]]

he2 = [ [1,0,0], [2,0,0], [3,0,0], [4,0,0]]

he3 = [ [0,1,0], [0,2,0], [0,3,0], [0,4,0]]

he4 = [ [0,0,1], [0,0,2], [0,0,3], [0,0,4]]


jihemen =  he1 +  he2+ he3 + he4

for fuckjihe in jihemen:
    
    i = fuckjihe[0]
    
    j = fuckjihe[1]
    
    m = fuckjihe[2]
    
    
    for n in seed_int: 

        xinwenjianjiaming =  'd' + str(i) + 'd' + 's' + str(j) + 's' +  'g' + str(m) + 'g' + 'r'  + str(n) +   'r'
        
        shutil.copytree("directory_to_be_copied_for_different_subcase_directory", str(i) + 'd' + str(j)  + 's'  + str(m) + 'g'  +'/' + xinwenjianjiaming)  
        
        allguys.append(str(i) + 'd' + str(m) + 'g' + str(j)  + 's' +'/' + xinwenjianjiaming)









