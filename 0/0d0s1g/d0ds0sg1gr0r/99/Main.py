import numpy as np
import os
import time
import torch
import numpy as np
import numpy.random as rd
import math
from copy import deepcopy
from numpy import cos,sin,tan,sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import matplotlib

from scipy import interpolate
from scipy.optimize import curve_fit

import glob
import shutil

import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

from aeropy_fix import jiayou,chazhi,sampling,zhaoweizhi,runaero,Math_integrate,jixu
from agent import *
import re



feifei = 0.6

yinbianquan_shangxian = 0.7

yinbianquan_xiaxian = 0.5


#获取文件名（含后缀）
name=os.path.basename(__file__)
print(name)

#去掉文件后缀，只要文件名称
# name=os.path.basename(__file__).split(".")[0]
# print(name)


your_daddad= os.path.basename(parent_path)

d = re.findall(r"d(.+?)d",your_daddad)
#d = int(d[0]) 
d = int(d[0]) 
print('d',d)

s = re.findall(r"s(.+?)s",your_daddad)
#s = int(s[0]) 
s = int(s[0] )
print('s',s)

j = re.findall(r"g(.+?)g",your_daddad)
#j = int(j[0])
j = int(j[0])
print('j',j)

randomseed = re.findall(r"r(.+?)r",your_daddad)
seed_int = int(randomseed[0]) - 1
print('seed_int',seed_int)


path=os.path.dirname(sys.argv[0])

fff = os.path.basename(path)

suanlixuhao = 99 #int(fff)

#xishuzonghe = d + s + j

xishuzonghe = 1


if d == 0:
    
    a_1 = 1
    
    a_2 = 1

if d == 1:
    
    a_1 = 0.5
    
    a_2 = 1


if d == 2:
    
    a_1 = 1
    
    a_2 = 2


if d == 5:
    
    a_1 = 1
    
    a_2 = 0.5


if d == 6:
    
    a_1 = 2
    
    a_2 = 1







if s == 0:
    
    b_1 = 1
    
    b_2 = 1


if s == 1:
    
    b_1 = 0.5
    
    b_2 = 1


if s == 2:
    
    b_1 = 1
    
    b_2 = 2


if s == 5:
    
    b_1 = 1
    
    b_2 = 0.5


if s == 6:
    
    b_1 = 2
    
    b_2 = 1
















if j == 0:
    
    c_1 = 1
    
    c_2 = 1



if j == 1:
    
    c_1 = 0.5
    
    c_2 = 1

if j == 2:
    
    c_1 = 1
    
    c_2 = 2

if j == 5:
    
    c_1 = 1
    
    c_2 = 0.5

if j == 6:
    
    c_1 = 2
    
    c_2 = 1











if  d==3 or d==4 or d==7 or d==8 :
    
    a_1 = 1
    
    a_2 = 1
    
    if d == 3:
        
        a_a_1 = 0.5
        
        a_a_2 = 1
    
    if d == 4:
        
        a_a_1 = 1
        
        a_a_2 = 2
        
    if d == 7:
        
        a_a_1 = 1
        
        a_a_2 = 0.5
    
    if d == 8:
        
        a_a_1 = 2
        
        a_a_2 = 1  
        
       


if s==3 or s==4 or s==7 or s==8 :
    
    b_1 = 1
    
    b_2 = 1
    
    if s == 3:
        
        b_b_1 = 0.5
        
        b_b_2 = 1
    
    if s == 4:
        
        b_b_1 = 1
        
        b_b_2 = 2
       
        
       
    if s == 7:
        
        b_b_1 = 1
        
        b_b_2 = 0.5
    
    if s == 8:
        
        b_b_1 = 2
        
        b_b_2 = 1  
       
        
       

if j==3 or j==4 or j==7 or j==8:

    c_1 = 1
    
    c_2 = 1
    
    if j == 3:
        
        c_c_1 = 0.5
        
        c_c_2 = 1
    
    if j == 4:
        
        c_c_1 = 1
        
        c_c_2 = 2
       

    if j == 7:
        
        c_c_1 = 1
        
        c_c_2 = 0.5
    
    if j == 8:
        
        c_c_1 = 2
        
        c_c_2 = 1  



initial = feifei #0.2

end = feifei

flyingh = 0.4* end

shijianbu = 0.02
Tzhong = 2.943000000000000*0.6/0.25
flap_max = 9.99999


alltang = 10000

lhou=2*0.4
lqian=1.5*0.4
gabove=0.31*0.4
ji=np.load(os.path.abspath(os.path.join(os.getcwd(), "../../../..")) + '//'+ 'datacollection.npy')

vmin=5
vmax=18
lenthsudu=100
suduzu=np.linspace(vmin,vmax,lenthsudu)
lentheta=100
thetazu=np.linspace(-11.5,5.5,lentheta);
hmax=0.4*1.31;  ###相当于1的相对飞高
hmin=0.4*0.41;  ###相当于0.1的相对飞高
lenh=100
hzu=np.linspace(hmin,hmax,lenh)
lenflap=100
flapzu=np.linspace(-10,10,lenflap)

chushisudubi = 1

allji = [0.3,0.6,0.9]

v_0_ji = v_aim_ji = [8.59920292,8.879512099,9.124996488]

Thrust_0_ji = Thrust_end_ji = [1.277167533,1.54278313,1.615092432]

theta_0_ji = theta_end_ji = [1.805507789,1.77565838,1.773445899]

h_0_ji = h_aim_ji = [0.243938438,0.363940457,0.483940605]

flap_0_ji = flap_end_ji = [-2.622866624,-2.303687751,-2.093826378]

feigao_0_ji = feigao_aim_ji =[0.3*0.4,0.6*0.4,0.9*0.4]

for i in  range(3):
    
    if int(10*initial)==int(10*allji[i])  :
        
        v_0 = v_0_ji[i]
        
        Thrust_0 = Thrust_0_ji[i]
        
        feigao_0 = feigao_0_ji[i]
        
        theta_0 = theta_0_ji[i]
        
        h_0 = h_0_ji[i]
        
        flap_0 = flap_0_ji[i]


mingzi = curr_path+'/'+'run.py'

fnew = open( mingzi,'w',encoding='utf-8')

file1=  open(parent_path+'/'+"part1.py",'r',encoding='utf-8')

file3=  open(parent_path+'/'+"part3.py",'r',encoding='utf-8')

part1 = file1.readlines()

part3 = file3.readlines()

for t in range(len(part1)):
    
    fnew.write(part1[t])
    
fnew.write('\n')                

if seed_int == 0:
    
    fnew.write('        self.random_seed = 0'  +'\n')

else:
    
    fnew.write('        self.random_seed = 2**' + str( seed_int )   +'\n')

for tt in range(len(part3)):
    
    fnew.write(part3[tt])  
    
file1.close()

file3.close()

fnew.close()


if os.path.exists('run.py'):
    
    print(os.getcwd())
    
    from run import *
    
    print('chenggongle')

else:
    
    print('fuck')

'''
jiao_ori =  20

xiaoduan =  v_0*shijianbu           

kaiduan = 40

middle = 50

M = 25


allstep = int((kaiduan * 2 + middle + M * 2)*1.2)

print('allstep',allstep)

kaiduan_chang = kaiduan * xiaoduan

middle_chang = middle * xiaoduan

M_chang = M * xiaoduan
'''


jiao_ori = 20

xiaoduan =  v_0*shijianbu           

kaiduan = 24

middle = 12

jieticha = 0.4 * 0.5

M_chang = jieticha/ sin(jiao_ori*np.pi/180)

allstep = 64


print('allstep',allstep)




kaiduan_chang = kaiduan * xiaoduan

middle_chang = middle * xiaoduan








L1 = kaiduan_chang + 1 
L2 = M_chang
L3 = middle_chang                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
L4 = M_chang


p1x = -1
p1y = 0

p2x = p1x + L1
p2y = p1y + 0

p3x = p2x + L2*cos(-jiao_ori*np.pi/180)
p3y = p2y + L2*sin(-jiao_ori*np.pi/180)

# p4x = p3x + 100
# p4y = p3y + 0


p4x = p3x + L3
p4y = p3y + 0

p5x = p4x + L4*cos(jiao_ori*np.pi/180)
p5y = p4y + L4*sin(jiao_ori*np.pi/180)

p6x = p5x + 160
p6y = p5y

X_ji = [p1x,p2x,p3x,p4x,p5x,p6x]
Y_ji = [p1y,p2y,p3y,p4y,p5y,p6y]

# X_ji = [p1x,p2x,p3x,p4x]
# Y_ji = [p1y,p2y,p3y,p4y]


# X_duandian_1 = p2x 

# X_duandian_2 = p5x


X_duandian = 0.5*( p3x + p4x )


x_shangxian = allstep*1.25 * xiaoduan

y_shangxian = max(Y_ji) + 1.0*0.4

y_xiaxian = min(Y_ji) +0.3*0.4


    
jieshoubu = allstep-2

alpha_0 = theta_0

NavDDTheta= np.deg2rad(alpha_0)
NavDDPsi=0
NavDDPhi=0
 
MatDDC_g2b = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
MatDDC_g2b[0,0] = np.cos(NavDDTheta)*np.cos(NavDDPsi)
MatDDC_g2b[0,1] = np.cos(NavDDTheta)*np.sin(NavDDPsi)
MatDDC_g2b[0,2] = -np.sin(NavDDTheta)
MatDDC_g2b[1,0] = np.sin(NavDDPhi)*np.sin(NavDDTheta)*np.cos(NavDDPsi) - np.cos(NavDDPhi)*np.sin(NavDDPsi)
MatDDC_g2b[1,1] = np.sin(NavDDPhi)*np.sin(NavDDTheta)*np.sin(NavDDPsi) + np.cos(NavDDPhi)*np.cos(NavDDPsi)
MatDDC_g2b[1,2] = np.sin(NavDDPhi)*np.cos(NavDDTheta)
MatDDC_g2b[2,0] = np.cos(NavDDPhi)*np.sin(NavDDTheta)*np.cos(NavDDPsi) + np.sin(NavDDPhi)*np.sin(NavDDPsi)
MatDDC_g2b[2,1] = np.cos(NavDDPhi)*np.sin(NavDDTheta)*np.sin(NavDDPsi) - np.sin(NavDDPhi)*np.cos(NavDDPsi)
MatDDC_g2b[2,2] = np.cos(NavDDPhi)*np.cos(NavDDTheta)

Vx0 = v_0

Vz0 = 0

EarthDDG = 9.81

z0 = - h_0

transform_sudu =np.dot(MatDDC_g2b,np.array([Vx0,0,Vz0]).T)

transform_jiasudu=np.dot(MatDDC_g2b,np.array([0,0,EarthDDG]).T) 

state_0 = np.array([alpha_0,0,transform_sudu[0],transform_jiasudu[0],transform_sudu[2],transform_jiasudu[2],0,0,Vx0,Vz0,0,z0])  ###朝向下为正方向


nnum = 0 
    
    
class Fuckingfly:
    
    def __init__(self,initia_state_ori, Tzhong,    flap_max,    initial,  shijianbu):
                
        self.state_0 = initia_state_ori[[0,2,3,4,5,6,7,10,11]] ##输出使用  z是11
 
        self.initia_state_ori = initia_state_ori  ##
        
        self.state_ori = initia_state_ori
        
        self.max_step = allstep
        
        self.env_name = 'fuck'
        
        self.xunlian = 0
        
        self.state_dim = 9
        
        self.action_dim= 2
                
        self.if_discrete=False
        
        self.target_return=10000000
        
        self.Tzhong = Tzhong
        self.flap_max = flap_max
        self.initial = initial
        self.shijianbu = shijianbu

        self.jilustates = np.append(initia_state_ori,[0,0,0,0,0,0]) # 两个新加的状态、reward、两个控制量和飞高
     
        self.shineng_last = 0    

        
        self.wave_num = -1
        
        self.z_last = 0
        
        self.x_last = 0
        
        self.z_shuiping = 0  
        
        self.dijibu = 0
        
        


    def bo(self,x):
        
        f = interpolate.interp1d(self.hengbeisaierdian, self.zongbeisaierdian, kind='slinear')
            
        Z = f(x)
        
        return Z 


    def dixing(self,x):

        Z = self.bo(x)
                     
        return Z  
    
    

    def op_xing(self,x):
        
        Z = self.bo(x) + 0.4*initial
                     
        return Z  
    

    
        
    def reset(self,seed_gei=None):
        
        self.dijibu = 0
        
       
        self.state_ori = self.initia_state_ori  # 
        
        #state_0_yuan = self.state_0
        
        if seed_gei==None:
            
            self.T_wave =0#A
                        
        else:
            
            self.T_wave = 0
            

        self.hengbeisaierdian =np.array(X_ji)
        
               
        self.zongbeisaierdian = np.array(Y_ji)

        

  
        self.X = np.array(X_ji)
        
        self.Z = np.array(Y_ji)
            
        self.xz = np.vstack((self.X,self.Z)).T    
  




        Theta_new = self.state_ori[0]
        
        Ggaodu = -1* self.state_ori[11]
        
        Xgaodu = self.state_ori[10]
   
        xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
        
        zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
                
        xhua_n_should = 0
        
        zhua_n_should = self.dixing(xhua_n_should) + initial*0.4
        
        Ggaodu_should_fu = zhua_n_should - zhua_n + Ggaodu
        
        Gx_should_fu =  xhua_n_should - xhua_n + Xgaodu
        
        self.state_ori[10] = Gx_should_fu
        
        self.state_ori[11] = - Ggaodu_should_fu

        L1  = zhua_n - self.dixing(xhua_n)
        
        self.state_0 = self.state_ori[[0,2,3,4,5,6,7,10,11]]

        state_0_wave =  self.state_0

        self.jilustates = np.append(self.state_ori,[L1,0,0,0,0,0])

        self.z_last = zhua_n_should

        self.x_last = xhua_n_should
         
        self.yiduan__cr, self.erduan__cr, self.yiduan__spd, self.erduan__spd,self.yiduan__bnd, self.erduan__bnd = 0,0,0,0,0,0
        
        self.yiduan__cr_rd, self.erduan__cr_rd, self.yiduan__spd_rd, self.erduan__spd_rd,self.yiduan__bnd_rd, self.erduan__bnd_rd = 0,0,0,0,0,0
      
        return state_0_wave.astype(np.float32)    
    
    

    
    def step(self, action):
        
        self.dijibu = self.dijibu + 1
        
        state_next_ori, reward_wancheng, done, fuck_fu = envstep(self.state_ori, action, self.Tzhong,    self.flap_max,  self.initial,    self.shijianbu,   self.jilustates, self.dixing)
        
        self.state_ori = state_next_ori  # state_next_ori[10]是x
        
        indexyao = [0,2,3,4,5,6,7,10,11]
        
        state_next_yuan = state_next_ori[indexyao]


        Theta_new = state_next_ori[0]
        
        Ggaodu = -1* state_next_ori[11]
        
        Xgaodu = state_next_ori[10]
   
        xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
        
        zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
         
        xiangduigaodu = zhua_n - self.dixing(xhua_n)
        
        fuck_fu[-6] = xiangduigaodu   # 实际的相对高度
        
        state_next_wave = state_next_yuan

        reward_wancheng = 0




        if  d==3 or d==4 or d==7 or d==8   or s==3 or s==4 or s==7 or s==8  or j==3 or j==4 or j==7 or j==8  :
            

            if xhua_n <  X_duandian :
                
                A_1, A_2, B_1, B_2, C_1, C_2 =  a_1, a_2, b_1, b_2, c_1, c_2
                                
            #if xhua_n <= X_duandian_2 and xhua_n >= X_duandian_1 :
                
            if xhua_n >= X_duandian :
                
                if d==3 or d==4 or d==7 or d==8:
                    
                    A_1, A_2, B_1, B_2, C_1, C_2 =  a_a_1, a_a_2, b_1, b_2, c_1, c_2
                    
                
                if s==3 or s==4 or s==7 or s==8:
                    
                    A_1, A_2, B_1, B_2, C_1, C_2 =  a_1, a_2, b_b_1, b_b_2, c_1, c_2
                    
                
                if j==3 or j==4 or j==7 or j==8:
                    
                    A_1, A_2, B_1, B_2, C_1, C_2 =  a_1, a_2, b_1, b_2, c_c_1, c_c_2
                
                
                
            
            # if xhua_n > X_duandian_2 :
                
            #     A_1, A_2, B_1, B_2, C_1, C_2 =  a_1, a_2, b_1, b_2, c_1, c_2

        else:
            
            A_1, A_2, B_1, B_2, C_1, C_2 =  a_1, a_2, b_1, b_2, c_1, c_2


    
 
        if ~np.isclose(xhua_n, self.x_last):
            
            gensuijiang, sudujiang, yinbijiang, piancha_1, piancha_2, piancha_3 = shinengjisuan(xiangduigaodu,xhua_n, self.x_last,a_1=A_1, a_2=A_2, b_1=B_1, b_2=B_2, c_1=C_1, c_2=C_2)
            
            shinengnow =  gensuijiang + sudujiang + yinbijiang 
            
        else:
            
            shinengnow = 0  
            
            
            gensuijiang = 0
            
            sudujiang = 0
            
            yinbijiang = 0
            
            piancha_1, piancha_2, piancha_3 = 0, 0, 0
     
        #reward = shinengnow - 0*self.shineng_last # + reward_bu # + (jiaodujiangli(state_next_yuan[0],L2) + sudujiangli(state_next_ori[8],state_next_ori[9],L2))*0.25
        
        
        

        
    
        if xhua_n <  X_duandian :
            
            self.yiduan__cr += piancha_1
            
            self.yiduan__spd += piancha_2
            
            self.yiduan__bnd += piancha_3
            
            self.yiduan__cr_rd += gensuijiang
            
            self.yiduan__spd_rd += sudujiang
            
            self.yiduan__bnd_rd +=  yinbijiang 

            
            

        #if xhua_n <= X_duandian_2 and xhua_n >= X_duandian_1 :
            
        if xhua_n >= X_duandian :
        
        
            self.erduan__cr += piancha_1
            
            self.erduan__spd += piancha_2
            
            self.erduan__bnd += piancha_3
            
            self.erduan__cr_rd += gensuijiang
            
            self.erduan__spd_rd += sudujiang
            
            self.erduan__bnd_rd +=  yinbijiang 
 
        
        
        
        
        
        reward = shinengnow


        reward_out = reward + reward_wancheng
        
        
        
        
        fuck_fu[-4] = reward
        
        self.shineng_last = shinengnow
        
        if ~np.isclose(xhua_n, self.x_last):
    
            self.jilustates = np.vstack((self.jilustates, fuck_fu ))
    
        self.z_last = zhua_n

        self.x_last = xhua_n        
        
        
        return state_next_wave.astype(np.float32) , reward_out , done , [gensuijiang, sudujiang, yinbijiang, piancha_1, piancha_2, piancha_3]
    
    
    
    def quanrender(self,casenum,episode_return, jiangji_part1, jiangji_part2, jiangji_part3, piancha_1, piancha_2, piancha_3):
        
        
        quanhuatu(self.jilustates, self.Tzhong, self.flap_max, self.initial, self.shijianbu, self.xz, casenum,episode_return, self.dixing,self.X,self.Z, jiangji_part1, jiangji_part2, jiangji_part3, piancha_1, piancha_2, piancha_3 , 
                  self.yiduan__cr, self.erduan__cr, self.yiduan__spd, self.erduan__spd,self.yiduan__bnd, self.erduan__bnd,              
                  self.yiduan__cr_rd, self.erduan__cr_rd, self.yiduan__spd_rd, self.erduan__spd_rd,self.yiduan__bnd_rd, self.erduan__bnd_rd)
        
    def render(self,casenum,episode_return, jiangji_part1, jiangji_part2, jiangji_part3, piancha_1, piancha_2, piancha_3):
        
        huatu(self.jilustates, self.Tzhong, self.flap_max, self.initial, self.shijianbu, self.xz, casenum,episode_return, self.dixing,self.X,self.Z, jiangji_part1, jiangji_part2, jiangji_part3, piancha_1, piancha_2, piancha_3,
                  self.yiduan__cr, self.erduan__cr, self.yiduan__spd, self.erduan__spd,self.yiduan__bnd, self.erduan__bnd,              
                  self.yiduan__cr_rd, self.erduan__cr_rd, self.yiduan__spd_rd, self.erduan__spd_rd,self.yiduan__bnd_rd, self.erduan__bnd_rd)
        
        
        

    
def shinengjisuan(xiangduigaodu,x_now,x_last,a_1, a_2, b_1, b_2, c_1, c_2):
   
        fenmu = 0.4         
            
        gensuijiang =  1 - abs(xiangduigaodu-flyingh)/fenmu
        
        piancha_1 = gensuijiang
        
        
        
        sudujiang = (x_now-x_last)/(vmax*shijianbu)
        
        piancha_2 =  sudujiang
        

        
        fenmu_2 = 0.4*(yinbianquan_shangxian-yinbianquan_xiaxian)
        
        zonghereward = 1
        
        piancha_3 = 0
            
        if xiangduigaodu > 0.4*yinbianquan_shangxian:
            
            zonghereward = 1-(xiangduigaodu-0.4*yinbianquan_shangxian)/fenmu_2
            
            #piancha_3 = xiangduigaodu-0.4*yinbianquan_shangxian
        

        if xiangduigaodu < 0.4*yinbianquan_xiaxian:
            
            zonghereward = 1-(0.4*yinbianquan_xiaxian - xiangduigaodu)/fenmu_2
            
            #piancha_3 = 0.4*yinbianquan_xiaxian - xiangduigaodu
                
        piancha_3 = zonghereward
        
        return a_1*((1+gensuijiang)**a_2), b_1*((1+sudujiang)**b_2), c_1*((1+zonghereward)**c_2), piancha_1, piancha_2, piancha_3


    

def envstep(statearg, actionarg, Tzhong,  flap_max,  initial,  shijianbu,  feixingjilu, dixing  ):

    done=False
   
    flap,Thrust = actionarg[0]*flap_max, (actionarg[1]+1) * Tzhong/2
    
    
    
    
    Theta = statearg[0]
    Thetadot = statearg[1]
    u=statearg[2]
    udot=statearg[3]
    w=statearg[4]
    wdot=statearg[5]
    q=statearg[6]
    qdot=statearg[7]
    Vx=statearg[8]
    Vz=statearg[9]  ##朝下为正
    x=statearg[10]
    
    z_wave = -dixing(x)  ## zheng
    
    z= statearg[11] - z_wave
    
    Ggaoduold = -z
    
    feigaoold=Ggaoduold-gabove*math.cos(Theta*math.pi/180)
    
    #print('feigaoold',feigaoold)
    
    cost_1 = 0
    
    Thetasuan= np.arctan(Vz/Vx)+Theta 
    
    if Thetasuan+flap>10 or Thetasuan+flap<-10:
        
       # print('fuck Thetasuan+flap')
        
        done = True
                
        cost_1 = -np.clip((abs(Thetasuan+flap)-10)/10,0,1)
        
        next_state = statearg
        
        feigao = feigaoold
                
        cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8=-1,-1,-1,-1,-1,-1,-1
        
        cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 
        
        temp = np.append(np.zeros(0+1),[cost,0,flap,Thrust, feigao ])
        
        return next_state, cost, done, np.append(next_state,temp)

    if np.isnan(flap):
    
        done = True
                
        cost_1 = -1
        
        next_state = statearg
        
        feigao = feigaoold
                
        cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8=-1,-1,-1,-1,-1,-1,-1
        
        cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 
        
        temp = np.append(np.zeros(0+1),[cost,0,flap,Thrust, feigao ])
        
        return next_state, cost, done, np.append(next_state,temp)



    if (-z)>=hmax:
        #print('fuck_no')
        
        signal_ori=chazhi(Vx,Thetasuan,hmax,flap,suduzu,thetazu,hzu,flapzu,ji)
        
        #print('fuck_yes')
        
    else:
        
        signal_ori=chazhi(Vx,Thetasuan,-1*z,flap,suduzu,thetazu,hzu,flapzu,ji)
    
    
    signal_sec=jiayou(Theta,-1*z,Thrust)
    signal=signal_ori+signal_sec
     
    Fx=signal[0]
    Fz=-1*signal[1]
    My=signal[2]
     
    Fy=0.  
    Mx=0.  
    Mz=0.
    Phi,Phidot=0.,0.   
    Psi,Psidot=0.,0.    
    v,vdot=0.,0.         
    p,pdot=0.,0.  
    r,rdot=0.,0.     
    Vy=0.   
    y=0.    
         
    outcome=runaero(Fx, Fy, Fz,Mx,My,Mz, \
             Phi,Theta,Psi,Phidot,Thetadot,Psidot, \
             u,v,w,udot,vdot,wdot, \
             p,q,r,pdot,qdot,rdot, \
             Vx,Vy,Vz,  \
             x,y,z,shijianbu,20,-20)
 
    Theta_new=outcome[1]
    Thetadot_new=outcome[4]
    u_new=outcome[6]
    udot_new=outcome[9]
    w_new=outcome[8]
    wdot_new=outcome[11]
    q_new=outcome[13]
    qdot_new=outcome[16]
    Vx_new=outcome[18]
    Vz_new=outcome[20]
    x_new=outcome[21]
    z_new=outcome[23]
    
    next_state=np.array([Theta_new,Thetadot_new,u_new,udot_new,w_new,wdot_new,q_new,qdot_new,Vx_new,Vz_new,x_new,z_new + z_wave ])

    jieshoubu = allstep-2
    
    if x_new > (jieshoubu+2) * shijianbu * 20:
        
            done = True
            
            next_state = statearg
            
            feigao = feigaoold       
            
            cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8=-1,-1,-1,-1,-1,-1,-1
        
            cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 
        
            temp = np.append(np.zeros(0+1),[cost,0,flap,Thrust, feigao ])
        
            return next_state, cost, done, np.append(next_state,temp)
                
    if x_new < -1 * shijianbu * 5:
        
            done = True
            
            next_state = statearg
            
            feigao = feigaoold       
            
            cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8=-1,-1,-1,-1,-1,-1,-1
        
            cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 
        
            temp = np.append(np.zeros(0+1),[cost,0,flap,Thrust, feigao ])

            return next_state, cost, done, np.append(next_state,temp)            
    
    z_new_wave = -dixing(x_new)
     
    Ggaodu= z_new_wave -( z_new + z_wave )

    
    feigao = Ggaodu-gabove*math.cos(Theta_new*math.pi/180)
    
    Thetasuan = np.arctan(Vz_new/Vx_new) + Theta_new 
      


    cost_2 = 0

    if Thetasuan<-11.5:
        
        done=True
         
        next_state = statearg
         
        feigao = feigaoold
            
        cost_2 = -np.clip( (np.abs(Thetasuan)-11.5)/10 , 0 , 1 )
         
        cost_3,cost_4,cost_5,cost_6,cost_7,cost_8 = -1,-1,-1,-1,-1,-1
        
        cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 

        temp = np.append(np.zeros(0+1),[cost,0,flap,Thrust, feigao ])
        
        return next_state, cost, done, np.append(next_state,temp)

    if Thetasuan>5.5:
        
        done=True
         
        next_state = statearg
        
        feigao = feigaoold
                    
        cost_2 = -np.clip( (np.abs(Thetasuan)-5.5)/10 , 0 , 1 )
        
        cost_3,cost_4,cost_5,cost_6,cost_7,cost_8 = -1,-1,-1,-1,-1,-1
        
        cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 

        temp = np.append(np.zeros(0+1),[cost,0,flap,Thrust, feigao ])
        
        return next_state, cost, done, np.append(next_state,temp)



    cost_3 = -1
    
    # if dixing(x_new) <= ( z_new + z_wave):
    
    #     done=True
         
    #     next_state = statearg
        
    #     feigao = feigaoold
                    
    #     cost_3 = -1
        
    #     cost_4,cost_5,cost_6,cost_7,cost_8 = -1,-1,-1,-1,-1
        
    #     cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 

    #     temp = np.append(np.zeros(0+1),[cost,0,flap,Thrust, feigao ])
        
    #     return next_state, cost, done, np.append(next_state,temp)


    cost_4 = 0 
    
    
    zhongdez = - ( z_new + z_wave )
    
    
    qiantou_x = x_new + gabove*math.sin(Theta_new*math.pi/180) + lqian*math.cos(Theta_new*math.pi/180)
    
    qiantou_z = zhongdez-gabove*math.cos(Theta_new*math.pi/180)+lqian*math.sin(Theta_new*math.pi/180)
    
    houtou_x = x_new - (lhou*math.cos(Theta_new*math.pi/180) - gabove*math.sin(Theta_new*math.pi/180))
    
    houtou_z = zhongdez-gabove*math.cos(Theta_new*math.pi/180)-lhou*math.sin(Theta_new*math.pi/180)
    
    qian_wave_z = dixing(qiantou_x)
    
    hou_wave_z = dixing(houtou_x)
    
    
    #print('qiantou_z - qian_wave_z',qiantou_z - qian_wave_z)
    
    #print('houtou_z - hou_wave_z',houtou_z - hou_wave_z)

    
    
    if qiantou_z - qian_wave_z <= 0 :
        
        done=True
        
        next_state = statearg
        
        feigao = feigaoold
        
        #cost_4 = -np.clip( (0.1*0.4-(qiantou_z - qian_wave_z))/0.4, 0 , 1 )
        cost_4 = -np.clip( (0-(qiantou_z - qian_wave_z))/0.4, 0 , 1 )

    if  houtou_z - hou_wave_z  <=  0:
        
        done=True
        
        next_state = statearg
        
        feigao = feigaoold
        
        #cost_4 = -np.clip( (0.1*0.4-(houtou_z - hou_wave_z))/0.4, 0 , 1 )
        cost_4 = -np.clip( (0-(qiantou_z - qian_wave_z))/0.4, 0 , 1 )



    cost_5 =-1


    cost_6 = 0
     
    if Ggaodu >= hmax:
         
          # done=True
         
          # next_state = statearg
         
          # feigao = feigaoold
         
          #cost_6 = -np.clip( (Ggaodu-hmax)/0.4, 0 , 1 )
          cost_6 = -(Ggaodu-hmax)/0.4
          
          
          if Ggaodu >= (yinbianquan_shangxian+1)*0.4:
              
               done=True
             
               next_state = statearg
             
               feigao = feigaoold              
         
         
          
    elif Ggaodu <= hmin:
             
             done = True
             
             next_state = statearg
             
             feigao = feigaoold
             
             cost_6 =  -np.clip( (hmin-Ggaodu)/0.4, 0 , 1 )
             






    cost_7 = 0
     
    if Vx_new >= vmax:
         
         done=True
         
         next_state = statearg
         
         feigao = feigaoold
         
         cost_7 =  -np.clip( (Vx_new-vmax)/(vmax-vmin), 0 , 1 )
         

    if vmin >= Vx_new:
         
         done=True
         
         next_state = statearg
         
         feigao = feigaoold
         
         cost_7 = -np.clip( (vmin-Vx_new)/(vmax-vmin), 0 , 1 )
         


    cost_8 = 0
    
    if Ggaodu < 0.4*0.5:
        
    
        if done==False:
            
            if jixu(Vx_new,Thetasuan,Ggaodu,0,suduzu,thetazu,hzu,flapzu,ji)==False:
                
                done = True
                
                next_state = statearg
                
                feigao = feigaoold
                
                cost_8 = -1
        
        if done==True:
    
                cost_8 = -1    
            

          
     
    if feixingjilu.shape[0]==1:
        
        bushu = 0

    else:
        
        bushu = len(feixingjilu) - 1
    

        
    jieshoubu = allstep-2
    
    if x_new > (jieshoubu+2) * shijianbu * 20:
        
            done = True
            
            next_state = statearg
            
            feigao = feigaoold       
                
    if x_new < -1 * shijianbu * 5:
        
            done = True
            
            next_state = statearg
            
            feigao = feigaoold                 
        

    if bushu >  jieshoubu :
         
         done = True
         



    cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 
    
    temp = np.append(np.zeros(0+1),[cost,0,flap,Thrust, feigao ])
    
    return next_state, cost, done, np.append(next_state,temp)


  
def envstepforMPC(statearg,  shijianbu=shijianbu  ):

    Theta = statearg[0]
    Thetadot = statearg[1]
    u=statearg[2]
    udot=statearg[3]
    w=statearg[4]
    wdot=statearg[5]
    q=statearg[6]
    qdot=statearg[7]
    Vx=statearg[8]
    Vz=statearg[9]  ##朝下为正
    x=statearg[10]
    z=statearg[11]

    Fx=0
    Fz=0
    My=0
     
    Fy=0.  
    Mx=0.  
    Mz=0.
    Phi,Phidot=0.,0.   
    Psi,Psidot=0.,0.    
    v,vdot=0.,0.         
    p,pdot=0.,0.  
    r,rdot=0.,0.     
    Vy=0.   
    y=0.    
         
    outcome=runaero(Fx, Fy, Fz,Mx,My,Mz, \
             Phi,Theta,Psi,Phidot,Thetadot,Psidot, \
             u,v,w,udot,vdot,wdot, \
             p,q,r,pdot,qdot,rdot, \
             Vx,Vy,Vz,  \
             x,y,z,shijianbu,20,-20)
 
    Theta_new=outcome[1]
    Thetadot_new=outcome[4]
    u_new=outcome[6]
    udot_new=outcome[9]
    w_new=outcome[8]
    wdot_new=outcome[11]
    q_new=outcome[13]
    qdot_new=outcome[16]
    Vx_new=outcome[18]
    Vz_new=outcome[20]
    x_new=outcome[21]
    z_new=outcome[23]
    
    next_state=np.array([Theta_new,Thetadot_new,u_new,udot_new,w_new,wdot_new,q_new,qdot_new,Vx_new,Vz_new,x_new,z_new ])
  
    return next_state














matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'



fucknum=[0 for i in range(3000)]

def huatu(jilustates,  Tzhong,    flap_max,    initial,  shijianbu,xz, casenum_ori, episode_return, dixing,XX_n,YY_n, jiangji_part1, jiangji_part2, jiangji_part3, piancha_1, piancha_2, piancha_3,
                  yiduan__cr, erduan__cr, yiduan__spd, erduan__spd,yiduan__bnd, erduan__bnd,              
                  yiduan__cr_rd, erduan__cr_rd, yiduan__spd_rd, erduan__spd_rd,yiduan__bnd_rd, erduan__bnd_rd):
        
    casenum=casenum_ori
    
    xzuobiao = jilustates[:,10]
    
    zzuobiao = -jilustates[:,11]
    
    jiao = jilustates[:,0]
    
    global fucknum
    
    fucknum[casenum]+=1
    
    pathh = os.getcwd()
    
    initial_zifu = str( int(initial*10))
    
    end_zifu = str( int(end*10))
    
    wenjianjia = initial_zifu  + 'to' + end_zifu +'_case_' + str(casenum)
    
    if not os.path.exists(pathh+'/' + wenjianjia):
            
        os.mkdir(pathh+'/' + wenjianjia )
    
    mingcheng = pathh+'/'+ wenjianjia+ '/' + 'theone'+'.png'
    
    data_save =pathh+ '/'+ wenjianjia+ '/'+ str(fucknum[casenum])+'.npy'
    
    np.save(data_save,jilustates)
    
    
    pianchaandrd_data = np.array([jiangji_part1, jiangji_part2, jiangji_part3, 
                                  yiduan__cr_rd, erduan__cr_rd, yiduan__spd_rd, erduan__spd_rd,yiduan__bnd_rd, erduan__bnd_rd,
                                  piancha_1, piancha_2, piancha_3,
                                  yiduan__cr, erduan__cr, yiduan__spd, erduan__spd,yiduan__bnd, erduan__bnd])
    
    data_save =pathh+ '/'+ wenjianjia+ '/'+ 'pianchaandrd'+'.npy'
    
    np.save(data_save,pianchaandrd_data)
    
    
    
    fig=plt.figure(figsize=(18,13))
        
    atr=[]
    
    linestylezu=['solid','dotted','dashed','dashdot']  #4
    markerzu=["v","^","<",">","s","p","P","X","D","d"]  #10
    
    for a in range(len(linestylezu)):
        for b in range(len(markerzu)):
            atr.append([a,b])
        
    random.shuffle(atr)
        
    axes1 = fig.add_subplot(1,1,1)
    #axes2 = fig.add_subplot(2,1,2)
    
    #RR=0.1
    
    xhua, zhua =[] , [] 
    
    gabove = 0.31*0.4
    
    #wavexian =  LineString(xz)
     
    for a,b,c in zip(xzuobiao,zzuobiao,jiao):
         
        Theta_new = c
         
        Ggaodu = b
         
        Xgaodu = a
    
        xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
         
        zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
         
        xhua.append(xhua_n)
         
        zhua.append(zhua_n)
            

    axes1.plot( xhua , zhua ,label='path',linestyle='-',marker="s",alpha=0.8)

    axes1.plot(xz[:,0],xz[:,1]+0.4*end,linestyle='-')
    
    # axes1.plot(xz[:,0],xz[:,1],linestyle='-')
    
    # axes1.plot(xz[:,0],xz[:,1]+0.4,linestyle='-')
    
    
    axes1.plot(xz[:,0],xz[:,1]+0.4*yinbianquan_shangxian,linestyle='--')
    
    axes1.plot(xz[:,0],xz[:,1]+0.4*yinbianquan_xiaxian,linestyle='--')
    
    
    axes1.spines['right'].set_color('none')
    axes1.spines['top'].set_color('none')
    
    
    font1 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 18,
    }
    font2 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 23,
    }
    font3 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 25,
    }
    legend = plt.legend(prop=font2)#,ncol=3,loc = 'upper center',borderaxespad=0.1,borderpad=0.1,columnspacing=0.5,handlelength=1.5,handletextpad=0.4)  
    
    axes1.set_xlabel('x',font3)
    
    axes1.set_ylabel('z',font3,rotation= 0)
    
    axes1.set_xlim([0, x_shangxian])
    
    axes1.set_ylim([y_xiaxian,y_shangxian])  
       
    axes1.xaxis.labelpad = 0
    axes1.yaxis.labelpad = 30
    
    axes1.tick_params(labelsize=20)
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.725, bottom=0.075)
    
    jianglihe = jiangji_part1 + jiangji_part2 + jiangji_part3
    
    plt.title(str(round(episode_return,2)) +'_'+'d'+str(d)+'d'+str(round(jiangji_part1,2))+'_'  + str(round(jiangji_part1/jianglihe,2))    +  '_' +'s'+str(s)+'s'+str(round(jiangji_part2,2))+'_'  + str(round(jiangji_part2/jianglihe,2))    +  '_' +'j'+str(j)+'j'+str(round(jiangji_part3,2)) +  '_'   + str(round(jiangji_part3/jianglihe,2)) +
              
              
              '\n ' + 'yiduancrrd_'+str(round(yiduan__cr_rd,4)) +  '_' +'yiduanspdrd_'+str(round(yiduan__spd_rd,4))+'_' +'yiduanbndrd_'+str(round(yiduan__bnd_rd,4))+
              
              '\n ' + 'erduancrrd_'+str(round(erduan__cr_rd,4)) +  '_' +'erduanspdrd_'+str(round(erduan__spd_rd,4))+'_' +'erduanbndrd_'+str(round(erduan__bnd_rd,4))+
              
              
              '\n ' + 'd'+str(d)+'d'+str(round(piancha_1,4)) +  '_' +'s'+str(s)+'s'+str(round(piancha_2,4))+'_' +'j'+str(j)+'j'+str(round(piancha_3,4))+
              
              '\n ' + 'yiduancr_'+str(round(yiduan__cr,4)) +  '_' +'yiduanspd_'+str(round(yiduan__spd,4))+'_' +'yiduanbnd_'+str(round(yiduan__bnd,4))+
              
              '\n ' + 'erduancr_'+str(round(erduan__cr,4)) +  '_' +'erduanspd_'+str(round(erduan__spd,4))+'_' +'erduanbnd_'+str(round(erduan__bnd,4))
              
              
              , fontsize = 40)



    plt.savefig(mingcheng,dpi=300)
    
    plt.clf() # 清图。
    plt.cla() # 清坐标轴。
    plt.close() # 关窗口
    






fuckquannum=[0 for i in range(3000)]

def quanhuatu(jilustates,  Tzhong,    flap_max,    initial,    shijianbu, xz,casenum_ori, episode_return, dixing,XX_n,YY_n, jiangji_part1, jiangji_part2, jiangji_part3, piancha_1, piancha_2, piancha_3,
                  yiduan__cr, erduan__cr, yiduan__spd, erduan__spd,yiduan__bnd, erduan__bnd,              
                  yiduan__cr_rd, erduan__cr_rd, yiduan__spd_rd, erduan__spd_rd,yiduan__bnd_rd, erduan__bnd_rd):
    
    casenum=casenum_ori
    
    #0  = kaolvbuchangshu
    
    #print('jilustates',jilustates)
    
    xzuobiao = jilustates[:,10]
    
    zzuobiao = -jilustates[:,11]
    
    jiao = jilustates[:,0]
    
    global fuckquannum
    
    fuckquannum[casenum] +=1
    
    pathh = os.getcwd()
    
    initial_zifu = str( int(initial*10))
    
    end_zifu = str( int(end*10))
    
    wenjianjia = 'quanhua'+ initial_zifu + 'to' + end_zifu +'_case_' + str(casenum)
    
    if not os.path.exists(pathh+'/' + wenjianjia):
            
        os.mkdir(pathh+'/' + wenjianjia )
    
    mingcheng = pathh+'/'+ wenjianjia+ '/' + 'quanhua_theone'+'.png'
    
    data_save =pathh+ '/'+ wenjianjia+ '/'+ str(fuckquannum[casenum])+'.npy'
    
    np.save(data_save,jilustates)
    
    
    
    pianchaandrd_data = np.array([jiangji_part1, jiangji_part2, jiangji_part3, 
                                  yiduan__cr_rd, erduan__cr_rd, yiduan__spd_rd, erduan__spd_rd,yiduan__bnd_rd, erduan__bnd_rd,
                                  piancha_1, piancha_2, piancha_3,
                                  yiduan__cr, erduan__cr, yiduan__spd, erduan__spd,yiduan__bnd, erduan__bnd])
    
    data_save =pathh+ '/'+ wenjianjia+ '/'+ 'pianchaandrd'+'.npy'
    
    np.save(data_save,pianchaandrd_data)
    
    
    
    # data_save =pathh+ '/'+ wenjianjia+ '/'+ str(fuckquannum[casenum])+'.npy'
    
    # np.save(data_save,jilustates)
    
    
    
    fig=plt.figure(figsize=(18,13))
        
    atr=[]
    
    linestylezu=['solid','dotted','dashed','dashdot']  #4
    markerzu=["v","^","<",">","s","p","P","X","D","d"]  #10
    
    for a in range(len(linestylezu)):
        for b in range(len(markerzu)):
            atr.append([a,b])
        
    random.shuffle(atr)
        
    axes1 = fig.add_subplot(1,1,1)
    #axes2 = fig.add_subplot(2,1,2)
    
    #RR=0.1
    
    xhua, zhua =[] , [] 
   
    #wavexian =  LineString(xz)
    
    for a,b,c in zip(xzuobiao,zzuobiao,jiao):
         
        Theta_new = c
         
        Ggaodu = b
         
        Xgaodu = a
    
        xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
         
        zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
         
        xhua.append(xhua_n)
         
        zhua.append(zhua_n)


    axes1.plot( xhua , zhua ,label='path',linestyle='-',marker="s",alpha=0.8)
    

    axes1.plot(xz[:,0],xz[:,1]+0.4*end,linestyle='-')
    
    # axes1.plot(xz[:,0],xz[:,1],linestyle='-')
    
    # axes1.plot(xz[:,0],xz[:,1]+0.4,linestyle='-')
    

    axes1.plot(xz[:,0],xz[:,1]+0.4*yinbianquan_shangxian,linestyle='--')
    axes1.plot(xz[:,0],xz[:,1]+0.4*yinbianquan_xiaxian,linestyle='--')
        


    axes1.spines['right'].set_color('none')
    axes1.spines['top'].set_color('none')
    
    
    font1 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 18,
    }
    font2 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 23,
    }
    font3 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 25,
    }
    legend = plt.legend(prop=font2)#,ncol=3,loc = 'upper center',borderaxespad=0.1,borderpad=0.1,columnspacing=0.5,handlelength=1.5,handletextpad=0.4)  
    
    axes1.set_xlabel('x',font3)
    
    axes1.set_ylabel('z',font3,rotation= 0)
    
    axes1.set_xlim([0, x_shangxian])
    
    axes1.set_ylim([y_xiaxian,y_shangxian])   
    
    axes1.xaxis.labelpad = 0
    
    axes1.yaxis.labelpad = 30
    
    axes1.tick_params(labelsize=20)
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.725, bottom=0.075)
    
    jianglihe = jiangji_part1 + jiangji_part2 + jiangji_part3
    
    
    plt.title(str(round(episode_return,2)) +'_'+'d'+str(d)+'d'+str(round(jiangji_part1,2))+'_'  + str(round(jiangji_part1/jianglihe,2))    +  '_' +'s'+str(s)+'s'+str(round(jiangji_part2,2))+'_'  + str(round(jiangji_part2/jianglihe,2))    +  '_' +'j'+str(j)+'j'+str(round(jiangji_part3,2)) +  '_'   + str(round(jiangji_part3/jianglihe,2)) +
              
              
              '\n ' + 'yiduancrrd_'+str(round(yiduan__cr_rd,4)) +  '_' +'yiduanspdrd_'+str(round(yiduan__spd_rd,4))+'_' +'yiduanbndrd_'+str(round(yiduan__bnd_rd,4))+
              
              '\n ' + 'erduancrrd_'+str(round(erduan__cr_rd,4)) +  '_' +'erduanspdrd_'+str(round(erduan__spd_rd,4))+'_' +'erduanbndrd_'+str(round(erduan__bnd_rd,4))+
              
              
              '\n ' + 'd'+str(d)+'d'+str(round(piancha_1,4)) +  '_' +'s'+str(s)+'s'+str(round(piancha_2,4))+'_' +'j'+str(j)+'j'+str(round(piancha_3,4))+
              
              '\n ' + 'yiduancr_'+str(round(yiduan__cr,4)) +  '_' +'yiduanspd_'+str(round(yiduan__spd,4))+'_' +'yiduanbnd_'+str(round(yiduan__bnd,4))+
              
              '\n ' + 'erduancr_'+str(round(erduan__cr,4)) +  '_' +'erduanspd_'+str(round(erduan__spd,4))+'_' +'erduanbnd_'+str(round(erduan__bnd,4))
              
              
              , fontsize = 40)


    
    plt.savefig(mingcheng,dpi=300)
    
    
    plt.clf() # 清图。
    plt.cla() # 清坐标轴。
    plt.close() # 关窗口
    


def demo3_custom_env_rl (    Tzhong,    flap_max,    initial,   shijianbu  ):
    
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    #args.random_seed = 1943
    
    args.bubu = 0
    


    '''choose an DRL algorithm'''
    #from elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.if_use_gae = True

    args.allstep_num = jieshoubu + 2

    
    args.gamma =  1-1/(jieshoubu+2)
    args.env = Fuckingfly(state_0, Tzhong,    flap_max,    initial,   shijianbu)
    args.env_eval = Fuckingfly(state_0, Tzhong,    flap_max,    initial,   shijianbu)  
    
    
    
    
    args.reward_scale = 2 ** 0 
    
    args.if_remove = False

    args.if_allow_break = False
    
    #args.max_memo = 2 ** 21
    #args.batch_size = 2 ** 9
    #args.repeat_times = 2 ** 1
    args.eval_gap = 0  # for Recorder
    # args.eval_times1 = 2 ** 1  # for Recorder
    # args.eval_times2 = 2 ** 3  # for Recorder

    # train_and_evaluate(args)
    #args.rollout_num = 4
    args.eval_times1 = 2 **0
    args.eval_times1 = 2 **0
    #args.if_per = True

    args.num_threads = 1

    train_and_evaluate(args,initial,end)









    

if __name__ == '__main__':

    demo3_custom_env_rl (  Tzhong,    flap_max,    initial,   shijianbu )


    # pathh = os.getcwd()
    
    # initial_zifu = str( int(initial*10))
    
    # end_zifu = str( int(end*10))
    
    # for i in range(2):
        
    #     casenum = i + 1
        
    #     wenjianjia =  initial_zifu + 'to' + end_zifu +'_case_' + str(casenum)   
        
    #     mingzi = wenjianjia+'/*.npy'
        
    #     loadming_new = max(glob.glob(mingzi), key=os.path.getmtime)
        
    #     shutil.copy(loadming_new, 'case' + str(casenum)   + '.npy') 
        
    #     shutil.copy(wenjianjia + '/theone.png', 'case' + str(casenum)   + '.png') 
        
        
        







    # pathh = os.getcwd()
    
    # initial_zifu = str( int(initial*10))
    
    # end_zifu = str( int(end*10))
    
    # for i in range(2):
        
    #     casenum = i + 1
    
    #     wenjianjia_quanhua = 'quanhua'+ initial_zifu + 'to' + end_zifu +'_case_' + str(casenum)
        
    #     wenjianjia =  initial_zifu + 'to' + end_zifu +'_case_' + str(casenum)      
        
    #     # yasuo(wenjianjia_quanhua)
        
    #     # yasuo(wenjianjia)
        
    #     with zipfile.ZipFile(wenjianjia_quanhua +'.zip','w') as target:
    #         for i in os.walk(wenjianjia_quanhua):
    #             for n in i[2]:
    #                 target.write(''.join((i[0],'/',n)))         
        

    #     with zipfile.ZipFile(wenjianjia +'.zip','w') as target:
    #         for i in os.walk(wenjianjia):
    #             for n in i[2]:
    #                 target.write(''.join((i[0],'/',n)))         
        
        
        
    #     shutil.rmtree(wenjianjia_quanhua)
        
    #     shutil.rmtree(wenjianjia)
        
  
# #压缩文件夹
# import zipfile,os

# with zipfile.ZipFile('yolo3_pytorch_master.zip','w') as target:
#     for i in os.walk('yolo3_pytorch_master'):
#         for n in i[2]:
#             target.write(''.join((i[0],'/',n))) 

        
