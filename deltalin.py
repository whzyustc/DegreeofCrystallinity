# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:35:31 2019

@author: whzyz
"""


import cv2
import numpy as np
import math

path='./'
filename='0015.jpg'
gray=cv2.imread(path+filename,0)

gray= cv2.GaussianBlur(gray,(15,15),1)

#cv2.imwrite(path+filename+'x', gray[:50][:50])
cv2.imwrite(path+'gray'+filename, gray)
print ('\n %s \n' % (gray))


m = np.matrix(gray)
def GuessMaxx(h,sizeX,sizeY,sigma1,sigma2):
    a = int(h.shape[0])
    b = int(h.shape[1])
    midx=(sizeX+1)/2
    midy=(sizeY+1)/2
    print (midx,midy)
    core = np.zeros([sizeX+1,sizeY+1])
    f = np.zeros((a,b))
    print (math.pi)
    sum = 0
    for i in range (sizeX):
        for j in range (sizeY):
            core[i][j]=math.e**(-((midx-j+1)**2/sigma2**2+(midy-i+1)**2/sigma1**2))/(2*math.pi*sigma1*sigma2)
            sum+=core[i][j]
    
    print (core)
    average=sum/sizeX/sizeY
    for i in range (sizeX):
        for j in range (sizeY):
            core[i][j]-=average*0
    
    print (core)
    print (sum,average)
    f = cv2.filter2D(h,-1,core)*2
    print (f) 
    maxdata=0
    cv2.imwrite(path+'guess1'+filename, f)
    
    for i in range (a):
        for j in range (b):
            if f[i][j]>maxdata:
                maxdata=f[i][j]
    print (maxdata)
    formf = np.zeros((a,b))
    for i in range (a):
        for j in range (b):
            formf[i][j]=int (f[i][j]/maxdata*255)
            if formf[i][j]<30:
                formf[i][j]=0
                
    cv2.imwrite(path+'guess2'+filename, formf)    
    return f   

def MartixMaxx(h,sizeX,sizeY):
    a = int(h.shape[0])
    b = int(h.shape[1])
    print (a,b)
    f=h
    midx=int ((sizeX+1)/2)
    midy=int ((sizeY+1)/2)
    print (sizeX,sizeY)
    print (midx,midy)
    print (math.pi)
    
    """maxdata=0
    for i in range (a):
        for j in range (b):
            if h[i][j]>maxdata:
                maxdata=h[i][j]
                
    print (maxdata)
    for i in range (a):
        for j in range (b):
            h[i][j]=int (h[i][j]/maxdata*255)
       """         
    f = np.zeros((a,b))
    
    core = np.zeros((sizeY+1,sizeY+1))
    
    for i in range (sizeY):
        for j in range (sizeY):
            core[i][j]=-1/(sizeY**2-sizeX**2)
    
    
    sizeYX= int ((sizeY-sizeX)/2)
    sizeYX2=int ((sizeX+sizeY)/2)
    print (sizeYX,sizeYX2,'\n')
    for i in range (sizeYX, sizeYX2):
        for j in range (sizeYX, sizeYX2):
            core[i][j]=1/sizeX**2
    
    print (core)
    
    f = cv2.filter2D(h,-1,core)                      
    
    cv2.imwrite(path+'xx'+filename, f)
    
    print (sum)
    return f


f=MartixMaxx(m,7,11)

a = int(f.shape[0])
b = int(f.shape[1])
def peakcheck(x,y):
    around=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
    for i in range (8):
        x1=x+around[i][0]
        if x1<0 or x1>a-1:
            continue
        y1=y+around[i][1]
        if y1<0 or y1>b-1:
            continue
        if f[x1][y1]>f[x][y]:
            return 0
        
    return 1
    
def searchaera(x,y,num):
    if f[x][y]==0:
        return
    af[x][y]=num
    around=[[-1,0],[0,-1],[0,1],[1,0]]
    for i in range (4):
        x1=x+around[i][0]
        if x1<0 or x1>a-1:
            return 
        y1=y+around[i][1]
        if y1<0 or y1>b-1:
            return
        if f[x1][y1]<f[x][y]+3 and af[x1][y1]==0 :
            searchaera(x1,y1,num)
    
def searchall():
    
    global number
    number=1
    global af 
    af = np.zeros((a,b))
    for i in range (a):
        for j in range (b):
            if f[i][j]==0:
                continue
            if peakcheck(i,j)==1 and af[i][j]==0:
                #print (i,j,number)
                searchaera(i,j,number)
                number=number+1
    return number
    
                
            
num=searchall()
print (num)
atomx = np.zeros(num)
atomy = np.zeros(num)
atomsum = np.zeros(num)
atomnum = np.zeros(num)
axisx = np.zeros(num,dtype='i')
axisy = np.zeros(num,dtype='i')
#print (af)

for i in range (a):
    for j in range (b):
        if af[i][j]==0:
            continue
        k=int (af[i][j])
        atomx[k]=int (atomx[k]+f[i][j]*i)
        atomy[k]=int (atomy[k]+f[i][j]*j)
        atomnum[k]=int (atomnum[k]+1)
        atomsum[k]=int (atomsum[k]+f[i][j])
        
atomposition = np.zeros((a,b)) 
atomp=np.zeros((a,b))
around=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]     
for i in range (1,num):
    if atomnum[i]<10 or atomsum[i]<600:
        continue 
    axisx[i]=int (atomx[i]/atomsum[i])
    axisy[i]=int (atomy[i]/atomsum[i])
    #brightness=atomsum[i]/atomnum[i]
    brightness=255
    atomp[axisx[i]][axisy[i]]=brightness
    for j in range (9):
        x1=axisx[i]+around[j][0]
        if x1<0 or x1>a-1:
            continue
        y1=axisy[i]+around[j][1]
        if y1<0 or y1>b-1:
            continue
        atomposition[x1][y1]=brightness
        f[x1][y1]=brightness

cv2.imwrite(path+'xx'+filename, f)
cv2.imwrite(path+'atompositionk'+filename, atomposition)


a = int(atomp.shape[0])
b = int(atomp.shape[1])
def renderpixel (x,y,bri):
    if x<0 or x>a-1:
        return
    if y<0 or y>b-1:
        return 
    if atomp[x][y]!=0:
        return 
    atomp[x][y]=bri
def renderpoint (x,y,bri,distance):
    r=int((distance **0.5))
    for i in range(r):
        for j in range(r):
            if (i*i+j*j>distance):
                continue
            renderpixel(x+i,y+j,bri)
            renderpixel(x-i,y+j,bri)
            renderpixel(x-i,y-j,bri)
            renderpixel(x+i,y-j,bri)
            
                
    
def searchnear (x,y,radius): 
    distance=0
    numberofpoint=0
    deltar=0
    #rotation=[[a+x,b+y],[b+x,y-a],[x-a,y-b],[x-b,y+a]]
    for r in range(radius):
        i=r
        for j in range(r):
            rotation=[[x+i,y+j],[x+j,y-i],[x-i,y-j],[x-j,y+i]]
            for k in range (4):
                tx=rotation[k][0]
                ty=rotation[k][1]
                if tx<0 or tx>a-1 or ty<0 or ty>b-1:
                    continue
                if (atomp[tx][ty]==255):
                    if distance==0:
                        distance=(i*i+j*j)
                        #print (i,j,distance)
                    if (i*i+j*j)>(distance*1.4):
                        #print ('out of range')
                        renderpoint (x,y,deltar/numberofpoint,distance)
                        return
                    atomstax[numberofpoint]=tx
                    atomstay[numberofpoint]=ty
                    numberofpoint=numberofpoint+1
                    deltar+=i*i+j*j
                    if (numberofpoint>8):
                        renderpoint (x,y,deltar/numberofpoint,distance)
                        return 
        j=r
        for i in range (1,r+1):
            rotation=[[x+i,y+j],[x+j,y-i],[x-i,y-j],[x-j,y+i]]
            for k in range (4):
                tx=rotation[k][0]
                ty=rotation[k][1]
                if tx<0 or tx>a-1 or ty<0 or ty>b-1:
                    continue
                if (atomp[tx][ty]==255):
                    if distance==0:
                        distance=(i*i+j*j)
                        #print (i,j,distance)
                    if (i*i+j*j)>(distance*1.4):
                        #print ('out of range')
                        renderpoint (x,y,deltar/numberofpoint,distance)
                        return
                    atomstax[numberofpoint]=tx
                    atomstay[numberofpoint]=ty
                    numberofpoint=numberofpoint+1
                    deltar+=i*i+j*j
                    if (numberofpoint>8):
                        renderpoint (x,y,deltar/numberofpoint,distance)
                        return 
            
            
minbri=10000
maxbri=0


for i in range (1,num):
    if axisx[i]==0 and axisy[i]==0:
        continue
    atomstax = np.zeros(10,dtype='i')
    atomstay = np.zeros(10,dtype='i')
    searchnear(axisx[i],axisy[i],100)
'''    print (numberofpoint)
    deltar=0
    tx=ty=0
    if numberofpoint==0:
        continue
    for j in range(numberofpoint):
        tx=atomstax[j]-axisx[i]
        ty=atomstay[j]-axisy[i]
        print (tx,ty)
        deltar=deltar+(tx*tx+ty*ty)
    deltar=deltar/numberofpoint
    print (deltar)
    atomp[axisx[i]][axisy[i]]=deltar
    if deltar>maxbri:
        maxbri=deltar
        #print (axisx[i],axisy[i],maxbri)
    if deltar<minbri:
        minbri=deltar

brirange=255/(maxbri-minbri)
print ('maxbri ',maxbri,',minbri' ,minbri)
for i in range (1,num):
    if axisx[i]==0 and axisy[i]==0:
        continue
    #atomp[axisx[i]][axisy[i]]=atomp[axisx[i]][axisy[i]]-minbri
    
'''
atomlin = cv2.GaussianBlur(atomp,(51,51),10)
maxe=0
for i in range (a):
    for j in range (b):
        if atomlin[i][j]>maxe:
            maxe=atomlin[i][j]

for i in range (a):
    for j in range (b):
        atomlin[i][j]=atomlin[i][j]/maxe*255
        if atomlin[i][j]==0:
            atomlin[i][j]=255
            

            
cv2.imwrite(path+'lin'+filename, atomlin)
    
    
        
    


