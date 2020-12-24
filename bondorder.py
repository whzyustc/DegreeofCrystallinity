

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:24:27 2019

@author: whzyz
"""


import cv2
import numpy as np
import math

path='./'
filename='0015.jpg'
gray=cv2.imread(path+filename,0)
#img=gray[0:500,0:500]
#cv2.imwrite(path+'cut'+filename, img)
gray= cv2.GaussianBlur(gray,(15,15),1)

#cv2.imwrite(path+filename+'x', gray[:50][:50])
cv2.imwrite(path+'gray'+filename, gray)
print ('\n %s \n' % (gray))


tmpm = np.matrix(gray)

def gamma_trans(img, gamma):
    # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    
    # 实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img, gamma_table)

m=gamma_trans(tmpm,2)

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
        if f[x1][y1]<f[x][y]+2 and af[x1][y1]==0 :
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
    if atomnum[i]<20 or atomsum[i]<270:
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
#def renderpoint (x,y,radius):
distance=36

def eio (x,y):
    ux=x/(x*x+y*y)**0.5
    uy=y/(x*x+y*y)**0.5
    theta=complex(ux, uy)
    #print (x,y,ux,uy,theta**6)
    return (theta**6)

def modofeio(a,numberofpoint):
    return ((a.real)**2+(a.imag)**2)**0.5/numberofpoint

def searchnear (x,y,radius): 
    numberofpoint=0
    mon=complex(0,0)
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
                    if i*i+j*j<distance:
                        continue
                    mon+=eio(tx-x,ty-y)
#                    print (tx-x,ty-y)
                    numberofpoint=numberofpoint+1
                    
                    if numberofpoint==6:
#                        print (mon,numberofpoint,x,y)
                        return modofeio(mon,numberofpoint)
#                else:
#                    atomp[tx][ty]+=140   
        j=r
        for i in range (1,r+1):
            rotation=[[x+i,y+j],[x+j,y-i],[x-i,y-j],[x-j,y+i]]
            for k in range (4):
                tx=rotation[k][0]
                ty=rotation[k][1]
                if tx<0 or tx>a-1 or ty<0 or ty>b-1:
                    continue
                if (atomp[tx][ty]==255):
                    if i*i+j*j<distance:
                        continue
                    mon+=eio(tx-x,ty-y)
#                    print (tx-x,ty-y)
                    numberofpoint=numberofpoint+1
                    if numberofpoint==6:
#                        print (mon,numberofpoint,x,y)
                        return modofeio(mon,numberofpoint)
#                else:
#                    atomp[tx][ty]+=140
    if numberofpoint!=0:
        return modofeio(mon,numberofpoint)
    return 0
            
            
minbri=10000
maxbri=0

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
    atomp[x][y]=bri
    for i in range(r):
        for j in range(r):
            if (i*i+j*j>distance):
                continue
            renderpixel(x+i,y+j,bri)
            renderpixel(x-i,y+j,bri)
            renderpixel(x-i,y-j,bri)
            renderpixel(x+i,y-j,bri)
            
    
for i in range (num):
    if axisx[i]==0 and axisy[i]==0:
        continue
#    searchnear(axisx[i],axisy[i],40)
    renderpoint(axisx[i],axisy[i],searchnear(axisx[i],axisy[i],50)*255,64)
    
#    cv2.imwrite(path+'atomp'+filename, atomp)
#    break

    
cv2.imwrite(path+'atomp'+filename, atomp)
    
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
    
# '''
# atomlin = cv2.GaussianBlur(atomp,(15,15),1)
    
    
        
    


