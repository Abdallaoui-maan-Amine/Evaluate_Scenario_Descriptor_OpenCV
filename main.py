# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 13:04:25 2021

@author: amine


##### Auteur : ABDALLAOUI MAAN Amine
##### Master : T.S.I
##### Prcours : I.M.O.V.I

"""
# %% Library
import pykitti
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Visualiser les données sous forme d'un tableau organisé
from prettytable import PrettyTable

# Permet d'utiliser le temps
import time

# tqdm permet d'afficher la barre de progression d'une boucle for
from tqdm import tqdm 

# Visualiser les donnees sous forme DataFrame
import pandas as pd

# Creation de plalette de couleur
import seaborn as sns

print("[+] Numpy version",np.__version__)
print("[+] OpenCV version :",cv2.__version__)

# [+] Numpy version 1.19.5
# [+] OpenCV version : 4.5.3

# Definir une liste de couleur
color=sns.color_palette(None,16)


# %% Import RAW demo

basedir = r'C:\Users\amine\Master TSI\Mise en œuvre de traitements avancés des images\TP2\KITTI_SAMPLE\RAW'
date = '2011_09_26'
drive = '0009'

data = pykitti.raw(basedir, date, drive, frames=range(0, 50, 1))

I = np.array(data.get_cam2(0))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%% visualize_Data_using_PrettyTable function %%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def visualize_Data_using_PrettyTable(data, Descriptor):
    y = PrettyTable()
    y.field_names = ["Méthode", "Temps d’exécution", "nombre de points/image", "Taux d'appariement correcte"]
    for d in range(len(Descriptor)):
        y.add_row([Descriptor[d], str(round(np.mean(data[d,:,0]),4))+' sec' , str(int(np.mean(data[d,:,1])))+' points', str(round(np.mean(data[d,:,2]),4))+' %'])
        
    return y




# %% 
def visualize_Data_using_DataFrame(data, Descriptor, scenario_nbr):
    df=pd.DataFrame(columns=["Méthode", "Temps d’exécution", "nombre de points/image", "Taux d'appariement correcte"])

    for d in range(len(Descriptor)):    
        df.at[d,'Méthode']=Descriptor[d]
        df.at[d,"Temps d’exécution"]=str(round(np.mean(data[d,:,0]),4))+' sec'
        df.at[d,"nombre de points/image"]=str(int(np.mean(data[d,:,1])))+' points'
        df.at[d,"Taux d'appariement correcte"]=str(round(np.mean(data[d,:,2]),4))+' %'


    df.to_csv("Scénario"+str(scenario_nbr)+"_Result.csv", index=False, encoding = 'utf-8-sig')

# %% %%%%%%%%%%%%%%%%%%%%% Detector function  %%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def detector_Method(detector, descriptor, Matching, I, I_in):
    
    if detector == 'SIFT' or detector == 'AKAZE' or detector == 'KAZE' or detector == 'ORB' or detector == 'BRISK':
        method = eval('cv2.'+detector+'_create()')
        
    if descriptor == 'SIFT' or descriptor == 'AKAZE' or descriptor == 'KAZE' or descriptor == 'ORB' or descriptor == 'BRISK':
        method_des = eval('cv2.'+descriptor+'_create()')
        #print(descriptor)
    
    if detector == 'STAR':
        method = cv2.xfeatures2d.StarDetector_create()
        
    if descriptor == 'BRIEF':
        method_des = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        
    if detector == 'FAST':
        method = cv2.FastFeatureDetector_create()
        
    kp1 = method.detect(I,None)
    kp2 = method.detect(I_in,None)
    des1 = method_des.compute(I,kp1)[-1]
    des2 = method_des.compute(I_in,kp2)[-1]
    
    if Matching == 'NORM_L1' or Matching == 'L1':
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)   
        
    elif Matching == 'NORM_L2' or Matching == 'L2':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 
        
    elif Matching == 'HAMMING' :
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    
    X = np.float32([kp1[m.queryIdx].pt for m in matches])
    Y = np.float32([kp2[m.trainIdx].pt for m in matches])
        
    return X,Y
        


## Sift avec flann
# def SIFT(I,I_in,Matching):
#     method = cv2.KAZE_create()
#     kp1, des1 = method.detectAndCompute(I,None)           
#     kp2, des2 = method.detectAndCompute(I_in,None)
    
#     # FLANN parameters
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#     search_params = dict(checks=100)   # or pass empty dictionary
    
#     flann = cv2.FlannBasedMatcher(index_params,search_params)
    
#     matches = np.array(flann.knnMatch(des1,des2,k=2))
    
#     X = np.float32([kp1[m.queryIdx].pt for m in matches[:,0]])
#     Y = np.float32([kp2[m.trainIdx].pt for m in matches[:,1]])  
#     return X,Y




# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Scenario 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def set_intensity(I):
    # Liste des images de sortie contenant 8 images avec intensité differente.
    Img8=[]
    
    # Liste contenant les valeurs des intensités
    Intensity_C=[]
    
    # Changement de type d'images vers float pour effectuer les calcul
    I_np= np.array(np.int16(I))
    
    # Parcourir les canstantes
    for i in np.linspace(-30, 30, 4):
        img=I_np+i
        Intensity_C.append(i)
        
        # Mettre les intensités en dehors de la plage [0,255] dedans
        img[img<0],img[img>255]=0,255
        
        # Remettre l'images en type uint8
        Img8.append(cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR))
        
    # Parcourir les multiplicateur
    for i in np.linspace(0.7, 1.3, 4):
        img=I_np*i
        Intensity_C.append(i)
        
        # Mettre les intensités en dehors de la plage [0,255] dedans
        img[img<0],img[img>255]=0,255
        
        # Remettre l'images en type uint8
        Img8.append(cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR))
    return Img8,Intensity_C
    
   
# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluate Scenario 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def evaluate_scenario_1(X,Y):
    
    # Nombre de points appariés correctement 
    pt_In=0
    
    # Parcourir les points d'interet des deux images en comparant les x et y des deux images
    for i in range(min(len(X),len(Y))):
        if(abs(X[i][1]-Y[i][1]) < 1 and  abs(X[i][0]-Y[i][0]) < 1):
            pt_In += 1
            
    # Calcul des du pourcentage des points appariés correctement
    prc = 100*(pt_In/len(X))

    return prc



# %% Resultat de l'evaluation du scenario 1
I_si,Intensity_C=set_intensity(I)
Descriptor=['SIFT_L1', 'SIFT_L2' ,'AKAZE_L2','BRISK_L2','ORB_L2','ORB_HAMMING','KAZE_L1','STAR_SIFT_L1','STAR_SIFT_L2' , 'FAST_SIFT_L1' ,'STAR_BRIEF_L2' ,'STAR_BRIEF_HAMMING','STAR_BRISK_L1','STAR_BRISK_HAMMING']

# Liste contenent les pourcentage de chaque methode et pour chaque image
prc=[]

# Parcourir les 8 images avec intensité differente
for i in range(len(I_si)):
    
    # Parcourir les differents méthodes declaré dans la liste Descriptor
    for j in tqdm(range(len(Descriptor))):
        method=Descriptor[j].split("_")
        
        # Si la méthode de detecteur est la meme celle du descripteur
        if len(method) == 2:
            X,Y = detector_Method(method[0], method[0], method[1], I, I_si[i])
          
        # Si la méthode de detecteur est differente de celle du descripteur
        elif len(method) == 3:
            X,Y = detector_Method(method[0], method[1], method[2], I, I_si[i])
        
        prc.append(evaluate_scenario_1(X,Y))
            
# Changement de dimension de la liste prc afin de rassembler les pourcentage des methodes de chaque image.
prc=np.array(prc).reshape(-1,len(Descriptor)).T


# Affichage dans la console la methode avec la liste des pourcentage trouvée.
for i in range(len(Descriptor)):
    print("\n\n[+]",Descriptor[i]," : ",prc[i])


# %% Courbe Scénario1


# Affichage des courbes pour la constante
plt.figure()      
for i in range(len(Descriptor)):
    plt.plot(Intensity_C[0:4],(prc[i])[0:4], c=color[i])
    plt.scatter(Intensity_C[0:4],(prc[i])[0:4])
plt.title('Scénario 1')
plt.xlabel('La constante')
plt.ylabel('Taux de points appariés correctement')
plt.legend(Descriptor, bbox_to_anchor=(1, 1))

    
# Affichage des courbes pour les multiplicateur de changement d'intensité
plt.figure()      
for i in range(len(Descriptor)):
    plt.plot(Intensity_C[4:],(prc[i])[4:], c=color[i])
    plt.scatter(Intensity_C[4:],(prc[i])[4:])
plt.title('Scénario 1')
plt.xlabel('multiplicateur de changement d''intensité')
plt.ylabel('Taux de points appariés correctement')
plt.legend(Descriptor,bbox_to_anchor=(1, 1))




# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Scenario 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Changement d'échelle
def change_scale(I):
    
    # Liste qui va englober les images après changement d'échelle.
    Is=[]
    
    # Liste qui va englober les valeurs d'échelle.
    scale=[]
    
    # Extraire les dimensions de l'image.
    h,w,_ = I.shape
    
    # Boucle pour parcourir les valeurs de changements d'échelle.
    for s in np.linspace(1.1, 2.3, 7):
        
        # Construire la matrice de trasformation y compris la matrice de changements d'échelle
        # Centre # rotation # scale
        M = cv2.getRotationMatrix2D((w//2,h//2), 0, s)
        
        # Application de changement d'échelle
        img = cv2.warpAffine(I, M, (w,h))
        
        # Insérer l'image sortie dans la liste Is 
        Is.append(img)
        
        # Insérer les valeurs de changement d'échelles dans la liste scale
        scale.append(s)
    return Is,scale


# %%

def evaluate_scenario_2(X,Y,I,s):
    
    # Extraire la matrice de transformation avec 0 comme rotation et s comme valeur de changement d'échelle
    M = cv2.getRotationMatrix2D((I.shape[1]//2,I.shape[0]//2), 0, s)
    
    # Application du changement d'échelles sur les points de l'image gauche.
    X[:,0]=(X[:,0]*s)+M[0,2]
    X[:,1]=(X[:,1]*s)+M[1,2]

    pt_In = 0
    for i in range(len(X)):
        
        # Verification des cordoonnees des points X apres transformation avec ceux de l'image droite Y
        if(abs(X[i][1]-Y[i][1]) < 1 and abs(X[i][0]-Y[i][0]) < 1):
            pt_In += 1
            
    # Calcul du pourcentage des points correctement appariés
    prc=100*(pt_In/len(X))
    return prc


 


# %% Resultat de l'evaluation du scenario 2
Is,scale = change_scale(I)
Descriptor=['SIFT_L1', 'SIFT_L2' ,'AKAZE_L2','BRISK_L2','ORB_L2','ORB_HAMMING','KAZE_L1','STAR_SIFT_L1','STAR_SIFT_L2' , 'FAST_SIFT_L1' ,'STAR_BRIEF_L2' ,'STAR_BRIEF_HAMMING','STAR_BRISK_L1','STAR_BRISK_HAMMING']

# Liste contenent les pourcentage de chaque methode et pour chaque image
prc=[]

# Parcourir les 7 images avec valeur d'échelle differentes
for i in range(len(Is)):
    
    # Parcourir les differents méthodes declarée dans la liste Descriptor
    for j in tqdm(range(len(Descriptor))):
        method=Descriptor[j].split("_")
        
        # Si la méthode de detecteur est la même de celle du descripteur
        if len(method) == 2:
            X,Y = detector_Method(method[0], method[0], method[1], I, Is[i])
            
        # Si la méthode de detecteur est differente de celle du descripteur
        elif len(method) == 3:
            X,Y = detector_Method(method[0], method[1], method[2], I, Is[i])
        
        prc.append(evaluate_scenario_2(X,Y, I,scale[i]))
            
# Changement de dimension de la liste prc afin de rassembler les pourcentage des methodes de chaque image.
prc=np.array(prc).reshape(-1,len(Descriptor)).T


# Affichage dans la console la methode avec la liste des pourcentage trouvée.
for i in range(len(Descriptor)):
    print("\n\n[+]",Descriptor[i]," : ",prc[i])
    

# %% Figure Scenario 2
plt.figure()      
for i in range(len(Descriptor)):
    plt.plot(scale,prc[i], c=color[i])
    plt.scatter(scale,prc[i])
plt.title('Scénario 2')
plt.xlabel('Scale')
plt.ylabel('Taux de points appariés correctement')
plt.legend(Descriptor,bbox_to_anchor=(1, 1))


# %%




# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Scenario 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Changement de rotation

def change_rot(I):
    Ir=[]
    degree=[]
    h,w,_ = I.shape
    for rot_deg in np.linspace(10, 90, 9):
        # Centre # rotation # scale
        M = cv2.getRotationMatrix2D((w//2,h//2), rot_deg, 1)
        img = cv2.warpAffine(I, M, (w,h))
        Ir.append(img)
        degree.append(rot_deg)
    return Ir,degree


# %%

def evaluate_scenario_3(X,Y,I,r):
    
    M = cv2.getRotationMatrix2D((I.shape[1]//2,I.shape[0]//2), r, 1)
    angle = np.radians(r)
    M_ = np.array([[np.cos(angle), -(np.sin(angle))],[np.sin(angle), np.cos(angle)]])
    X=X@M_
    X[:,0]=X[:,0]+M[0,2]
    X[:,1]=X[:,1]+M[1,2]
    
    pt_In = 0
    for i in range(len(X)):
        if(abs(X[i][1]-Y[i][1]) < 1 and abs(X[i][0]-Y[i][0]) < 1):
            pt_In += 1
    prc=100*(pt_In/len(X))
    return prc


# %% Resultat de l'evaluation du scenario 3
Ir,degree = change_rot(I)
Descriptor=['SIFT_L1', 'SIFT_L2' ,'AKAZE_L2','BRISK_L2','ORB_L2','ORB_HAMMING','KAZE_L1','STAR_SIFT_L1','STAR_SIFT_L2' , 'FAST_SIFT_L1' ,'STAR_BRIEF_L2' ,'STAR_BRIEF_HAMMING','STAR_BRISK_L1','STAR_BRISK_HAMMING']

# Liste contenent les pourcentage de chaque methode et pour chaque image
prc=[]

# Parcourir les images avec les degrées de rotation
for i in range(len(Ir)):
    
    # Parcourir les differents méthodes declaré dans la liste Descriptor
    for j in tqdm(range(len(Descriptor))):
        method=Descriptor[j].split("_")
        
        # Si la méthode de detecteur est la même celle du descripteur
        if len(method) == 2:
            X,Y = detector_Method(method[0], method[0], method[1], I, Ir[i])
            
        # Si la méthode de detecteur est differente de celle du descripteur
        elif len(method) == 3:
            X,Y = detector_Method(method[0], method[1], method[2], I, Ir[i])
        
        prc.append(evaluate_scenario_3(X,Y, I, degree[i]))
            
# Changement de dimension de la liste prc afin de rassembler les pourcentage des methodes de chaque image.
prc=np.array(prc).reshape(-1,len(Descriptor)).T


# Affichage dans la console la methode avec la liste des pourcentage trouvée.
for i in range(len(Descriptor)):
    print("\n\n[+]",Descriptor[i]," : ",prc[i])
    


# %% Figure Scenario 3

plt.figure()      
for i in range(len(Descriptor)):
    plt.plot(degree,prc[i], c=color[i])
    plt.scatter(degree,prc[i])
plt.title('Scénario 3')
plt.xlabel('Degree de rotation')
plt.ylabel('Taux de points appariés correctement')
plt.legend(Descriptor,bbox_to_anchor=(1, 1))





# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Scénario4__get_stereo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_stereo(idx):
    cam2_image = np.array(data.get_cam2(idx))
    cam3_image = np.array(data.get_cam3(idx))
    return cam2_image,cam3_image
    

# %% evaluate_Scenario_4
def evaluate_Scenario_4(X,Y):
    F, mask =cv2.findFundamentalMat(X,Y,method=cv2.FM_RANSAC+cv2.FM_8POINT)
    pt_In = (mask == 1).sum()
    prc=100*(pt_In/len(mask))
    return prc

# %% evaluate_Scenario_4__Calcul
Descriptor=['SIFT_L1', 'SIFT_L2' ,'AKAZE_L2','BRISK_L2','ORB_L2','ORB_HAMMING','KAZE_L1','STAR_SIFT_L1','STAR_SIFT_L2' , 'FAST_SIFT_L1' ,'STAR_BRIEF_L2' ,'STAR_BRIEF_HAMMING','STAR_BRISK_L1','STAR_BRISK_HAMMING']

tab=[]
number_Image=50

for i in tqdm(range(len(Descriptor))):
    for k in range(number_Image):
        img,img_ = get_stereo(k)
    
        method=Descriptor[i].split("_")
        
        start=time.time()
        # Si la méthode de detecteur est la même celle du descripteur
        if len(method) == 2:
            start=time.time()
            X,Y = detector_Method(method[0], method[0], method[1], img, img_)
            
        # Si la méthode de detecteur est differente de celle du descripteur
        elif len(method) == 3:
            start=time.time()
            #X,Y = eval(method[0] + "_" + method[1] + "(img, img_, method[2])")
            X,Y = detector_Method(method[0], method[1], method[2], img, img_)
        
        prc=evaluate_Scenario_4(X,Y)
        tab.append([time.time()-start, len(X), prc])
        

# %%
tab=np.array(tab).reshape(len(Descriptor),number_Image,3)

y = visualize_Data_using_PrettyTable(tab, Descriptor)


# %% evaluate_Scenario_4__DataFrame

visualize_Data_using_DataFrame(tab, Descriptor, 4)

# %%


# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Scénario5__get_consecutive_image %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_consecutive_image(idx1,idx2):
    cam2_image = np.array(data.get_cam2(idx1))
    cam2_image_ = np.array(data.get_cam2(idx2))
    return cam2_image, cam2_image_
    

# %% evaluate_Scenario_5

def evaluate_Scenario_5(X,Y):
    F, mask =cv2.findFundamentalMat(X,Y,method=cv2.FM_RANSAC+cv2.FM_8POINT)
    pt_In = (mask == 1).sum()
    prc=100*(pt_In/len(mask))
    return prc



# %% evaluate_Scenario_5__Calcul
Descriptor=['SIFT_L1', 'SIFT_L2' ,'AKAZE_L2','BRISK_L2','ORB_L2','ORB_HAMMING','KAZE_L1','STAR_SIFT_L1','STAR_SIFT_L2' , 'FAST_SIFT_L1' ,'STAR_BRIEF_L2' ,'STAR_BRIEF_HAMMING','STAR_BRISK_L1','STAR_BRISK_HAMMING']

tab=[]
number_Image=50-1

for i in range(len(Descriptor)):
    for k in tqdm(range(number_Image)):
        img,img_ = get_consecutive_image(k,k+1)
    
        method=Descriptor[i].split("_")
        
        start=time.time()
        # Si la méthode de detecteur est la même celle du descripteur
        if len(method) == 2:
            start=time.time()
            X,Y = detector_Method(method[0], method[0], method[1], img, img_)
            
        # Si la méthode de detecteur est differente de celle du descripteur
        elif len(method) == 3:
            start=time.time()
            X,Y = detector_Method(method[0], method[1], method[2], img, img_)
        
        prc=evaluate_Scenario_5(X,Y)
        tab.append([time.time()-start, len(X), prc])



# %% evaluate_Scenario_5__PrettyTable
tab=np.array(tab).reshape(len(Descriptor),number_Image,3)

y = visualize_Data_using_PrettyTable(tab, Descriptor)


# %% evaluate_Scenario_5__DataFrame
visualize_Data_using_DataFrame(tab, Descriptor, 5)



# %%


# %% %%%%%%%%%%%%%%%%%%%%% Sénario6__get_Consecutive1_Image_With2Camera %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_Consecutive1_Image_With2Camera(idx):
    
    # Obtenir l'image idx de la cam2
    cam2_image = np.array(data.get_cam2(idx))
    
    # Obtenir l'image idx+1 de la cam3
    cam3_image = np.array(data.get_cam3(idx+1))
    return cam2_image, cam3_image
    

# %% evaluate_Scenario_6
def evaluate_Scenario_6(X,Y):
    
    # Calcul de la fonction fondamental 
    F, mask =cv2.findFundamentalMat(X,Y,method=cv2.FM_RANSAC+cv2.FM_8POINT)
    
    # Calcul de nombre de points ou le mask egal à 1
    pt_In = (mask == 1).sum()
    prc=100*(pt_In/len(mask))
    return prc


# %% evaluate_Scenario_6__Calcul
Descriptor=['SIFT_L1', 'SIFT_L2' ,'AKAZE_L2','BRISK_L2','ORB_L2','ORB_HAMMING','KAZE_L1','STAR_SIFT_L1','STAR_SIFT_L2' , 'FAST_SIFT_L1' ,'STAR_BRIEF_L2' ,'STAR_BRIEF_HAMMING','STAR_BRISK_L1','STAR_BRISK_HAMMING']

tab=[]
number_Image=50-1

for i in tqdm(range(len(Descriptor))):
    for k in range(number_Image):
        img,img_ = get_Consecutive1_Image_With2Camera(k)
    
        method=Descriptor[i].split("_")
        
        start=time.time()
        # Si la méthode de detecteur est la même celle du descripteur
        if len(method) == 2:
            start=time.time()
            X,Y = detector_Method(method[0], method[0], method[1], img, img_)
            
        # Si la méthode de detecteur est differente de celle du descripteur
        elif len(method) == 3:
            start=time.time()
            X,Y = detector_Method(method[0], method[1], method[2], img, img_)
        
        prc=evaluate_Scenario_6(X,Y)
        tab.append([time.time()-start, len(X), prc])


# %% evaluate_Scenario_6__PrettyTable
tab=np.array(tab).reshape(len(Descriptor),number_Image,3)

y = visualize_Data_using_PrettyTable(tab, Descriptor)

# %% evaluate_Scenario_6__DataFrame

visualize_Data_using_DataFrame(tab, Descriptor, 6)

# %%



# %% %%%%%%%%%%%%%%%%%%%%% Scénario7__get_Consecutive2_Image_With2Camera %%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_Consecutive2_Image_With2Camera(idx):
    cam2_image = np.array(data.get_cam2(idx))
    cam3_image = np.array(data.get_cam3(idx+2))
    return cam2_image, cam3_image
    

# %% evaluate_Scenario_7

def evaluate_Scenario_7(X,Y):
    F, mask =cv2.findFundamentalMat(X,Y,method=cv2.FM_RANSAC+cv2.FM_8POINT)
    pt_In = (mask == 1).sum()
    prc=100*(pt_In/len(mask))
    return prc

# %% evaluate_Scenario_7__Calcul
Descriptor=['SIFT_L1', 'SIFT_L2' ,'AKAZE_L2','BRISK_L2','ORB_L2','ORB_HAMMING','KAZE_L1','STAR_SIFT_L1','STAR_SIFT_L2' , 'FAST_SIFT_L1' ,'STAR_BRIEF_L2' ,'STAR_BRIEF_HAMMING','STAR_BRISK_L1','STAR_BRISK_HAMMING']

tab=[]
number_Image=50-2

for i in tqdm(range(len(Descriptor))):
    for k in range(number_Image):
        img,img_ = get_Consecutive2_Image_With2Camera(k)
    
        method=Descriptor[i].split("_")
        
        start=time.time()
        # Si la méthode de detecteur est ka même celle du descripteur
        if len(method) == 2:
            start=time.time()
            X,Y = detector_Method(method[0], method[0], method[1], img, img_)
            
        # Si la méthode de detecteur est differente de celle du descripteur
        elif len(method) == 3:
            start=time.time()
            X,Y = detector_Method(method[0], method[1], method[2], img, img_)
        
        prc=evaluate_Scenario_7(X,Y)
        tab.append([time.time()-start, len(X), prc])



# %% evaluate_Scenario_7__PrettyTable
tab=np.array(tab).reshape(len(Descriptor),number_Image,3)

y = visualize_Data_using_PrettyTable(tab, Descriptor)


# %% evaluate_Scenario_7__DataFrame

visualize_Data_using_DataFrame(tab, Descriptor, 7)
