# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 00:14:21 2021
based on....
Analyze5_DistancePairs_v2
@author: pc
"""

import cv2
import pylab as pl 
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from itertools import combinations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from scipy.spatial import Delaunay
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
    
import sys
# import ClearMap
# import ClearMap.Settings as settings
# import ClearMap.IO as io

from pathlib import Path
import pandas as pd

# =============================================================================
# 
# =============================================================================

def getData(BaseDirectory, filename):  
    myData = None
    myPath = os.path.join(BaseDirectory, filename)
    if  os.path.exists(myPath):
        myData = np.load(myPath)
        return myData
    else:
        print ('ATTENTION: Path not found. \nPlease, check the path and file name')
        return None
    
#From [-1..0..1]  float64 to [0...128...255] uint8  
def convertTo255(v): 
    lim_inf = v.min()   
    lim_sup = v.max()      
    new_sup = 255
    new_inf = 0     
    v = np.round(((v-lim_inf)/float(lim_sup-lim_inf))*(new_sup-new_inf) + new_inf)
    v = np.uint8(v)
    return v 

def plot_tetra(tetra, pts, color="green", alpha=0.1, lc="k", lw=1, verbose=False):
    if verbose==True:
        print()
        print('Start: plot_tetra()')
    
    # Get Triangles from a Tetraedron (i.e. 4 collection of 3 points)
    combs = combinations(tetra, 3)
    # print('tetra: \n', tetra)
    # print('combs: \n', list(combs))
    for comb in combs:
        X = pts[comb, 0]
        Y = pts[comb, 1]
        Z = pts[comb, 2]        
        verts = [list(zip(X, Y, Z))]
        # print('XYZ', X, Y, Z)
        # print('verts: \n', verts)
        
        triangle = Poly3DCollection(verts, facecolors=color, alpha=0.1)
        lines = Line3DCollection(verts, colors=lc, linewidths=lw)
        ax.add_collection3d(triangle)
        ax.add_collection3d(lines)
    
    if verbose==True:
        print('Stop: plot_tetra()')
    
def icosahedron():
    h = 0.5*(1+np.sqrt(5))
    p1 = np.array([[0, 1, h], [0, 1, -h], [0, -1, h], [0, -1, -h]])
    p2 = p1[:, [1, 2, 0]]
    p3 = p1[:, [2, 0, 1]]
    return np.vstack((p1, p2, p3))


def cube():
    points = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ])
    return points  
    
#==============================================================================
# MAIN
#==============================================================================
if __name__=='__main__':
    
    
    #==============================================================================
    # Load Data    
    #==============================================================================
    # ResolLabel = 'RES(620x905x1708)'
    # BaseDirectory = 'D:/000_Test/3_ClearMap/' + ResolLabel +  '/0_WholeBrain_bk/' 
    
    # filename = 'coordXYZ'  + '.npy'
    # p0 = getData(BaseDirectory, filename)
    
    # filename = 'coordXYZ_Filtered'  + '.npy'
    # p1 = getData(BaseDirectory, filename)

    # I_scale = 100.0
    # filename = 'Iraw_Idog_Iopen_Vol'  + '.npy'
    # v0 = getData(BaseDirectory, filename)
    # v0 = v0[:,[0, 2, 1, 3]]
    # v0[:,0] = v0[:,0]/I_scale
    # v0[:,1] = v0[:,1]/I_scale
    # v0[:,2] = v0[:,2]/I_scale
    
    # filename = 'Iraw_Idog_Iopen_Vol_Filtered'  + '.npy'
    # v1 = getData(BaseDirectory, filename)
    # v1 = v1[:,[0, 2, 1, 3]]
    # v1[:,0] = v1[:,0]/I_scale
    # v1[:,1] = v1[:,1]/I_scale
    # v1[:,2] = v1[:,2]/I_scale
    
    # filename = 'myVolThreshold.npy'
    # Vol_Threshold = getData(BaseDirectory, filename)
    # minVol, maxVol= Vol_Threshold[0], Vol_Threshold[1]
    
    # filename =  'myMinIntensity.npy'
    # minInt = getData(BaseDirectory, filename)

    
    #==============================================================================
    #   1) Select the DataSet to compute the <Delaunay Triangularization>
    #   The input data is 3-dimensional.
    #   Dimension of the Matrix: nx3 
    #       Rows: n (number of three-dimensional points)
    #       Colu: x y z
    #==============================================================================
    boolTest = False
    pts = None
    
    #A) Testing DataSet: (use this set to understand the algorithm)    
    pts0 = np.asarray([[3,6,0],[6,8,0],[8,2,0],[5,5,5]]) 
    pts1 = np.asarray([[3,6,0],[6,8,0],[8,2,0],[10,8,0],[4,3,2]]) 
    pts2 = np.asarray([[3,6,0],[6,8,0],[8,2,0],[10,8,0],[4,3,2],
                      [7,12,0],[15,10,0],[10,4,0],
                      [10,6,2],[7,4,2],[4,10,2],
                      [6,10,-2],[3,16,-2],[7,2,-2]
                      ])  
    pts3 = np.asarray([[1, 1, 1],[-1, -1, 1],[-1, 1, -1],[1, -1, -1],
                       [-1, 1, 1],[1, -1, 1],[1, 1, -1],[-1, -1, -1] ])                    
    pts4 = np.asarray([[1, 0, -1/np.sqrt(2)],[.1, 0, -1/np.sqrt(2)],
                        [0, 1, 1/np.sqrt(2)],[0, -1, 1/np.sqrt(2)],
                        [0, 0, 0]
                          ]) 
    pts5 = np.asarray([[1, 1, 1],[-1, -1, 1],[-1, 1, -1],[1, -1, -1],
                       [-1, 1, 1],[1, -1, 1],[1, 1, -1],[-1, -1, -1] ])    
    l = 1.0
    hxy = np.sqrt(3.)/2.0*l
    hz  = np.sqrt(6.)/3.0*l
    d   = np.sqrt(3.)/6.0*l
    pts5 = np.asarray ([[0,0,0],[l,0,0],[0,hxy,0],[l/2,d,hz]])
    # pts5 = np.asarray ([[0,0,0],[l,0,0],[0,hxy,0],[l/2,d,hz], [1,1,1]])
    
    v1 = ( np.sqrt(8./9), 0, -1./3 )
    v2 = ( -np.sqrt(2./9), np.sqrt(2./3), -1./3 )
    v3 = ( -np.sqrt(2./9), -np.sqrt(2./3), -1./3 )
    v4 = ( 0, 0, 1. )
    v5 = (1,1,0)
    pts6 = np.asarray([v1,v2,v3,v4,v5]) 
# =============================================================================
#     
# =============================================================================
    # tetraedro
    l = 10
    h = (np.sqrt(3)/2)*l
    H = (np.sqrt(6)/3)*l
    c = (1/3)*h
    pts7 = np.asarray([[0,0,0],[l,0,0],[l/2,h,0],[l/2,c,H]]) 
    # pts7 = np.asarray([[0,0,0],[l,0,0],[l/2,h,0],[l/2,c,H],
    #                    [2*l,0,0],[l+l/2,h,0],[l+l/2,c,H],[l,h-c,H]]) 
    pts = pts7
    
   
# =============================================================================
#     
# =============================================================================
    # pts = pts5
    # pts = pts5
    # pts = cube()
    # pts = icosahedron()
    # pts_norm = pts/pts.max()  
    # pts_norm = pts_norm - pts.min() 
    # pts_norm = (np.round(100*pts_norm)).astype(np.uint)
    # pts = pts_norm
# =============================================================================
#     
# =============================================================================
      
    # Read Table 
    # pathFolder = Path(r'C:\Users\aarias\MySpyderProjects\p6_Cell_v9\Results\SUB')
    # pathFolder = Path(r'C:\Users\aarias\MySpyderProjects\p6_Cell_v9\Results\CA1')
    # pathFolder = Path(r'C:\Users\aarias\MySpyderProjects\p6_Cell_v9\Results\CA1a')
    # pathFolder = Path(r'C:\Users\aarias\MySpyderProjects\p6_Cell_v9\Results\DGif')
    # fileName = '2b_Cells.csv'
    pathFolder = Path(r'C:\Users\aarias\MySpyderProjects\p6_Cell_v15\Results\mainTest\Detections')
    fileName = '1c_MultiScale_Detections_Filter.csv'
    
    pathFile = os.path.join(pathFolder, fileName) 
    df = pd.read_csv(pathFile)
    pts = df[['X_abs_out_px', 'Y_abs_out_px', 'Z_abs_out_px']] #px
    pts = df[['X_abs_um', 'Y_abs_um', 'Z_abs_um']]
    pts = np.array(pts)
    
# =============================================================================
# 
# =============================================================================
    
    # pts[:,0] = pts[:,0] - pts[:,0].mean()
    # pts[:,1] = pts[:,1] - pts[:,1].mean()
    # pts[:,2] = pts[:,2] - pts[:,2].mean()
    
    
# =============================================================================
#   
# =============================================================================




#     #B) Custom DataSet:
#     if boolTest==False:     
#         p1 = p1.astype(int) 
#         pts = p1[0:200]
#         pts = p1[0:500]
# #        pts = p1
    
    #==============================================================================
    #    2) Compute: Delaunay Triangulation of 3D DatSet
    #==============================================================================
    tri = Delaunay(pts)
    T = tri.simplices
    
    # =============================================================================
    #     
    # =============================================================================
    tetra = pts[tri.simplices]
    # pts[tri.neighbors]
    
    isInside = (tri.neighbors>=0)
    isInside = np.all(isInside , axis=1)
    
    tetraInside = tetra[isInside]

    #==============================================================================
    #   3) Visualize: Delaunay Triangularization  
    #==============================================================================
    boolVis = True    
    if  boolVis== True:   
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for k, tetra in enumerate(tri.simplices):
            # color = plt.cm.Accent(k/(tri.nsimplex - 1))
            color = plt.cm.Accent(k/(tri.nsimplex))
            # print()
            # print('k:', k)
            # print('pts', pts)
            # print('tetra', tetra)
            plot_tetra(tetra, pts, color=color, alpha=0.1, lw=0.5, lc="k")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='k')
        # pts_borders = pts[list([0, 1, 2, 4, 6, 7, 11, 12, 17, 18, 19, 20, 21, 23, 30, 31, 32, 35, 36, 38, 40, 42, 43, 45, 47, 49])]
        # ax.scatter(pts_borders[:, 0], pts_borders[:, 1], pts_borders[:, 2], c='r')
        
        plt.show()
    
    # sys.exit()
    # %matplotlib qt
    # %matplotlib inline
    # ==============================================================================
    #   4) Compute:  The Angles of a each Vertex in the Triangularization
    #   
    #   Two types of angles:
    #       a) The <Simplex Angles> of all Triangles that convert into a Vertex
    #       b) The <Solid Angle> of all Tethraedron that convert into a Vertex 
    #   
    #   Note1: The dimensionless unit of a solid angle is the steradian,
    #   with 4π steradians in a full sphere (720 degrees) 
    #    
    #   Note2: The <Dihedral Angles> of the tethraedron are the angles between  
    #   two intersecting planes (not computed)
    #==============================================================================
    
    start_time = time.time()
    #4.0) Initialize Variables    
    V_angle_tri  = []   #store the <Simplex Angles> of all Triangles
    V_angle_tet  = []   #store the <Solid Angle> of all Tethraedrons
    V_border_ix  = []   #store the index of the points that are at the borders
    V_polygon    = []   #store the number Triangles    that converge into a vertex (i.e. gives the sides of the polygon    -> Polygon    Classification)
    V_polyhedron = []   #store the number Tethraedrons that converge into a vertex (i.e. gives the faces of the polyhedron -> Polyhedron Classification)
    
    #4.0) Initialized parameters 
    #Number of vertex of the Delaunay Triangularization:
    #N=3 for 2D DataSets since Delanauy algorithm returns Triangles that have 3 pts
    #N=4 for 3D DataSets since Delanauy algorithm returns Tethraedrons that have 4 points
    N = 4   

    #4.1) Flatten the Vector which contains the indexes of 3 pts that define each triangle
    t = T.flatten()
    for i in range(0, pts.shape[0]):    
            
        #4.2) Take the index of a point. Index: (x,y)
        pt_ref_ix = i       
        
        #4.3) Get all triangles around that point:
        ix = np.where(t==pt_ref_ix)[0]//N
        T_near = T[ix]
        
        #4.4) Get the vectors of each triangule around pt_ref
        V = pts[pt_ref_ix] - pts[T_near[T_near!=pt_ref_ix]]
        n = V.shape[0]                  #number of vectors connected to pts_ref
        V_ix = np.arange(0, n, N-1)     #index to select V1, V2, V3 for each Tethraedron that shares the same vertex (p_ref)
        V1, V2, V3 = V[V_ix], V[V_ix + 1], V[V_ix + 2]
        
        #4.5) Calculate: Simplex Angles of vectors that share the same vertex (i.e. p_ref)
        #for all the triangles that are connected to that vertex
        #a) V1,V2
        s = np.sum(V1*V2,axis=1)
        m = np.sqrt(np.sum(V1**2,axis=1)) * np.sqrt(np.sum(V2**2,axis=1))
#        s, m = np.abs(s), np.abs(m)
        # theta1 = np.arccos(s/m)*(180./np.pi) 
        theta1 = np.rad2deg(np.arccos(np.clip(s/m, -1.0, 1.0)))
        
        # print(theta1)
        # v1v2 = np.sum(V1*V2,axis=1)
        # theta1 = np.rad2deg(np.arccos(np.clip(v1v2, -1.0, 1.0)))
        # print(theta1)
        # sys.exit()
        
        #b) V1,V3
        s = np.sum(V1*V3,axis=1)
        m = np.sqrt(np.sum(V1**2,axis=1)) * np.sqrt(np.sum(V3**2,axis=1))
#        s, m = np.abs(s), np.abs(m)
        # theta2 = np.arccos(s/m)*(180./np.pi)
        
        v1v3 = np.sum(V1*V3,axis=1)
        # theta2 = np.rad2deg(np.arccos(np.clip(v1v3, -1.0, 1.0)))
        theta2 = np.rad2deg(np.arccos(np.clip(s/m, -1.0, 1.0)))
        
        #c) V2,V3
        s = np.sum(V2*V3,axis=1)
        m = np.sqrt(np.sum(V2**2,axis=1)) * np.sqrt(np.sum(V3**2,axis=1))
#        s, m = np.abs(s), np.abs(m)
        # theta3 = np.arccos(s/m)*(180./np.pi)

        v2v3 = np.sum(V2*V3,axis=1)
        # theta3 = np.rad2deg(np.arccos(np.clip(v2v3, -1.0, 1.0)))   
        theta3 = np.rad2deg(np.arccos(np.clip(s/m, -1.0, 1.0)))
        
        #d) Concatenate all angles 
        theta_tri = np.concatenate([theta1,theta2,theta3])
     
        #4.6) Cualculate: Solid Angle of the three vectors that form a Tethraedron
        s = np.sum(V1*np.cross(V2, V3))
        s = np.sum(V1*V2*V3,axis=1)
        m = (
            np.sqrt(np.sum(V1**2,axis=1))*np.sqrt(np.sum(V2**2,axis=1))*np.sqrt(np.sum(V3**2,axis=1)) + 
            np.sum(V1*V2,axis=1)*np.sqrt(np.sum(V3**2,axis=1)) +
            np.sum(V1*V3,axis=1)*np.sqrt(np.sum(V2**2,axis=1)) +
            np.sum(V2*V3,axis=1)*np.sqrt(np.sum(V1**2,axis=1))
            )
#        s, m = np.abs(s), np.abs(m)
        theta_tet = 2*np.arctan(s/m)
        # theta_tet = np.abs(theta_tet)
        
        # theta_tet = 2*np.arctan(np.clip(s/m, -1.0, 1.0))
        

               
        #4.7) Get the index of the pts of the border:
        #ATTENTION: 3D (theta_tri<900deg)
#        if theta_tet.sum()<(np.pi/4.0)*(180./np.pi)**2 -10: #41252
#            V_border_ix.append(i)
        if theta_tet.sum()<(np.pi/4.0)-0.0001*(np.pi/4.0): #41252
            V_border_ix.append(i) 
        
        V_polygon.append(theta_tri.shape[0])
        for j in theta_tri:            
            V_angle_tri.append(j)
            
        V_polyhedron.append(theta_tet.shape[0])        
        for k in theta_tet: 
            V_angle_tet.append(k)
            
#        print''
#        print'------------'
#        print'Simplex Angle: \n', theta_tri
#        print''
#        print'Simplex Angle Sum: \n', theta_tri.sum()

#        print''
#        print'Solid Angle: \n', theta_tet
#        print''
#        print'Solid Angle Sum: \n', theta_tet.sum()
#        if i==1:
#            break
    
    #4.8) Store in a numpy array (list to numpy)        
    V_angle_tri = np.asarray(V_angle_tri)
    V_polygon = np.asarray(V_polygon)
    
    V_angle_tet = np.asarray(V_angle_tet)
    V_angle_tet = np.rad2deg(V_angle_tet) #????
    V_polyhedron = np.asarray(V_polyhedron)
    
    
    dt = time.time() - start_time
    print('')
    print('Time')
    print('dt=',dt) 
    print(len(V_border_ix))
    
    # print()
    # print('V_angle_tri:\n', V_angle_tri)
    # print('V_angle_tet:\n', V_angle_tet)    
    # print('V_polyhedron:\n', V_polyhedron)
    # print('V_border_ix:\n', V_border_ix)
    # sys.exit()
    
    #==============================================================================
    #   #Plot 1: Simplex Angle Distribution
    #==============================================================================
    fig, axs = plt.subplots(1,1) 
    #Calculate the Distribution of the variable "Angle"
    p = V_angle_tri
    p = np.round(p, 2)
    n = 30
    print('Histogram: Simplex')
    print('n=', n)
    hist, bins = np.histogram(p, bins=n, density=False)
    width = 0.7 * (bins[1] - bins[0])
    if width<1:
        width=2
    center = (bins[:-1] + bins[1:]) / 2
    norm = hist/float(hist.max())

    ax = axs
    ax.bar(center, hist, align='center', width=width)
    ax.set_xlim([np.min([p.min(), 0]), np.max([p.max(), 180])])
#    ax.vlines(x=p.min(), ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle=':', color='g', linewidth=2.0, label='Theta_min')           
#    ax.vlines(x=1.0/f_G_max, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle=':', color='r', linewidth=2.0, label='Ts_Fourier')           

    mystr = ('Histogram: Simplex Angle')
    ax.set_title(mystr)
    ax.set_xlabel('Simplex Angle [deg]')
    ax.set_ylabel('Count')
    
    mystr = (
    'min : ' + "{0:.2f}".format(p.min()) + '\n' + 
    'avg : ' + "{0:.2f}".format(p.mean()) + '\n' +
    'med: ' + "{0:.2f}".format(np.median(p)) + '\n' +
    'max: ' + "{0:.2f}".format(p.max()) + '\n' + '\n' +
    
    'std  : ' + "{0:.2f}".format(p.std()) + '\n' +
    'stdN: ' + "{0:.2f}".format(p.std()/p.mean())    
    )              
    # ax.text(0.60, 0.95, mystr,
    #          fontsize=12,
    #          horizontalalignment='left',
    #          verticalalignment='top', 
    #          transform = ax.transAxes
    #          ) 
    ax.text(1.05, 1.01, mystr,
             fontsize=12,
             horizontalalignment='left',
             verticalalignment='top', 
             transform = ax.transAxes
             ) 
    

    plt.plot()
    
    #==============================================================================
    #   #Plot 2: Solid Angle Distribution
    #==============================================================================
    fig, axs = plt.subplots(1,1) 
    #Calculate the Distribution of the variable "Angle"
    p = V_angle_tet
    n = 30
    print('n=', n)
    hist, bins = np.histogram(p, bins=n, density=False)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    norm = hist/float(hist.max())

    ax = axs
    ax.bar(center, hist, align='center', width=width)
#    ax.set_xlim([0,180])
#    ax.vlines(x=p.min(), ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle=':', color='g', linewidth=2.0, label='Theta_min')           
#    ax.vlines(x=1.0/f_G_max, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle=':', color='r', linewidth=2.0, label='Ts_Fourier')           

    mystr = ('Histogram: Solid Angle [Sr]')
    ax.set_title(mystr)
    ax.set_xlabel('Solid Angle [deg]')
    ax.set_ylabel('Count') 
    
    mystr = (
    'min : ' + "{0:.2f}".format(p.min()) + '\n' + 
    'avg : ' + "{0:.2f}".format(p.mean()) + '\n' +
    'med: ' + "{0:.2f}".format(np.median(p)) + '\n' +
    'max: ' + "{0:.2f}".format(p.max()) + '\n' + '\n' +
    
    'std  : ' + "{0:.2f}".format(p.std()) + '\n' +
    'stdN: ' + "{0:.2f}".format(p.std()/p.mean())    
    )              
    ax.text(1.05, 1.01, mystr,
             fontsize=12,
             horizontalalignment='left',
             verticalalignment='top', 
             transform = ax.transAxes
             ) 

    plt.plot()    

    #==============================================================================
    #   #Plot: Polygon
    #==============================================================================
    fig, axs = plt.subplots(1,1) 
    #Calculate the Distribution of the variable "Angle"
    p = V_polygon
    n = 30
    print('n=', n)
    hist, bins = np.histogram(p, bins=n, density=False)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    norm = hist/float(hist.max())

    ax = axs
    ax.bar(center, hist, align='center', width=width)
    ax.set_xlim([np.min([p.min(), 0]), 2*np.max([p.max(), 0])])
#    ax.set_xlim([0,180])
#    ax.vlines(x=p.min(), ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle=':', color='g', linewidth=2.0, label='Theta_min')           
#    ax.vlines(x=1.0/f_G_max, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle=':', color='r', linewidth=2.0, label='Ts_Fourier')           

    mystr = ('Histogram: Polygon')
    ax.set_title(mystr)
    ax.set_xlabel('number of Polygon surrounding a Vertex [edges]')
    ax.set_ylabel('Count')    
    
    mystr = (
    'min : ' + "{0:.2f}".format(p.min()) + '\n' + 
    'avg : ' + "{0:.2f}".format(p.mean()) + '\n' +
    'med: ' + "{0:.2f}".format(np.median(p)) + '\n' +
    'max: ' + "{0:.2f}".format(p.max()) + '\n' + '\n' +
    
    'std  : ' + "{0:.2f}".format(p.std()) + '\n' +
    'stdN: ' + "{0:.2f}".format(p.std()/p.mean())    
    )              
    ax.text(1.05, 1.01, mystr,
             fontsize=12,
             horizontalalignment='left',
             verticalalignment='top', 
             transform = ax.transAxes
             )   

    plt.plot() 
    

    #==============================================================================
    #   #Plot: polyhedron
    #==============================================================================
    fig, axs = plt.subplots(1,1) 
    #Calculate the Distribution of the variable "Angle"
    p = V_polyhedron
    n = 30
    print('n=', n)
    hist, bins = np.histogram(p, bins=n, density=False)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    norm = hist/float(hist.max())

    ax = axs
    ax.bar(center, hist, align='center', width=width)
    ax.set_xlim([np.min([p.min(), 0]), 2*np.max([p.max(), 0])])
#    ax.set_xlim([0,180])
#    ax.vlines(x=p.min(), ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle=':', color='g', linewidth=2.0, label='Theta_min')           
#    ax.vlines(x=1.0/f_G_max, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle=':', color='r', linewidth=2.0, label='Ts_Fourier')           

    mystr = ('Histogram: Polyhedron')
    ax.set_title(mystr)
    ax.set_xlabel('number of Polyhedron surrounding a Vertex')
    ax.set_ylabel('Count')
    
    
    mystr = (
    'min : ' + "{0:.2f}".format(p.min()) + '\n' + 
    'avg : ' + "{0:.2f}".format(p.mean()) + '\n' +
    'med: ' + "{0:.2f}".format(np.median(p)) + '\n' +
    'max: ' + "{0:.2f}".format(p.max()) + '\n' + '\n' +
    
    'std  : ' + "{0:.2f}".format(p.std()) + '\n' +
    'stdN: ' + "{0:.2f}".format(p.std()/p.mean())    
    )              
    ax.text(1.05, 1.01, mystr,
             fontsize=12,
             horizontalalignment='left',
             verticalalignment='top', 
             transform = ax.transAxes
             ) 

    plt.plot() 
    
    
#==============================================================================
#     Calculate 2a: The Distribution of variable "distance"
#                from the Delaunay Triangulation         
#==============================================================================

    start_time = time.time()
    #Delete indexes of contour pts form the Triangularization
    pt_ix = np.arange(0,pts.shape[0])
    pt_inner_ix = np.delete(pt_ix, V_border_ix, axis=0)
    pt_inner_ix = pt_ix # ??? bypass border poinnts
    V_d = []
    N = 4 #tetraedro
    
    for i in pt_inner_ix: 
        #0) Flatten the Vector which contains the indexes of 3 pts...
        #   ....that define each triangle
        t = T.flatten()
    
        #1) Take the index of a point. index: (x,y)
        pt_ref_ix = i
        
        #2) Get all triangles around that point:
        ix = np.where(t==pt_ref_ix)[0]//N
        T_near = T[ix]
        
        #3) Get the surrounding pts
        pt_near_ix = np.unique(T_near[T_near!=pt_ref_ix])
    
        #4) Get distances between pt_ref and all the pt_near
        d1 = np.sqrt(np.sum((pts[pt_ref_ix] - pts[pt_near_ix])**2, axis=1))
        for j in d1:            
            V_d.append(j)
               
        #5) Delete the triangules that were already evaluated
        # T = np.delete(T, ix, axis=0)

    
    
    V_d = np.asarray(V_d)
    dt = time.time() - start_time
    print('')
    print('---------------')
    print('dt=',dt) 

    

#==============================================================================
#    Plot: Distribution Distance (Euclidean Distance between Pairs of Points)
#==============================================================================
    fig, axs = plt.subplots(1,1) 
    #Calculate the Distribution of the variable "d"
    p = V_d
    n = 30
    hist, bins = np.histogram(p, bins=n, density=False)
    width = 0.7 * (bins[1] - bins[0])
    if width==0:
        width = 1
    center = (bins[:-1] + bins[1:]) / 2

    ax = axs
    ax.bar(center, hist, align='center', width=width)
    ax.set_xlim([np.min([p.min(), 0]), 2*np.max([p.max(), 0])])
    
#    ax.vlines(x=Ts, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle='--',      color='k', linewidth=2.0, label='Ts_In')       
#    ax.vlines(x=p.min(), ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle=':', color='g', linewidth=2.0, label='d_min')           
#    ax.vlines(x=1.0/f_G_max, ymin=ax.axes.get_ylim()[0], ymax=ax.axes.get_ylim()[1], linestyle=':', color='r', linewidth=2.0, label='Ts_Fourier')           

    mystr = ('Histogram: Distance')
    ax.set_title(mystr)
    ax.set_ylabel('Counts', fontsize= 12)
    ax.set_xlabel('Distance [um]', fontsize= 12)

    mystr = (
    'min : ' + "{0:.2f}".format(p.min()) + '\n' + 
    'avg : ' + "{0:.2f}".format(p.mean()) + '\n' +
    'med: ' + "{0:.2f}".format(np.median(p)) + '\n' +
    'max: ' + "{0:.2f}".format(p.max()) + '\n' + '\n' +
    
    'std  : ' + "{0:.2f}".format(p.std()) + '\n' +
    'stdN: ' + "{0:.2f}".format(p.std()/p.mean())    
    )              
    ax.text(1.05, 1.01, mystr,
             fontsize=12,
             horizontalalignment='left',
             verticalalignment='top', 
             transform = ax.transAxes
             )   
    plt.plot()     
    
    sys.exit()
    
    