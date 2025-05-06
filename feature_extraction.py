#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import os
import time
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
import trimesh
import pyvista as pv
import pymeshfix as mf
from pymeshfix._meshfix import PyTMesh


import vedo
from vedo import *

import open3d as o3d
from open3d import *


# In[ ]:


# Correct Mesh
def correct_mesh(iso, smooth=0, taubin=False):
   # Extract vertices and faces
    vertices = iso.vertices 
    faces = iso.cells
    # Clean vertices and faces
    vertices, faces = mf.clean_from_arrays(vertices, faces)

    # Load PyTMesh
    mfix = PyTMesh(False)  # False removes extra verbose output
    # Create array
    mfix.load_array(vertices, faces)
    
    # Fix mesh
        # Fills all the holes having at at most 'nbe' boundary edges. If
        # 'refine' is true, adds inner vertices to reproduce the sampling
        # density of the surroundings. Returns number of holes patched.  If
        # 'nbe' is 0 (default), all the holes are patched.
    mfix.fill_small_boundaries(nbe=0, refine=True)
    
    # Return vertices and faces
    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3
    # Create mesh
    mesh = pv.PolyData(vert, triangles)
    # Apply Smoothing if Assigned
    if smooth > 0:
        if taubin:
            # Smooth mesh
            mesh = mesh.smooth_taubin(n_iter=smooth, non_manifold_smoothing=True)
        else:
            mesh = mesh.smooth(n_iter=smooth)

    return mesh



# In[ ]:


# 12 features derived from shape measurements
def shape_measurments(pv_mesh, r=1.0): # set search radius [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]
    # Load the point cloud and normals
    point_cloud = np.asarray(pv_mesh.points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
# Remove Outliers ### EXPERIMENTAL ###########################
    # Downsample the Point Cloud with a Voxel of 0.01
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # Statistical Oulier Removal
    pcd, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=100,
                                                        std_ratio=2.0)
### END OF EXPERIMENTA ####################################### 

    pcd.compute_convex_hull()
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)
    
    mesh_verts = np.asarray(pcd.points)
    mesh_norms = np.asarray(pcd.normals)
        
    # #  Transform the point cloud to the a PointCloud() object used by open3d        
    pcd.points = utility.Vector3dVector(mesh_verts)
    pcd.normals = utility.Vector3dVector(mesh_norms)
    
    # Calculate the KDTree
    pcd_tree = geometry.KDTreeFlann(pcd)
    
    allVerts = np.ones([len(mesh_verts),1])
    
    # how many scales will be used 
    numScales = 1
    
    # Place holder for set of features
    feature_set = []
    
    i=0
    
    #  go through all the points
    while i<len(mesh_verts):
            
            pointScaleFeatures = []
            
            for j in [r]: 
                
                # Constant for Stability 
                constant = 0.0
                #  for each radius area scale find neighbours
                [k_small, idx_small, distances_small] = pcd_tree.search_radius_vector_3d(pcd.points[i], j) # radius search set radius
            
              
                currNdx = np.array(idx_small)
                #  if there are less than 2 neightbours just add 0s 
                if (len(currNdx) <=2):
                    linearity= 0 + constant
                    planarity=0 + constant
                    sphericity=0 + constant
                    omnivariance=0 + constant
                    anisotropy=0 + constant
                    eigenentropy=0 + constant
                    sumOFEigs=0 + constant
                    changeOfCurvature=0 + constant
                    farthestDist =  distances_small[len(distances_small)-1]/j
                    pointDensity = k_small/j
                    heightStd=0 + constant
                    heightMax=0 + constant
                    
                    shapeDist_curr = np.zeros(50)
                #  if there are enouigh neighbours then continue with computation
                else:
                    
                    #  get heighbourhood points and normals
                    nearestNeighbors_normals = mesh_norms[currNdx]
                    nearestNeighbors_verts = mesh_verts[currNdx]
                    
                    #  calculate the covariance matrix, and eigenvalues
                    cov_mat = np.cov([nearestNeighbors_verts[:,0],nearestNeighbors_verts[:,1],nearestNeighbors_verts[:,2]])
                    eig_val_cov, eig_vec_cov = np.linalg.eigh(cov_mat)
                    idx = eig_val_cov.argsort()[::-1]  
                    eig_val_cov = eig_val_cov[idx]
                    
                    #  calculate the first 12 features derived from shape measurements
                    linearity = (eig_val_cov[0] - eig_val_cov[1])/eig_val_cov[0]
                    planarity = (eig_val_cov[1] - eig_val_cov[2])/eig_val_cov[0]
                    sphericity = eig_val_cov[2]/eig_val_cov[0]
                    omnivariance =(eig_val_cov[0]*eig_val_cov[1]*eig_val_cov[2]) **(1./3.)
                    anisotropy = (eig_val_cov[0] - eig_val_cov[2])/eig_val_cov[0]
                    eigenentropy = -(( eig_val_cov[0] * np.log(eig_val_cov[0])) + ( eig_val_cov[1] * np.log(eig_val_cov[1])) + ( eig_val_cov[2] * np.log(eig_val_cov[2])))
                    sumOFEigs = eig_val_cov[0]+ eig_val_cov[1]+eig_val_cov[2]
                    changeOfCurvature = eig_val_cov[2]/(eig_val_cov[0]+ eig_val_cov[1]+eig_val_cov[2])
                    
                    farthestDist =  distances_small[len(distances_small)-1]/j
                    pointDensity = k_small/j
                    heightMax = np.abs(np.dot(nearestNeighbors_normals.mean(axis=0),nearestNeighbors_verts.T)).max() - np.abs(np.dot(nearestNeighbors_normals.mean(axis=0),nearestNeighbors_verts.T)).min()
                    heightStd = np.abs(np.dot(nearestNeighbors_normals.mean(axis=0),nearestNeighbors_verts.T)).std()
               
                #  Added to the other features and a feature vector is created
                pointScaleFeatures.extend([linearity,planarity,sphericity,omnivariance,anisotropy,eigenentropy,sumOFEigs,changeOfCurvature,farthestDist,pointDensity,heightMax,heightStd])
            #  The feature vector for each point is checked for NaN values and then concatenated
            where_are_NaNs = np.isnan(pointScaleFeatures)
            pointScaleFeatures = np.array(pointScaleFeatures)
            pointScaleFeatures[where_are_NaNs] = 0
            pointScaleFeatures = pointScaleFeatures.tolist()
            feature_set.append(pointScaleFeatures)
            
            i+=1
    
    # DataFrame of averaged features
    col_names = ['linearity','planarity','sphericity','omnivariance','anisotropy','eigenentropy','sumOFEigs','changeOfCurvature','farthestDist','pointDensity','heightStd', 'heightMax']
    feature_df =  pd.DataFrame(feature_set, columns = col_names) # TO GET ENTIRE DATA FOR PLOTTTING

    # Get  values
    final_df = feature_df.mean()
        
    return final_df
    


# In[ ]:


# Compute Mesh Features

# Container for All Data
all_data_list = []

# Setup Parameters
structure=[24,25] # flag surface of interest
ndecim = 6000 # mesh granularity
# Smooth Parameters
lamb = 1.0
itr = 5

# Scans to Exclude
corrupt_scan_id = [] # subject IDs to exclude

# Directory to mine
main_dir = '' # input path to directory of interest


vent_size = [] # non processed mesh size

# Starting Time
program_starts = time.time()
        
# ID's directory
id_dirs = os.listdir(main_dir)
# Sub-directory
sub_dirs = [os.path.join(main_dir, sub_id) for sub_id in id_dirs if not 'DS_Store' in sub_id and not 'Mesh_Params' in sub_id]
for sub_dir in sub_dirs:
                ses_dir = os.listdir(os.path.join(sub_dir))
                for ses in ses_dir[:2]:
                    if not 'DS_Store' in ses:
                        scan_dir = os.listdir(os.path.join(sub_dir, ses))
                        synthseg_files = [file for file in scan_dir if 'sub' in file] # cneuro
                        for synthseg_file in synthseg_files:
                            # Define source path
                            src_path = os.path.join(sub_dir, ses, synthseg_file)
                            # Path details
                            file_path = src_path
                            # Find a base name
                            basename = os.path.basename(file_path)
                            # Get ID
                            sub_id = basename.split('_')[0] # cNeuro
                            if sub_id not in corrupt_scan_id:
                                
                                    ### MESH PROCESSING ########################################################################### 
                                    # Load File
                                    nifti_file = nib.load(file_path) # load file from the path
                                    segmentation_array  = nifti_file.get_fdata() # convert nifti segmentation file into numpy array
                                    # Filter brain structure of interest and set a scene for mesh
                                    scene = np.where(
                                                    (segmentation_array!=0) & # remove background
                                                    np.isin(segmentation_array.astype(int), structure), # flag structure of interest
                                                    1, 0) # assign values
                                                     
                                    # Create volume using Vedo surfnets and use dual approach to create isosurface     
                                    # Extract volume
                                    vol = Volume(scene.astype(int),spacing =nifti_file.header.get_zooms()) # Extract volume and correct strectching along all axis
                                    # Create mesh from mask
                                    iso = vol.isosurface_discrete([1,]) # discreate
                                
                                    ### CLEAN ###########################################################################
                                    # Create Clean Mesh
                                    mesh_obj = correct_mesh(iso) # smooth=s, taubin=taubin_state
                                                                
                                    # Pyvista Mesh
                                    pv_mesh = pv.wrap(mesh_obj)
                                    # Extract Faces
                                    pv_faces = pv_mesh.faces
                                    # Extract Vertices
                                    pv_vertices = pv_mesh.points
                                    # Back to Vedo Mesh
                                    iso = vedo.Mesh([pv_vertices, pv_faces])
                                    
                                    ### SUBDIVIDE ###########################################################################
                                    if pv_vertices.shape[0] < ndecim:
                                        # Extract Vertices
                                        vedo_vertices = iso.vertices 
                                        # Extract Faces
                                        vedo_faces = iso.cells
                                        # Subdivide Mesh
                                        tri_vertices, tri_faces =  trimesh.remesh.subdivide_loop(vedo_vertices, np.array(vedo_faces).astype(int), iterations=1)
                                        # Tri Mesh
                                        tri_iso = trimesh.Trimesh(tri_vertices, tri_faces) 
                                        # Convert to Vedo Mesh
                                        iso = vedo.trimesh2vedo(tri_iso)

                                    ### SMOOTH ###########################################################################
                                    # Correct Mesh
                                    pv_mesh =  pv.wrap(correct_mesh(iso))
                                    # Convert to Tri Mesh
                                    pv_mesh_faces = pv_mesh.faces.reshape((pv_mesh.n_cells, 4))[:, 1:] 
                                    # Convert to Tri Mesh
                                    tri_iso = trimesh.Trimesh(pv_mesh.points, pv_mesh_faces) 
                                    # Convert to Vedo Mesh
                                    iso = vedo.trimesh2vedo(tri_iso)
                                
                                    # ### DECIMATE ###########################################################################
                                    # Decimate Pro
                                    iso = iso.decimate_pro(
                                        	n=ndecim,
                                        	# preserve_topology=False,
                                        	# preserve_boundaries=False,
                                        	splitting=True
                                            )
                                    ### VEDO Smooth ###########################################################################

                                    # Correct Mesh
                                    pv_mesh =  pv.wrap(correct_mesh(iso))
                                    # Convert to Tri Mesh
                                    pv_mesh_faces = pv_mesh.faces.reshape((pv_mesh.n_cells, 4))[:, 1:] 
                                    # Convert to Tri Mesh
                                    tri_mesh = trimesh.Trimesh(pv_mesh.points, pv_mesh_faces) 
                                    # Smooth Mesh
                                    trimesh.smoothing.filter_laplacian(tri_mesh, lamb=lamb,  iterations=itr,)

#### TEST 
                                    # Convert to Vedo Mesh
                                    iso = vedo.trimesh2vedo(tri_mesh)
                                    # Correct Mesh
                                    pv_mesh =  pv.wrap(correct_mesh(iso))
                                    # Convert to Tri Mesh
                                    pv_mesh_faces = pv_mesh.faces.reshape((pv_mesh.n_cells, 4))[:, 1:] 
                                    # Convert to Tri Mesh
                                    tri_mesh = trimesh.Trimesh(pv_mesh.points, pv_mesh_faces) 

                                
                                    ### COMPUTE FEATURES ###########################################################################
                                    # Register Setup Parameters
                                    d = ndecim
                                    setup = f"s{lamb}_i{itr}_d{d}"
                                                                            
                                    # Compute Curvature Features
                                    # Tri mesh
                                    IMC = tri_mesh.integral_mean_curvature # Integral Mean Curvature
                                                                                                                
                                    # Compute Global Features
                                    # Tri mesh
                                    gf_tri_mesh = tri_mesh
                                    SA = gf_tri_mesh.area
                                    V = gf_tri_mesh.volume
                                    SAVR  = SA/V
                                                                         
                                    # Capture Asphericity
                                    # Generate points
                                    pts = pv_mesh.points
                                    # Find the best fitting ellipsoid to the points
                                    elli = pca_ellipsoid(pts, pvalue=0.95) #  https://vedo.embl.es/docs/vedo/pointcloud.html#pca_ellipsoid
                                    AS = elli.asphericity()  # asphericity
                                    ASE = elli.asphericity_error()  # error on asphericity
                                                                        
                                    # Capture Convexity and Geometry Features
                                    # Tri mesh
                                    cx_tri_mesh = tri_mesh
                                    # Compute Shape volume to Convex Hull volume ratio
                                    CxVR = V / cx_tri_mesh.convex_hull.volume # Volume devided by convex hull volume
                                    # Compute Shape area to Convex Hull area ratio
                                    CxSAR = SA / cx_tri_mesh.convex_hull.area
                                    # Compute Convex Hull volume
                                    CxV = cx_tri_mesh.convex_hull.volume
                                    # Compute Convex Hull area
                                    CxSA = cx_tri_mesh.convex_hull.area   
                                    # Compute Convex Hull area to volume ratio
                                    CxSAVR = CxSA / CxV
                                
                                    # Save Mesh
                                    ses_path =  os.path.join(sub_dir, ses) # session dir path
                                    mesh_path =  os.path.join(ses_path, f'{sub_id}_mesh.stl') # mesh destination path
                                    pv_mesh.save(mesh_path)  
                                                                        
                                    # Features: ['linearity','planarity','sphericity','omnivariance','anisotropy','eigenentropy','sumOFEigs','changeOfCurvature','farthestDist','pointDensity','heightStd', 'heightMax']
                                    shape_m = shape_measurments(pv_mesh)
                                    L = shape_m['linearity']
                                    P = shape_m['planarity']
                                    SP = shape_m['sphericity']
                                    O = shape_m['omnivariance']
                                    AT = shape_m['anisotropy']
                                    ET = shape_m['eigenentropy']
                                    ES = shape_m['sumOFEigs']
                                    CC = shape_m['changeOfCurvature']
                                    FD = shape_m['farthestDist']
                                    PD = shape_m['pointDensity']
                                    Hmax = shape_m['heightMax']
                                    Hsd = shape_m['heightStd']
        
                                                                    
                                    # PyVista Curvature Mean
                                    GGC = np.mean(pv_mesh.curvature('gaussian'))
                                    GMC = np.mean(pv_mesh.curvature('mean'))
                                    Cmax = np.mean(pv_mesh.curvature('maximum'))
                                    Cmin = np.mean(pv_mesh.curvature('minimum'))
                                                                            
                                    # Create feature DataFrame
                                    # Measurment Names
                                    var_names = ['SA', 'V', 'SAVR',  'IMC', 'As', 'AsE', 
                                    'GC', 'MC', 'CMax', 'CMin', 
                                    'L', 'P', 'S', 'O', 'A', 'EE', 'SE', 'CC', 'PD', 'FD', 'HMax', 'HSD',  
                                    'CHV', 'CHSA', 'CHSAVR', 'CHSAR', 'CHVR',]
                                    # Measurments
                                    measurments = [SA,V,SAVR,IMC,AS,ASE,
                                                   GGC,GMC,Cmax,Cmin,
                                                   L,P,SP,O,AT,ET,ES,CC,FD,PD,Hmax,Hsd,
                                                   CxV, CxSA,CxSAVR,CxSAR,CxVR,]
                                
                                    col_names = ['MEASURMENT', 'VALUE']
                                    measurment_DF = pd.DataFrame(zip(var_names, measurments), columns=col_names)
                                    print(measurment_DF)
                                    # Extract Feature Values
                                    f_values = measurments
                                    # Add ID
                                    f_values.append(sub_id)
                                    # Add Setup
                                    f_values.append(setup)
                                    # Add Row to All Data List 
                                    all_data_list.append(f_values)
                                    # Time per Loop
                                    now = time.time()
                                    print("It took {} seconds to process {} setup {}".format(now - program_starts,sub_id,setup))
                                                                                           
        
# Create Final DataFrame 
all_data_cols = ['SA', 'V', 'SAVR',  'IMC', 'As', 'AsE', 
                    'GC', 'MC', 'CMax', 'CMin', 
                    'L', 'P', 'S', 'O', 'A', 'EE', 'SE', 'CC', 'PD', 'FD', 'HMax', 'HSD',  
                    'CHV', 'CHSA', 'CHSAVR', 'CHSAR', 'CHVR',
                     'ID', 'SETUP'] 
all_data_df = pd.DataFrame(all_data_list, columns = all_data_cols)
# Date Params
year = datetime.now().year
month = datetime.now().month
day = datetime.now().day
feature_data_dst = os.path.join(main_dir, f'feature_df_{day}-{month}-{year}.csv') # feature data dst
# Save Data
all_data_df.to_csv(feature_data_dst)



