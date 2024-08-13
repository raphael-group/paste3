#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import squidpy as sq
import pandas as pd
import numpy as np


# # Install PASTE2 python package
# 
# You can install the paste2 package at https://pypi.org/project/paste2/. We import paste2 as follows:

# In[2]:


from paste2 import PASTE2, projection


# # Read in Spatial Transcriptomics slices as AnnData objects
# 
# We provide four example ST slices from DLPFC patient 3, cropped to form partially overlapping subslices (See Figure 3A of our paper). Each slice is stored in an [AnnData](https://anndata.readthedocs.io/en/latest/) object.

# In[3]:


sliceA_filename = 'sample_data/151673.h5ad'
sliceB_filename = 'sample_data/151674.h5ad'
sliceC_filename = 'sample_data/151675.h5ad'
sliceD_filename = 'sample_data/151676.h5ad'
sliceA = sc.read_h5ad(sliceA_filename)
sliceB = sc.read_h5ad(sliceB_filename)
sliceC = sc.read_h5ad(sliceC_filename)
sliceD = sc.read_h5ad(sliceD_filename)


# Each AnnData object consists of a gene expression matrx and spatial coordinate matrix. The gene expression matrix is stored in the .X field. The spatial coordiante matrix is stored in the .obsm['spatial'] field.

# In[4]:


sliceA.X


# In[5]:


sliceA.obsm['spatial']


# The rows of the AnnData objects are spots. The columns are genes.

# In[6]:


sliceA.obs


# In[7]:


sliceA.var


# We can visualize the slices using [squidpy](https://squidpy.readthedocs.io/en/stable/index.html). In this case, the .obs["layer_guess_reordered"] field stores the layer annotation of each slice, so we use this field to color each spot.

# In[8]:


sq.pl.spatial_scatter(
    sliceA,
    frameon=False,
    shape=None,
    color='layer_guess_reordered',
    figsize=(10, 10)
)
sq.pl.spatial_scatter(
    sliceB,
    frameon=False,
    shape=None,
    color='layer_guess_reordered',
    figsize=(10, 10)
)
sq.pl.spatial_scatter(
    sliceC,
    frameon=False,
    shape=None,
    color='layer_guess_reordered',
    figsize=(10, 10)
)
sq.pl.spatial_scatter(
    sliceD,
    frameon=False,
    shape=None,
    color='layer_guess_reordered',
    figsize=(10, 10)
)


# # Compute partial pairwise alignment using PASTE2
# 
# Give a pair of partially overlapping slices, we can use PASTE2.partial_pairwise_align( ) to find an alignment matrix. To call the function, you need to input the AnnData objects of the two slices, as well as a parameter s, which indicates the overlap percentage of the two slices. In this tutorial, each pair of cropped subslices overlap at 70% of the areas, so we set s=0.7. For your own datasets you should visualize the slices and manually determine the approxiamte overlap percentage (this parameter does not have to be very accurate).
# 
# Now we compute an alignment matrix between each pair of slices in our example dataset:

# In[10]:


pi_AB = PASTE2.partial_pairwise_align(sliceA, sliceB, s=0.7)


# In[11]:


pi_BC = PASTE2.partial_pairwise_align(sliceB, sliceC, s=0.7)


# In[12]:


pi_CD = PASTE2.partial_pairwise_align(sliceC, sliceD, s=0.7)


# Let's check the shape of each alignment matrix. For aligning a slice with n1 spots and a slice with n2 spots, the alignment matrix should be of shape (n1 * n2)

# In[13]:


print(pi_AB.shape)
print(pi_BC.shape)
print(pi_CD.shape)


# There are other optional parameters to PASTE2.partial_pairwise_align() as well. You can checkout the original function signature in the souce code with documentation.
# 
# Let's visualize the alignment between sliceA and sliceB:

# In[14]:


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def plot2D_samples_mat(xs, xt, G, thr=1e-8, alpha=0.2, top=1000, weight_alpha=False, **kwargs):
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    #     idx = np.where(G/mx>=thr)
    idx = largest_indices(G, top)
    for l in range(len(idx[0])):
        plt.plot([xs[idx[0][l], 0], xt[idx[1][l], 0]], [xs[idx[0][l], 1], xt[idx[1][l], 1]],
                 alpha=alpha * (1 - weight_alpha) + (weight_alpha * G[idx[0][l], idx[1][l]] / mx), c='k')


def plot_slice_pairwise_alignment(slice1, slice2, pi, thr=1 - 1e-8, alpha=0.05, top=1000, name='',
                                  weight_alpha=False):
    coordinates1, coordinates2 = slice1.obsm['spatial'], slice2.obsm['spatial']
    offset = (coordinates1[:, 0].max() - coordinates2[:, 0].min()) * 1.1
    temp = np.zeros(coordinates2.shape)
    temp[:, 0] = offset
    plt.figure(figsize=(20, 10))
    plot2D_samples_mat(coordinates1, coordinates2 + temp, pi, thr=thr, c='k', alpha=alpha, top=top,
                       weight_alpha=weight_alpha)
    plt.scatter(coordinates1[:, 0], coordinates1[:, 1], linewidth=0, s=100, marker=".", color=list(
        slice1.obs['layer_guess_reordered'].map(
            dict(zip(slice1.obs['layer_guess_reordered'].cat.categories, slice1.uns['layer_guess_reordered_colors'])))))
    plt.scatter(coordinates2[:, 0] + offset, coordinates2[:, 1], linewidth=0, s=100, marker=".", color=list(
        slice2.obs['layer_guess_reordered'].map(
            dict(zip(slice2.obs['layer_guess_reordered'].cat.categories, slice2.uns['layer_guess_reordered_colors'])))))
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()
    
    
plot_slice_pairwise_alignment(sliceA, sliceB, pi_AB)


# In[ ]:





# # Project all slices onto the same coordiante system according to the alignment
# 
# Once the alignment matrix between each pair of adjacent slices in a sequence of consecutive slices are computed, we can use this information to project all slices onto the same 2D coordinate system. 3D reconstruction can be done by assiging a z-coordiante to each slice after the projection.
# 
# Specifically, we use projection.partial_stack_slices_pairwise( ):

# In[15]:


pis = [pi_AB, pi_BC, pi_CD]
slices = [sliceA, sliceB, sliceC, sliceD]

new_slices = projection.partial_stack_slices_pairwise(slices, pis)


# Now let's plot the coordinates of all slices after the projection:

# In[16]:


layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]
def plot_slices_overlap(slices, layer_to_color_map=layer_to_color_map):
    plt.figure(figsize=(10,10))
    for i in range(len(slices)):
        adata = slices[i]
        colors = list(adata.obs['layer_guess_reordered'].astype('str').map(layer_to_color_map))
        plt.scatter(adata.obsm['spatial'][:,0],adata.obsm['spatial'][:,1],linewidth=0,s=100, marker=".",color=colors)
    plt.legend(handles=[mpatches.Patch(color=layer_to_color_map[adata.obs['layer_guess_reordered'].cat.categories[i]], label=adata.obs['layer_guess_reordered'].cat.categories[i]) for i in range(len(adata.obs['layer_guess_reordered'].cat.categories))],fontsize=10,title='Cortex layer',title_fontsize=15,bbox_to_anchor=(1, 1))
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()
    
plot_slices_overlap(new_slices)


# Or just the first two, which reproduces Figure 3C of the paper:

# In[17]:


plot_slices_overlap(new_slices[:2])


# # Let me know if PASTE2 runs on your machine! If you run into any problem don't hesitate to reach out at xl5434@princeton.edu. I will respond as quickly as possible.

# In[ ]:




