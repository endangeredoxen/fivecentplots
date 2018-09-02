
# coding: utf-8

# # <b>heatmap</b>

# This section provides examples of how to use the <b>heatmap</b> function.  At a minimum, the `heatmap` function requires the following keywords:
# <ul>
# <li>`df`: a pandas DataFrame</li>
# <li>`x`: the name of the DataFrame column containing the x-axis data</li>
# <li>`y`: the name of the DataFrame column containing the y-axis data</li>
# <li>`z`: the name of the DataFrame column containing the z-axis data</li>
# </ul>
# 
# Heatmaps in <b><font color="blue" style="font-family:'Courier New'">fivecentplots </font></b> can display both categorical and non-categorical data on either a uniform or non-uniform grid.

# ## Setup

# ### Imports

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import fivecentplots as fcp
import pandas as pd
import numpy as np
import os, sys, pdb
osjoin = os.path.join
st = pdb.set_trace


# ### Sample data

# In[2]:


df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_heatmap.csv'))
df.head()


# ### Set theme

# In[3]:


#fcp.set_theme('gray')
#fcp.set_theme('white')


# ### Other

# In[4]:


SHOW = False


# ## Categorical heatmap

# First consider a case where both the `x` and `y` DataFrame columns contain categorical data values:

# ### No data labels

# In[5]:


fcp.heatmap(df=df, x='Category', y='Player', z='Average', cbar=True, show=SHOW)


# Note that for heatmaps the `x` tick labels are rotated 90&#176; by default.  This can be overridden via the keyword `tick_labels_major_x_rotation`.

# ### With data labels

# In[6]:


fcp.heatmap(df=df, x='Category', y='Player', z='Average', cbar=True, data_labels=True, 
            heatmap_font_color='#aaaaaa', show=SHOW, tick_labels_major_y_edge_width=0, ws_ticks_ax=5)


# ## Non-uniform data

# A major difference between heatmaps and contour plots is that contour plots assume that the `x` and `y` DataFrame column values are numerical and continuous.  With a heatmap, we can cast numerical data into categorical form.  Note that any missing values get mapped as `nan` values are not not plotted.  

# In[7]:


# Read the contour DataFrame
df2 = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_contour.csv'))


# In[8]:


fcp.heatmap(df=df2, x='X', y='Y', z='Value', row='Batch', col='Experiment', 
            cbar=True, show=SHOW, share_z=True, ax_size=[400, 400],
            data_labels=False, label_rc_font_size=12, filter='Batch==103', cmap='viridis')


# Note that the x-axis width is not 400px as specified by the keyword `ax_scale`.  This occurs because the data set does not have as many values on the x-axis as on the y-axis.  <b><font color="blue" style="font-family:'Courier New'">fivecentplots </font></b> applies the axis size to the axis with the most items and scales the other axis accordingly.

# ## imshow alternative

# We can also use `fcp.heatmap` to display images (similar to `imshow` in matplotlib).  Here we will take a random image from the world wide web, place it in a pandas DataFrame, and display.

# In[9]:


# Read an image
import imageio
url = 'https://s4827.pcdn.co/wp-content/uploads/2011/04/low-light-iphone4.jpg'
imgr = imageio.imread(url)

# Convert to grayscale
r, g, b = imgr[:,:,0], imgr[:,:,1], imgr[:,:,2]
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

# Convert image data to pandas DataFrame
img = pd.DataFrame(gray)
img.head()


# Display the image as a colored heatmap:

# In[10]:


fcp.heatmap(img, cmap='inferno', cbar=True, ax_size=[600, 600])


# Now let's enhance the contrast of the same image by limiting our color range to the mean pixel value +/- 3 * sigma:

# In[11]:


uu = img.stack().mean()
ss = img.stack().std()
fcp.heatmap(img, cmap='inferno', cbar=True, ax_size=[600, 600], zmin=uu-3*ss, zmax=uu+3*ss)


# We can also crop the image by specifying range value for `x` and `y`.  Unlike `imshow`, the actual row and column values displayed on the x- and y-axis are preserved after the zoom (not reset to 0, 0):

# In[12]:


fcp.heatmap(img, cmap='inferno', cbar=True, ax_size=[600, 600], xmin=1400, xmax=2000, ymin=500, ymax=1000)


# <i> private eyes are watching you... </i>
