# Generic imports
import os
import random
import shutil
import progress.bar
from datetime import datetime

from shapes_utils import *
from meshes_utils import *

### ************************************************
### Generate full dataset
## Parameters
# Shape controls
n_pts          = 6
n_sampling_pts = 40
radius         = [0.5]
edgy           = [1.0]

# Output control
mesh_domain    = True
plot_pts       = True
n_shapes       = 20

# Saving parameters
time           = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
dataset_dir    = 'dataset_'+time+'/'
mesh_dir       = dataset_dir+'meshes/'
img_dir        = dataset_dir+'images/'
csvf_dir        = dataset_dir+'csv/'
filename       = 'shape'

# Domain parameters
magnify        = 1.0
xmin           =-10.0
xmax           = 25.0
ymin           =-10.0
ymax           = 10.0
domain_h       = 0.2
n_cells_max    = 900000
n_cells_min    = 5000

# Create directories if necessary
if not os.path.exists(mesh_dir):
    os.makedirs(mesh_dir)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(csvf_dir):
    os.makedirs(csvf_dir)

# Generate dataset
bar = progress.bar.Bar('Generating shapes', max=n_shapes)
for i in range(0,n_shapes):

    generated = False
    while (not generated):

        # n_pts  = random.randint(3, 7)
        # radius = np.random.uniform(0.0, 1.0, size=n_pts)
        # edgy   = np.random.uniform(0.0, 1.0, size=n_pts)
        shape  = Shape(filename+'_'+str(i),
                       None,n_pts,n_sampling_pts,radius,edgy)

        shape.generate(magnify=magnify)
        meshed, n_cells = shape.mesh(mesh_domain=mesh_domain,
                                     xmin=xmin,
                                     xmax=xmax,
                                     ymin=ymin,
                                     ymax=ymax)
        #import pdb; pdb.set_trace()
        if (meshed and (n_cells < n_cells_max and n_cells > n_cells_min)):
            shape.generate_image(plot_pts=plot_pts,
                                 xmin=xmin,
                                 xmax=xmax,
                                 ymin=ymin,
                                 ymax=ymax)
            shape.write_csv()
            img  = filename+'_'+str(i)+'.png'
            mesh = filename+'_'+str(i)+'.msh'
            csvf = filename+'_'+str(i)+'.csv'
            shutil.move(img,  img_dir)
            shutil.move(mesh, mesh_dir)
            shutil.move(csvf, csvf_dir)
            generated = True

    bar.next()

# End bar
bar.finish()
