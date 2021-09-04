# Generic imports
import os
import os.path
import PIL
import math
import scipy.special
import matplotlib
import numpy             as np
import matplotlib.pyplot as plt
import pygmsh, meshio, gmsh
import itertools

from .meshes_utils import *

### ************************************************
### Class defining shape object
class Shape:
    ### ************************************************
    ### Constructor
    default_name_counter = itertools.count()
    def __init__(self,
                 name          =None,
                 control_pts   =None,
                 n_control_pts =None,
                 n_sampling_pts=None,
                 radius        =None,
                 edgy          =None):
        if (name           is None): name           = '/tmp/shape_' + str(next(self.default_name_counter))
        if (control_pts    is None): control_pts    = np.array([])
        if (n_control_pts  is None): n_control_pts  = 0
        if (n_sampling_pts is None): n_sampling_pts = 0
        if (radius         is None): radius         = np.array([])
        if (edgy           is None): edgy           = np.array([])
        
        self.name           = name
        self.control_pts    = control_pts
        self.n_control_pts  = n_control_pts
        self.n_sampling_pts = n_sampling_pts
        self.curve_pts      = np.array([])
        self.area           = 0.0
        self.size_x         = 0.0
        self.size_y         = 0.0
        self.index          = 0
	

        if (len(radius) == n_control_pts): self.radius = radius
        if (len(radius) == 1):             self.radius = radius*np.ones([n_control_pts])

        if (len(edgy) == n_control_pts):   self.edgy = edgy
        if (len(edgy) == 1):               self.edgy = edgy*np.ones([n_control_pts])
	
        subname             = name.split('_')
        if (len(subname) == 2): # name is of the form shape_?.xxx
            self.name       = subname[0]
            index           = subname[1].split('.')[0]
            self.index      = int(index)
        if (len(subname) >  2): # name contains several '_'
            raise(ValueError('Please do not use several "_" char in shape name'))

        if (len(control_pts) > 0):
            self.control_pts   = control_pts
            self.n_control_pts = len(control_pts)

    ### ************************************************
    ### Reset object
    def reset(self):
        self.name           = 'shape'
        self.control_pts    = np.array([])
        self.n_control_pts  = 0
        self.n_sampling_pts = 0
        self.radius         = np.array([])
        self.edgy           = np.array([])
        self.curve_pts      = np.array([])
        self.area           = 0.0

    ### ************************************************
    ### Generate shape
    def generate(self, *args, **kwargs):
        # Handle optional argument
        centering = kwargs.get('centering', True)
        cylinder  = kwargs.get('cylinder',  False)
        magnify   = kwargs.get('magnify',   1.0)
        ccws      = kwargs.get('ccws',      True)

        # Generate random control points if empty
        if (len(self.control_pts) == 0):
            if (cylinder):
                self.control_pts = generate_cylinder_pts(self.n_control_pts)
            else:
                self.control_pts = generate_random_pts(self.n_control_pts)

        # Magnify
        self.control_pts *= magnify

        # Center set of points
        center = np.mean(self.control_pts, axis=0)
        if (centering):
            self.control_pts -= center

        # Sort points counter-clockwise
        if (ccws):
            control_pts, radius, edgy  = ccw_sort(self.control_pts,
                                                  self.radius,
                                                  self.edgy)
        else:
            control_pts = np.array(self.control_pts)
            radius      = np.array(self.radius)
            edgy        = np.array(self.edgy)

        local_curves = []
        delta        = np.zeros([self.n_control_pts,2])
        radii        = np.zeros([self.n_control_pts,2])
        delta_b      = np.zeros([self.n_control_pts,2])

        # Compute all informations to generate curves
        for i in range(self.n_control_pts):
            # Collect points
            prv  = (i-1)
            crt  = i
            nxt  = (i+1)%self.n_control_pts
            pt_m = control_pts[prv,:]
            pt_c = control_pts[crt,:]
            pt_p = control_pts[nxt,:]

            # Compute delta vector
            diff         = pt_p - pt_m
            diff         = diff/np.linalg.norm(diff)
            delta[crt,:] = diff

            # Compute edgy vector
            delta_b[crt,:] = 0.5*(pt_m + pt_p) - pt_c

            # Compute radii
            dist         = compute_distance(pt_m, pt_c)
            radii[crt,0] = 0.5*dist*radius[crt]
            dist         = compute_distance(pt_c, pt_p)
            radii[crt,1] = 0.5*dist*radius[crt]

        # Generate curves
        for i in range(self.n_control_pts):
            crt  = i
            nxt  = (i+1)%self.n_control_pts
            pt_c = control_pts[crt,:]
            pt_p = control_pts[nxt,:]
            dist = compute_distance(pt_c, pt_p)
            smpl = math.ceil(self.n_sampling_pts*math.sqrt(dist))

            local_curve = generate_bezier_curve(pt_c,           pt_p,
                                                delta[crt,:],   delta[nxt,:],
                                                delta_b[crt,:], delta_b[nxt,:],
                                                radii[crt,1],   radii[nxt,0],
                                                edgy[crt],      edgy[nxt],
                                                smpl)
            local_curves.append(local_curve)

        curve          = np.concatenate([c for c in local_curves])
        x, y           = curve.T
        z              = np.zeros(x.size)
        self.curve_pts = np.column_stack((x,y,z))
        self.curve_pts = remove_duplicate_pts(self.curve_pts)

        # Center set of points
        if (centering):
            center            = np.mean(self.curve_pts, axis=0)
            self.curve_pts   -= center
            self.control_pts[:,0:2] -= center[0:2]

        # Compute area
        self.compute_area()

        # Compute dimensions
        self.compute_dimensions()

    ### ************************************************
    ### Write image
    def generate_image(self, *args, **kwargs):
        # Handle optional argument
        plot_pts       = kwargs.get('plot_pts',       False)
        override_name  = kwargs.get('override_name',  '')
        show_quadrants = kwargs.get('show_quadrants', False)
        max_radius     = kwargs.get('max_radius',     1.0)
        min_radius     = kwargs.get('min_radius',     0.2)
        xmin           = kwargs.get('xmin',          -1.0)
        xmax           = kwargs.get('xmax',           1.0)
        ymin           = kwargs.get('ymin',          -1.0)
        ymax           = kwargs.get('ymax',           1.0)

        # Plot shape
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.fill([xmin,xmax,xmax,xmin],
                 [ymin,ymin,ymax,ymax],
                 color=(0.784,0.773,0.741),
                 linewidth=2.5,
                 zorder=0)
        plt.fill(self.curve_pts[:,0],
                 self.curve_pts[:,1],
                 'black',
                 linewidth=0,
                 zorder=1)

        # Plot points
        # Each point gets a different color
        if (plot_pts):
            colors = matplotlib.cm.ocean(np.linspace(0, 1, self.n_control_pts))
            plt.scatter(self.control_pts[:,0],
                        self.control_pts[:,1],
                        color=colors,
                        s=16,
                        zorder=2,
                        alpha=0.5)

        # Plot quadrants
        if (show_quadrants):
            for pt in range(self.n_control_pts):
                dangle = (360.0/float(self.n_control_pts))
                angle  = dangle*float(pt)+dangle/2.0
                x_max  = max_radius*math.cos(math.radians(angle))
                y_max  = max_radius*math.sin(math.radians(angle))
                x_min  = min_radius*math.cos(math.radians(angle))
                y_min  = min_radius*math.sin(math.radians(angle))
                plt.plot([x_min, x_max],
                         [y_min, y_max],
                         color='w',
                         linewidth=1)

            circle = plt.Circle((0,0),max_radius,fill=False,color='w')
            plt.gcf().gca().add_artist(circle)
            circle = plt.Circle((0,0),min_radius,fill=False,color='w')
            plt.gcf().gca().add_artist(circle)

        # Save image
        filename = self.name+'_'+str(self.index)+'.png'
        if (override_name != ''): filename = override_name

        plt.savefig(filename,
                    dpi=400)
        plt.close(plt.gcf())
        plt.cla()
        trim_white(filename)

    ### ************************************************
    ### Write csv
    def write_csv(self):
        with open(self.name+'_'+str(self.index)+'.csv','w') as file:
            # Write header
            file.write('{} {}\n'.format(self.n_control_pts,
                                        self.n_sampling_pts))

            # Write control points coordinates
            for i in range(0,self.n_control_pts):
                file.write('{} {} {} {}\n'.format(self.control_pts[i,0],
                                                  self.control_pts[i,1],
                                                  self.radius[i],
                                                  self.edgy[i]))

    ### ************************************************
    ### Read csv and initialize shape with it
    def read_csv(self, filename, *args, **kwargs):
        # Handle optional argument
        keep_numbering = kwargs.get('keep_numbering', False)

        if (not os.path.isfile(filename)):
            print('I could not find csv file: '+filename)
            print('Exiting now')
            exit()

        self.reset()
        sfile  = filename.split('.')
        sfile  = sfile[-2]
        sfile  = sfile.split('/')
        name   = sfile[-1]

        if (keep_numbering):
            sname = name.split('_')
            name  = sname[0]
            name  = name+'_'+str(self.index)

        x      = []
        y      = []
        radius = []
        edgy   = []

        with open(filename) as file:
            header         = file.readline().split()
            n_control_pts  = int(header[0])
            n_sampling_pts = int(header[1])

            for i in range(0,n_control_pts):
                coords = file.readline().split()
                x.append(float(coords[0]))
                y.append(float(coords[1]))
                radius.append(float(coords[2]))
                edgy.append(float(coords[3]))
                control_pts = np.column_stack((x,y))

        self.__init__(name,
                      control_pts,
                      n_control_pts,
                      n_sampling_pts,
                      radius,
                      edgy)

    ### ************************************************
    ### Mesh shape
    def mesh(self, mesh_domain = False, xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0, mesh_format = 'msh', extruded = False, element_scaler = 3, wake_refined = True):
        Ny_outer       = 8*element_scaler if wake_refined else (30*element_scaler)
        Ny_inner       = 12*element_scaler if wake_refined else (10*element_scaler)
        Ny_outer_right = 26*element_scaler if wake_refined else (86*element_scaler)
        Ny_outer_left  = 3*element_scaler if wake_refined else (30*element_scaler)
        Nx_inner       = 4*element_scaler if wake_refined else (10*element_scaler)
        circle_rad     = 1.5
        # Convert curve to polygon
        with pygmsh.geo.Geometry() as geom:
            # bounding box
            p1 = geom.add_point((xmin, ymax, 0))
            p2 = geom.add_point((xmax, ymax, 0))
            p3 = geom.add_point((xmax, ymin, 0))
            p4 = geom.add_point((xmin, ymin, 0))
            poly = geom.add_polygon(self.curve_pts,
                                    mesh_size=0.006233,
                                    make_surface=not mesh_domain)

            # Mesh domain if necessary
            if (mesh_domain):
                # ********************  Adding geometry points  ********************
                # outer circular points
                p5 = geom.add_point((circle_rad, circle_rad, 0))
                p6 = geom.add_point((-circle_rad, circle_rad, 0))
                p7 = geom.add_point((circle_rad, -circle_rad, 0))
                p8 = geom.add_point((-circle_rad, -circle_rad, 0))

                # intermediate outer bounds
                p9= geom.add_point((xmin, circle_rad, 0))
                p10 = geom.add_point((xmin, -circle_rad, 0))
                p11 = geom.add_point((-circle_rad, ymin, 0))
                p12 = geom.add_point((-circle_rad, ymax, 0))
                p13 = geom.add_point((circle_rad, ymax, 0))
                p14 = geom.add_point((circle_rad, ymin, 0))
                p15 = geom.add_point((xmax, circle_rad, 0))
                p16 = geom.add_point((xmax, -circle_rad, 0))

                # ********************  Adding geometry lines  ********************
                # inside circle point
                p17 = geom.add_point((0, 0, 0))

                # Bounding box
                l1 = geom.add_line(p1, p9)
                l2 = geom.add_line(p9, p10)
                l3 = geom.add_line(p10,p4)
                l4 = geom.add_line(p4, p11)
                l5 = geom.add_line(p11,p14)
                l6 = geom.add_line(p14,p3)
                l7 = geom.add_line(p3, p16)
                l8 = geom.add_line(p16,p15)
                l9 = geom.add_line(p15,p2)
                l10 = geom.add_line(p2, p13)
                l11 = geom.add_line(p13, p12)
                l12 = geom.add_line(p12, p1)

                # Cross lines
                l13 = geom.add_line(p9, p6)
                l14 = geom.add_line(p10, p8)
                l15 = geom.add_line(p11, p8)
                l16 = geom.add_line(p14, p7)
                l17 = geom.add_line(p16, p7)
                l18 = geom.add_line(p15, p5)
                l19 = geom.add_line(p13, p5)
                l20 = geom.add_line(p12, p6)

                # Outer circle
                l21 = geom.add_circle_arc(p6, p17, p8)
                l22 = geom.add_circle_arc(p8, p17, p7)
                l23 = geom.add_circle_arc(p7, p17, p5)
                l24 = geom.add_circle_arc(p5, p17, p6)

                # Outer domain
                curve_loop_1 = geom.add_curve_loop([l1, l13, -l20, l12])
                
                curve_loop_2 = geom.add_curve_loop([l11, l20, -l24, -l19])
                curve_loop_3 = geom.add_curve_loop([l10, l19, -l18, l9])
                curve_loop_4 = geom.add_curve_loop([l8, l18, -l23, -l17])
                curve_loop_5 = geom.add_curve_loop([l7, l17, -l16, l6])
                curve_loop_6 = geom.add_curve_loop([l5, l16, -l22, -l15])
                curve_loop_7 = geom.add_curve_loop([l4, l15, -l14, l3])
                curve_loop_8 = geom.add_curve_loop([l2, l14, -l21, -l13])

                curve_loop_9 = geom.add_curve_loop([l24, l21, l22, l23])

                # ********************  Setting mesh points  ********************
                # Outer domain
                # x-direction 
                geom.set_transfinite_curve(l1, Ny_outer, 'Progression', 1.0)
                geom.set_transfinite_curve(l20, Ny_outer, 'Progression', 1.0)
                geom.set_transfinite_curve(l19, Ny_outer, 'Progression', 1.0)
                geom.set_transfinite_curve(l9, Ny_outer, 'Progression', 1.0)

                geom.set_transfinite_curve(l3, Ny_outer, 'Progression', 1.0)
                geom.set_transfinite_curve(l15, Ny_outer, 'Progression', 1.0)
                geom.set_transfinite_curve(l16, Ny_outer, 'Progression', 1.0)
                geom.set_transfinite_curve(l7, Ny_outer, 'Progression', 1.0)
                
                # y-direction
                geom.set_transfinite_curve(l12, Ny_outer_left, 'Progression', 1.0)
                geom.set_transfinite_curve(l13, Ny_outer_left, 'Progression', 1.0)
                geom.set_transfinite_curve(l14, Ny_outer_left, 'Progression', 1.0)
                geom.set_transfinite_curve(l4, Ny_outer_left, 'Progression', 1.0)

                geom.set_transfinite_curve(l10, Ny_outer_right, 'Progression', 1.0)
                geom.set_transfinite_curve(l18, Ny_outer_right, 'Progression', 1.0)
                geom.set_transfinite_curve(l17, Ny_outer_right, 'Progression', 1.0)
                geom.set_transfinite_curve(l6, Ny_outer_right, 'Progression', 1.0)

                # Cross domain
                # horizontal
                geom.set_transfinite_curve(l2, Ny_inner//2 if wake_refined else Ny_inner, 'Progression', 1.0)
                geom.set_transfinite_curve(l21, Ny_inner//2 if wake_refined else Ny_inner, 'Progression', 1.0)

                geom.set_transfinite_curve(l8, Ny_inner, 'Progression', 1.0)
                geom.set_transfinite_curve(l23, Ny_inner, 'Progression', 1.0)
                
                # vertical
                geom.set_transfinite_curve(l11, Nx_inner, 'Progression', 1.0)
                geom.set_transfinite_curve(l24, Nx_inner, 'Progression', 1.0)

                geom.set_transfinite_curve(l5, Nx_inner, 'Progression', 1.0)
                geom.set_transfinite_curve(l22, Nx_inner, 'Progression', 1.0)

                # 
                plane_surface_1 = geom.add_plane_surface(curve_loop_1)
                plane_surface_2 = geom.add_plane_surface(curve_loop_2)
                plane_surface_3 = geom.add_plane_surface(curve_loop_3)
                plane_surface_4 = geom.add_plane_surface(curve_loop_4)
                plane_surface_5 = geom.add_plane_surface(curve_loop_5)
                plane_surface_6 = geom.add_plane_surface(curve_loop_6)
                plane_surface_7 = geom.add_plane_surface(curve_loop_7)
                plane_surface_8 = geom.add_plane_surface(curve_loop_8)

                plane_surface_9 = geom.add_plane_surface(curve_loop_9, holes=[poly.curve_loop])

                geom.set_transfinite_surface(plane_surface_1, 'Left', [])
                geom.set_transfinite_surface(plane_surface_2, 'Left', [])
                geom.set_transfinite_surface(plane_surface_3, 'Left', [])
                geom.set_transfinite_surface(plane_surface_4, 'Left', [])
                geom.set_transfinite_surface(plane_surface_5, 'Left', [])
                geom.set_transfinite_surface(plane_surface_6, 'Left', [])
                geom.set_transfinite_surface(plane_surface_7, 'Left', [])
                geom.set_transfinite_surface(plane_surface_8, 'Left', [])

                # ********************  Quad mesh surface ********************
                geom.set_recombined_surfaces([
                    plane_surface_1, plane_surface_2, plane_surface_3,
                    plane_surface_4, plane_surface_5, plane_surface_6,
                    plane_surface_7, plane_surface_8, 
                ])
                
                if extruded:
                    # ********************  Extruding ********************
                    # Outer domain
                    extruded_1 = geom.extrude(
                        plane_surface_1,
                        (0,0,1),
                        num_layers=1,
                        heights=[1],
                        recombine=True,
                    )

                    extruded_2 = geom.extrude(
                        plane_surface_2,
                        (0,0,1),
                        num_layers=1,
                        heights=[1],
                        recombine=True,
                    )

                    extruded_3 = geom.extrude(
                        plane_surface_3,
                        (0,0,1),
                        num_layers=1,
                        heights=[1],
                        recombine=True,
                    )

                    extruded_4 = geom.extrude(
                        plane_surface_4,
                        (0,0,1),
                        num_layers=1,
                        heights=[1],
                        recombine=True,
                    )

                    extruded_5 = geom.extrude(
                        plane_surface_5,
                        (0,0,1),
                        num_layers=1,
                        heights=[1],
                        recombine=True,
                    )

                    extruded_6 = geom.extrude(
                        plane_surface_6,
                        (0,0,1),
                        num_layers=1,
                        heights=[1],
                        recombine=True,
                    )

                    extruded_7 = geom.extrude(
                        plane_surface_7,
                        (0,0,1),
                        num_layers=1,
                        heights=[1],
                        recombine=True,
                    )

                    extruded_8 = geom.extrude(
                        plane_surface_8,
                        (0,0,1),
                        num_layers=1,
                        heights=[1],
                        recombine=True,
                    )

                    # Inner domain
                    extruded_9 = geom.extrude(
                        plane_surface_9,
                        (0,0,1),
                        num_layers=1,
                        heights=[1],
                        recombine=True,
                    )
		
                    # ******************** Setting physical groups ********************
                    # surfaces
                    geom.add_physical([
                        plane_surface_1, plane_surface_2, plane_surface_3,
                        plane_surface_4, plane_surface_5, plane_surface_6,
                        plane_surface_7, plane_surface_8, plane_surface_9
                    ], label='back')
                
                    geom.add_physical([
                        extruded_1[0], extruded_2[0], extruded_3[0],
                        extruded_4[0], extruded_5[0], extruded_6[0],
                        extruded_7[0], extruded_8[0], extruded_9[0],
                    ], label='front')

                    geom.add_physical([
                        extruded_1[2][0], extruded_7[2][3], extruded_8[2][0], 
                    ], label='in')

                    geom.add_physical([
                        extruded_3[2][3], extruded_4[2][0], extruded_5[2][0],
                    ], label='out')

                    geom.add_physical([
                        extruded_1[2][3], extruded_2[2][0], extruded_3[2][0],
                    ], label='sym1')

                    geom.add_physical([
                        extruded_5[2][3], extruded_6[2][0], extruded_7[2][0],
                    ], label='sym2')

                    geom.add_physical(extruded_9[2][4:], label='obstacle')

                    # volume
                    geom.add_physical([
                        extruded_1[1], extruded_2[1], extruded_3[1], 
                        extruded_4[1], extruded_5[1], extruded_6[1], 
                        extruded_7[1], extruded_8[1], extruded_9[1], 
                    ], label='fluid')
                else:
                    geom.add_physical([
                        plane_surface_1, plane_surface_2, plane_surface_3,
                        plane_surface_4, plane_surface_5, plane_surface_6,
                        plane_surface_7, plane_surface_8, plane_surface_9
                    ], label='fluid')
                    geom.add_physical([l1,l2,l3], label='in')
                    geom.add_physical([l9,l8,l7], label='out')
                    geom.add_physical([l4,l5,l6,l12,l11,l10], label='topbottom')
                    #import pdb; pdb.set_trace()
                    geom.add_physical([*poly.curve_loop.curves], label='obstacle')
                geom.synchronize()
                
                # geom.save_geometry('test.geo_unrolled')

            # Generate mesh and write in medit format
            try:
                mesh = geom.generate_mesh()
                filename = f'{self.name}_{self.index}.{mesh_format}'
                pygmsh.write(filename)
            except AssertionError:
                print('\n'+'!!!!! Meshing failed !!!!!')
                return False, 0

        # Compute data from mesh
        n_tri = len(mesh.cells_dict.get('triangle',[]))
        n_quad = len(mesh.cells_dict.get('quad',[]))
        n_cells = n_tri + n_quad

        return filename, n_cells

    ### ************************************************
    ### Get shape bounding box
    def compute_bounding_box(self):
        x_max, y_max = np.amax(self.control_pts,axis=0)
        x_min, y_min = np.amin(self.control_pts,axis=0)

        dx = x_max - x_min
        dy = y_max - y_min

        return dx, dy

    ### ************************************************
    ### Modify shape given a deformation field
    def modify_shape_from_field(self, deformation, *args, **kwargs):
        # Handle optional argument
        replace  = kwargs.get('replace',  False)
        pts_list = kwargs.get('pts_list', [])

        # Check inputs
        if (pts_list == []):
            if (len(deformation[:,0]) != self.n_control_pts):
                print('Input deformation field does not have right length')
                quit()
        if (len(deformation[0,:]) not in [2, 3]):
            print('Input deformation field does not have right width')
            quit()
        if (pts_list != []):
            if (len(pts_list) != len(deformation)):
                print('Lengths of pts_list and deformation are different')
                quit()

        # If shape is to be replaced entirely
        if (    replace):
            # If a list of points is provided
            if (pts_list != []):
                for i in range(len(pts_list)):
                    self.control_pts[pts_list[i],0] = deformation[i,0]
                    self.control_pts[pts_list[i],1] = deformation[i,1]
                    self.edgy[pts_list[i]]          = deformation[i,2]
            # Otherwise
            if (pts_list == []):
                self.control_pts[:,0] = deformation[:,0]
                self.control_pts[:,1] = deformation[:,1]
                self.edgy[:]          = deformation[:,2]
        # Otherwise
        if (not replace):
            # If a list of points to deform is provided
            if (pts_list != []):
                for i in range(len(pts_list)):
                    self.control_pts[pts_list[i],0] += deformation[i,0]
                    self.control_pts[pts_list[i],1] += deformation[i,1]
                    self.edgy[pts_list[i]]          += deformation[i,2]
            # Otherwise
            if (pts_list == []):
                self.control_pts[:,0] += deformation[:,0]
                self.control_pts[:,1] += deformation[:,1]
                self.edgy[:]          += deformation[:,2]

        # Increment shape index
        self.index += 1

    ### ************************************************
    ### Compute shape area
    def compute_area(self):
        self.area = 0.0

        # Use Green theorem to compute area
        for i in range(0,len(self.curve_pts)-1):
            x1 = self.curve_pts[i-1,0]
            x2 = self.curve_pts[i,  0]
            y1 = self.curve_pts[i-1,1]
            y2 = self.curve_pts[i,  1]

            self.area += 2.0*(y1+y2)*(x2-x1)

    ### ************************************************
    ### Compute shape dimensions
    def compute_dimensions(self):
        self.size_y = 0.0
        self.size_x = 0.0
        xmin = 1.0e20
        xmax =-1.0e20
        ymin = 1.0e20
        ymax =-1.0e20

        for i in range(len(self.curve_pts)):
            xmin = min(xmin, self.curve_pts[i,0])
            xmax = max(xmax, self.curve_pts[i,0])
            ymin = min(ymin, self.curve_pts[i,1])
            ymax = max(ymax, self.curve_pts[i,1])

        self.size_x = xmax - xmin
        self.size_y = ymax - ymin

### End of class Shape
### ************************************************

### ************************************************
### Compute distance between two points
def compute_distance(p1, p2):

    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

### ************************************************
### Generate n_pts random points in the unit square
def generate_random_pts(n_pts):

    return np.random.rand(n_pts,2)

### ************************************************
### Generate cylinder points
def generate_cylinder_pts(n_pts):
    if (n_pts < 4):
        print('Not enough points to generate cylinder')
        exit()

    pts = np.zeros([n_pts, 2])
    ang = 2.0*math.pi/n_pts
    for i in range(0,n_pts):
        pts[i,:] = [math.cos(float(i)*ang),math.sin(float(i)*ang)]

    return pts

### ************************************************
### Compute minimal distance between successive pts in array
def compute_min_distance(pts):
    dist_min = 1.0e20
    for i in range(len(pts)-1):
        p1       = pts[i  ,:]
        p2       = pts[i+1,:]
        dist     = compute_distance(p1,p2)
        dist_min = min(dist_min,dist)

    return dist_min

### ************************************************
### Remove duplicate points in input coordinates array
### WARNING : this routine is highly sub-optimal
def remove_duplicate_pts(pts):
    to_remove = []

    for i in range(len(pts)):
        for j in range(len(pts)):
            # Check that i and j are not identical
            if (i == j):
                continue

            # Check that i and j are not removed points
            if (i in to_remove) or (j in to_remove):
                continue

            # Compute distance between points
            pi = pts[i,:]
            pj = pts[j,:]
            dist = compute_distance(pi,pj)

            # Tag the point to be removed
            if (dist < 1.0e-8):
                to_remove.append(j)

    # Sort elements to remove in reverse order
    to_remove.sort(reverse=True)

    # Remove elements from pts
    for pt in to_remove:
        pts = np.delete(pts, pt, 0)

    return pts

### ************************************************
### Counter Clock-Wise sort
###  - Take a cloud of points and compute its geometric center
###  - Translate points to have their geometric center at origin
###  - Compute the angle from origin for each point
###  - Sort angles by ascending order
def ccw_sort(pts, rad, edg):
    geometric_center = np.mean(pts,axis=0)
    translated_pts   = pts - geometric_center
    angles           = np.arctan2(translated_pts[:,1], translated_pts[:,0])
    x                = angles.argsort()
    pts2             = np.array(pts)
    rad2             = np.array(rad)
    edg2             = np.array(edg)

    return pts2[x,:], rad2[x], edg2[x]

### ************************************************
### Compute Bernstein polynomial value
def compute_bernstein(n,k,t):
    k_choose_n = scipy.special.binom(n,k)

    return k_choose_n * (t**k) * ((1.0-t)**(n-k))

### ************************************************
### Sample Bezier curves given set of control points
### and the number of sampling points
### Bezier curves are parameterized with t in [0,1]
### and are defined with n control points P_i :
### B(t) = sum_{i=0,n} B_i^n(t) * P_i
def sample_bezier_curve(control_pts, n_sampling_pts):
    n_control_pts = len(control_pts)
    t             = np.linspace(0, 1, n_sampling_pts)
    curve         = np.zeros((n_sampling_pts, 2))

    for i in range(n_control_pts):
        curve += np.outer(compute_bernstein(n_control_pts-1, i, t),
                          control_pts[i])

    return curve

### ************************************************
### Generate Bezier curve between two pts
def generate_bezier_curve(p1,       p2,
                          delta1,   delta2,
                          delta_b1, delta_b2,
                          radius1,  radius2,
                          edgy1,    edgy2,
                          n_sampling_pts):

    # Lambda function to wrap angles
    #wrap = lambda angle: (angle >= 0.0)*angle + (angle < 0.0)*(angle+2*np.pi)

    # Sample the curve if necessary
    if (n_sampling_pts != 0):
        # Create array of control pts for cubic Bezier curve
        # First and last points are given, while the two intermediate
        # points are computed from edge points, angles and radius
        control_pts      = np.zeros((4,2))
        control_pts[0,:] = p1[:]
        control_pts[3,:] = p2[:]

        # Compute baseline intermediate control pts ctrl_p1 and ctrl_p2
        ctrl_p1_base = radius1*delta1
        ctrl_p2_base =-radius2*delta2

        ctrl_p1_edgy = radius1*delta_b1
        ctrl_p2_edgy = radius2*delta_b2

        control_pts[1,:] = p1 + edgy1*ctrl_p1_base + (1.0-edgy1)*ctrl_p1_edgy
        control_pts[2,:] = p2 + edgy2*ctrl_p2_base + (1.0-edgy2)*ctrl_p2_edgy

        # Compute points on the Bezier curve
        curve = sample_bezier_curve(control_pts, n_sampling_pts)

    # Else return just a straight line
    else:
        curve = p1
        curve = np.vstack([curve,p2])

    return curve

### Crop white background from image
def trim_white(filename):

    im   = PIL.Image.open(filename)
    bg   = PIL.Image.new(im.mode, im.size, (255,255,255))
    diff = PIL.ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    cp   = im.crop(bbox)
    cp.save(filename)
