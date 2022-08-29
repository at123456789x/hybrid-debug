# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: wang.q
@Contact: wang.q@mail.hnust.edu.cn
@Software: PyCharm
@File: Hybird_test.py
@Time: 8/29/22 10:58 AM
@Function:
"""

import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import trange
import os
import firedrake
import icepack
import geojson
import rasterio
import icepack.plot
import icepack.models.hybrid
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
from firedrake import grad, max_value
from rasterio.plot import show
from icepack.constants import ice_density, gravity
from rasterio.fill import fillnodata

# Setting ==============================================================================================================
# set env ---
test_folder = r'/mnt/hgfs/F/Project_MBG/test'           # test folder
mbg_input = test_folder + '/MBG'
glc_input = mbg_input
iceflow_input = test_folder + r'/Iceflow'

# Load and prep general data ===========================================================================================
# read the sampled mean ostrem curve
ostrem = pd.read_csv(mbg_input + r'/ostrem.csv')
rgi_pt = pd.read_csv(mbg_input + r'/Glc_Info_C.csv')

# load glacier info
rgi_ids = rgi_pt['RGIId']

# debug ================================================================================================================
# read shapes & structure mesh
cur_glc = rgi_pt.loc[0, :]

outline_filename = iceflow_input + r'/shapefile.json'
with open(outline_filename, 'r') as outline_file:
    outline = geojson.load(outline_file)

boundary_ids = [i + 1 for i in list(range(len(outline['features'][0]['geometry']['coordinates'])))]
boundary_ids.remove(1)
# boundary_ids = []

geometry = icepack.meshing.collection_to_geo(outline)
with open(iceflow_input + r'/mesh.geo', 'w') as geo_file:
    geo_file.write(geometry.get_code())

os.system(r'gmsh -2 -format msh2 -v 2 -o %s %s' % ((iceflow_input + r'/mesh.msh'),
                                                   (iceflow_input + r'/mesh.geo')))

# mesh & func space
mesh2d = firedrake.Mesh(iceflow_input + r'/mesh.msh')

# fig, axes = icepack.plot.subplots()
# axes.set_xlabel('meters')
# axes.set_title(cur_glc['RGIId'])
#
# kwargs = {
#     'interior_kw': {'linewidth': .25},
#     'boundary_kw': {'linewidth': 2}
# }
# icepack.plot.triplot(mesh2d, axes=axes, **kwargs)
# axes.legend()

mesh3d = firedrake.ExtrudedMesh(mesh2d, layers=1)

Q2d = firedrake.FunctionSpace(mesh2d, family='CG', degree=2)
V2d = firedrake.VectorFunctionSpace(mesh2d, dim=2, family='CG', degree=2, vfamily='GL', vdegree=2)

Q3d = firedrake.FunctionSpace(mesh3d, family='CG', degree=2, vfamily='R', vdegree=0)
V3d = firedrake.VectorFunctionSpace(mesh3d, dim=2, family='CG', degree=2, vfamily='GL', vdegree=0)
V3d_interpolate = firedrake.VectorFunctionSpace(mesh3d, dim=2, family='CG', degree=2, vfamily='GL', vdegree=2)

# import dataset
thickness_filename = iceflow_input + r'/thickness.tif'
with rasterio.open(thickness_filename, 'r') as thickness:
    h0 = icepack.interpolate(thickness, Q2d)
    h3 = firedrake.Function(Q3d)
    h3.dat.data[:] = h0.dat.data[:]

velocity_filename = {'Vx': iceflow_input + r'/vx.tif',
                     'Vy': iceflow_input + r'/vy.tif'}

with rasterio.open(velocity_filename['Vx'], 'r') as vx:
    with rasterio.open(velocity_filename['Vy'], 'r') as vy:
        x = vx.read()
        y = vy.read()
        u0 = icepack.interpolate((vx, vy), V2d)
        # u0.dat.data[:] = u0.dat.data[:] * 365
        u3 = firedrake.Function(V3d)
        u3.dat.data[:] = u0.dat.data[:]
        u0 = firedrake.interpolate(u3, V3d_interpolate)

surface_filename = iceflow_input + r'/dem.tif'
with rasterio.open(surface_filename, 'r') as surface:
    surface_arr = surface.read()
    surface_meta = surface.meta.copy()
    surface_arr[surface_arr == surface_meta['nodata']] = np.nan
    s0 = icepack.interpolate(surface, Q2d)
    s3 = firedrake.Function(Q3d)
    s3.dat.data[:] = s0.dat.data[:]

# structure field & coord transfer
bed = icepack.interpolate(s0 - h0, Q2d)
h0 = h3
s0 = s3

b3 = firedrake.Function(Q3d)
b3.dat.data[:] = bed.dat.data[:]
bed = b3

h3, s3, b3 = None, None, None

print('Set Hybrid parameters has done. ')

# read model parameter ---
mb_gradient_ssp = pd.read_csv(mbg_input + '/mb_gradient.csv', index_col=0)
debris_ssp = pd.read_csv(mbg_input + '/debris_ssp.csv', index_col=0)
df_abl_max_ssp = pd.read_csv(mbg_input + '/abl_max_ssp.csv', index_col=0)
df_acc_max_ssp = pd.read_csv(mbg_input + '/acc_max_ssp.csv', index_col=0)
df_rgi_tas_ssp = pd.read_csv(mbg_input + '/rgi_tas_ssp.csv', index_col=0)

debris_arr = np.loadtxt(mbg_input + r'/debris_arr.txt')
ddf_cls = np.loadtxt(mbg_input + r'/ddf_cls.txt')
band_arr = np.loadtxt(mbg_input + r'/band_arr.txt')
dem_ras_cls = np.loadtxt(mbg_input + r'/dem_ras_cls.txt')

with rio.open(glc_input + r'/dem.tif') as dem_ras:
    dem_meta = dem_ras.meta.copy()

print('Import model parameters has done. ')

# loop scenarios =======================================================================================================
# for i in mb_gradient_ssp.index:
# debug with the ssp1
scene = 0
scene = mb_gradient_ssp.index[scene]

# Hybrid model ---
# Bed friction
def friction(**kwargs):
    u = kwargs['velocity']
    h = kwargs['thickness']
    s = kwargs['surface']
    C = kwargs['friction']

    return icepack.models.hybrid.bed_friction(
        velocity=u,
        friction=C
    )

# Model structure
model = icepack.models.HybridModel(friction=friction)
opts = {'dirichlet_ids': boundary_ids}
# opts = {"dirichlet_ids": boundary_ids,
#         "diagnostic_solver_type": "petsc",
#         "diagnostic_solver_parameters": {
#             "ksp_type": "preonly",
#             "pc_type": "lu",
#             "pc_factor_mat_solver_type": "mumps",
#         },
#         }
solver = icepack.solvers.FlowSolver(model, **opts)

h = h0.copy(deepcopy=True)
s = s0.copy(deepcopy=True)
u = u0.copy(deepcopy=True)

fig, axes = icepack.plot.subplots(2, 5)
colors = icepack.plot.tripcolor(h, axes=axes[0][0], vmax=200)
fig.colorbar(colors, ax=axes[0][0], label='meters', fraction=0.012, pad=0.04)
axes[0][0].set_title('Initial')
axes[0][0].axis('off')

t_ser = range(2015, 2100)

# loop iceflow ---
for t in trange(85, desc='Loop %s ...' % scene):

    t = str(t_ser[t])

    # calculate ice accumulation ---------------------------------------------------------------------------------------
    elr = cur_glc[t]

    red_arr = debris_arr * 100 + debris_ssp.loc[scene][0]
    red_arr[red_arr < 0] = 0
    red_arr = np.interp(red_arr, ostrem['pred.x'], ostrem['value'])
    red_arr[(ddf_cls == 1) & ~np.isnan(band_arr)] = 0

    mb_arr = (dem_ras_cls - 1) * mb_gradient_ssp.loc[scene][0] + df_abl_max_ssp[t][scene] + red_arr
    mb_arr[mb_arr > df_acc_max_ssp[t][scene]] = df_acc_max_ssp[t][scene]
    mb_arr[np.isnan(mb_arr)] = 0

    tas = df_rgi_tas_ssp[t][scene]
    elr_span = (surface_arr - cur_glc['min_Elev']) / 1000
    elr_delta = elr * elr_span
    tas_arr = np.full_like(surface_arr, tas)
    tas_arr[np.isnan(surface_arr)] = np.nan
    tas_arr = tas_arr + elr_delta + 273.15

    tas_utm = iceflow_input + r'/tas_utm.tif'
    with rio.open(tas_utm, 'w', **surface_meta) as ds:
        ds.write(tas_arr)

    def mbg2iceflow(fn, ras_arr):
        meta = dem_meta
        meta.update({'nodata': 0})

        with rio.open(fn, 'w', **dem_meta) as ds:
            ds.write(ras_arr, indexes=1)

        with rasterio.open(fn) as src:
            utm_zone = 32600 + int(int((src.bounds.right + src.bounds.left) / 2) / 6) + 31
            dst_crs = 'EPSG:%d' % utm_zone
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(fn.replace('.tif', '_utm.tif'), 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)

        return fn.replace('.tif', '_utm.tif')

    mb_utm = mbg2iceflow(iceflow_input + r'/mb.tif', mb_arr)

    with rio.open(mb_utm, 'r') as in_ds:
        with rio.open(thickness_filename, 'r') as ref_ds:

            in_ds_meta = in_ds.meta.copy()
            ref_ds_meta = ref_ds.meta.copy()

            in_ds_arr = in_ds.read(1)
            ref_ds_arr = ref_ds.read(1)

            out_ds_arr = np.zeros_like(ref_ds_arr)

            in_x_ser = [in_ds_meta['transform'][2] + (0.5 + i) * in_ds_meta['transform'][0] for i in range(in_ds_arr.shape[1])]
            in_y_ser = [in_ds_meta['transform'][5] + (0.5 + i) * in_ds_meta['transform'][4] for i in range(in_ds_arr.shape[0])]

            ref_x_ser = [ref_ds_meta['transform'][2] + (0.5 + i) * ref_ds_meta['transform'][0] for i in range(ref_ds_arr.shape[1])]
            ref_y_ser = [ref_ds_meta['transform'][5] + (0.5 + i) * ref_ds_meta['transform'][4] for i in range(ref_ds_arr.shape[0])]

            for y in range(ref_ds_arr.shape[0]):
                for x in range(ref_ds_arr.shape[1]):

                    thick_val = ref_ds_arr[y, x]

                    if thick_val != 0:
                        coord_x = ref_x_ser[x]
                        coord_y = ref_y_ser[y]

                        distance_x_ls = [abs(in_x - coord_x) for in_x in in_x_ser]
                        distance_y_ls = [abs(in_y - coord_y) for in_y in in_y_ser]

                        in_x_ind = distance_x_ls.index(np.min(distance_x_ls))
                        in_y_ind = distance_y_ls.index(np.min(distance_y_ls))

                        out_ds_arr[y, x] = in_ds_arr[in_y_ind, in_x_ind]

            fillnodata(out_ds_arr, ~((out_ds_arr == 0) & (ref_ds_arr != 0)))

            mb_utm = mb_utm.replace('_utm', '_rectangle')

            with rio.open(mb_utm, 'w', **ref_ds_meta) as out_ds:
                out_ds.write(out_ds_arr, indexes=1)

    # Hybrid model paras -----------------------------------------------------------------------------------------------
    # solve ice hardness
    with rio.open(tas_utm, 'r') as ds_tas:
        A = firedrake.interpolate(icepack.rate_factor(icepack.interpolate(ds_tas, Q2d)), Q2d)
        A3 = firedrake.Function(Q3d)
        A3.dat.data[:] = A.dat.data[:]
        A = A3

    # accumulation
    with rio.open(mb_utm, 'r') as ds_tas:
        a = icepack.interpolate(ds_tas, Q2d)
        a3 = firedrake.Function(Q3d)
        a3.dat.data[:] = a.dat.data[:]
        a = a3

    # interpolate value to function space
    thickness = firedrake.Function(Q2d)
    surface = firedrake.Function(Q2d)
    thickness.dat.data[:] = h.dat.data[:]
    surface.dat.data[:] = s.dat.data[:]

    # calc friction
    drive = firedrake.interpolate(-ice_density * gravity * icepack.interpolate(thickness, Q2d) *
                                  grad(icepack.interpolate(surface, Q2d)), V2d)
    drive_stress = firedrake.Function(Q3d)
    drive_stress.dat.data[:] = [np.sqrt(i ** 2 + j ** 2) for i, j in drive.dat.data[:]]

    C = drive_stress.copy(deepcopy=True)

    # fig, axes = icepack.plot.subplots()
    # colors = icepack.plot.tripcolor(C, axes=axes)
    # fig.colorbar(colors, label='mPa')
    # axes.axis('off')

    A3, a3, drive = None, None, None

    # solve velocity
    u = solver.diagnostic_solve(
        velocity=u,
        thickness=h,
        surface=s,
        fluidity=A,
        friction=C
    )

    h.interpolate(max_value(h, 0))
    s = icepack.compute_surface(thickness=h, bed=bed)

    # plot the result per decade
    if (int(t) % 10) == 0:
        plot_loc_ls = ['2020', '2030', '2040', '2050', '2060', '2070', '2080', '2090']
        plot_x_ind = [1, 2, 3, 4, 0, 1, 2, 3]
        plot_y_ind = [0, 0, 0, 0, 1, 1, 1, 1]
        # fig, axes = icepack.plot.subplots()
        colors = icepack.plot.tripcolor(h, axes=axes[plot_y_ind[plot_loc_ls.index(t)]][plot_x_ind[plot_loc_ls.index(t)]], vmin=0, vmax=200)
        fig.colorbar(colors, ax=axes[plot_y_ind[plot_loc_ls.index(t)]][plot_x_ind[plot_loc_ls.index(t)]], label='meters', fraction=0.012, pad=0.04)
        axes[plot_y_ind[plot_loc_ls.index(t)]][plot_x_ind[plot_loc_ls.index(t)]].set_title(t)
        axes[plot_y_ind[plot_loc_ls.index(t)]][plot_x_ind[plot_loc_ls.index(t)]].axis('off')

    # fig, axes = icepack.plot.subplots()
    # colors = icepack.plot.tripcolor(u, axes=axes)
    # fig.colorbar(colors, ax=axes, label='meters yr-1')
    # axes.axis('off')

    # solve ice thickness
    δt = 1

    h = solver.prognostic_solve(
        δt,
        thickness=h,
        velocity=u,
        accumulation=a,
        # thickness_inflow=h
    )

    # h.interpolate(max_value(h, 0))
    s = icepack.compute_surface(thickness=h, bed=bed)

h.interpolate(max_value(h, 0))
s = icepack.compute_surface(thickness=h, bed=bed)

colors = icepack.plot.tripcolor(h, axes=axes[-1][-1], vmin=0, vmax=200)
fig.colorbar(colors, ax=axes[-1][-1], label='meters', fraction=0.012, pad=0.04)
axes[-1][-1].set_title('Final')
axes[-1][-1].axis('off')
