import trimesh
import seissolxdmf
import numpy as np
import argparse
from scipy import spatial
import matplotlib.pylab as plt
from scipy import ndimage as nd

import scipy.interpolate as sp_int



class seissolxdmfExtended(seissolxdmf.seissolxdmf):
    def compute_centers(self):
        xyz = self.ReadGeometry()
        connect = self.ReadConnect()
        return (xyz[connect[:, 0]] + xyz[connect[:, 1]] + xyz[connect[:, 2]]) / 3.0


def get_fault_trace(args):
    # generated with
    # python ~/SeisSol/Meshing/vizualizeBoundaryConditions/vizualizeBoundaryConditions.py Ridgecrest_NewModel1_f200_topo1000_noRef_xml_UBC.xdmf 68
    if args.event[0] == "foreshock":
        fn = "mesh_files/Ridgecrest_NewModel1_f200_topo1000_noRef_xml_UBC_bc67.xdmf"
    else:
        fn = "mesh_files/Ridgecrest_NewModel1_f200_topo1000_noRef_xml_UBC_bc68.xdmf"
    sx = seissolxdmf.seissolxdmf(fn)
    geom = sx.ReadGeometry()
    connect = sx.ReadConnect()
    mesh = trimesh.Trimesh(geom, connect)
    # list vertex of the face boundary
    unique_edges = mesh.edges[
        trimesh.grouping.group_rows(mesh.edges_sorted, require_count=2)
    ]
    unique_edges = unique_edges[:, :, 1]
    ids_external_nodes = np.unique(unique_edges.flatten())

    nodes = mesh.vertices[ids_external_nodes, :]
    nodes = nodes[nodes[:, 2] > 0]
    nodes = nodes[nodes[:, 1].argsort()]

    # Compute strike vector to filter boundaries of near-vertical edges
    grad = np.gradient(nodes, axis=0)
    grad = grad / np.linalg.norm(grad, axis=1)[:, None]
    ids_top_trace = np.where(np.abs(grad[:, 2]) < 0.8)[0]
    nodes = nodes[ids_top_trace]

    return nodes


def compute_strike_and_points_across_fault(trace_nodes, dx):
    grad = np.gradient(trace_nodes, axis=0)
    strike = grad[:, 0:2] / np.linalg.norm(grad[:, 0:2], axis=1)[:, None]
    fault_normal_vector = np.zeros_like(strike)
    fault_normal_vector[:, 0] = -strike[:, 1]
    fault_normal_vector[:, 1] = -strike[:, 0]
    pointsPlus = trace_nodes[:, 0:2] + 0.5 * dx * fault_normal_vector
    pointsMinus = trace_nodes[:, 0:2] - 0.5 * dx * fault_normal_vector
    points = np.vstack((pointsPlus, pointsMinus))
    return strike, points


def compute_distance_from_epicenter(nodes, xhypo):
    x = nodes[:, 0] - xhypo[0]
    y = nodes[:, 1] - xhypo[1]
    return -np.sign(y) * np.sqrt(x**2 + y**2) / 1e3


def compute_distance_from_epicenter_az(nodes, xhypo, azimuth):
    az = np.radians(azimuth)
    x = nodes[:, 0] - xhypo[0]
    y = nodes[:, 1] - xhypo[1]
    return (np.cos(az) * x + np.sin(az) * y) / 1e3


def prepare_plot(fig, ax):
    plt.xlabel("distance from epicenter (km)")
    plt.ylabel("fault parallel-offset (m)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def compareOffset(faultFile, event = 'foreshock'):
    class Args:
        def __init__(self, fault, event=None, downsample=1, half_width=400.0):
            self.fault = fault
            self.event = event
            self.downsample = downsample
            self.half_width = half_width

    # Example of creating an instance of Args
    args = Args(
        fault=[faultFile],
        event=[event],
        downsample=[1],
        half_width=[400.0]
    )

    if args.event[0] == "foreshock":
        azimuth = 38
        xc = -3583.91
        yc = 9291.69
    else:
        azimuth = 330.0
        xc = -3583.91
        yc = 9291.69

    xc += 450000.0
    yc += 3950000.0

    trace_nodes = get_fault_trace(args)[:: args.downsample[0]]
    distance_from_epicenter = compute_distance_from_epicenter_az(
        trace_nodes, [xc, yc], azimuth
    )

    sx = seissolxdmfExtended(args.fault[0])
    fault_centers = sx.compute_centers()
    ndt = sx.ReadNdt()
    if args.event[0] == "foreshock":
        Sls = sx.ReadData("Sls", 3)
        ASl = sx.ReadData("ASl", 3)
    else:
        Sls = sx.ReadData("Sls", ndt - 1)
        ASl = sx.ReadData("ASl", ndt - 1)

    tree = spatial.KDTree(fault_centers)
    dist, idsf = tree.query(trace_nodes)

    # Remove point(s) picked from the wrong segment
    if args.event[0] == "foreshock":
        intersection_node = np.where(Sls[idsf] > 0)[0]
    else:
        intersection_node = np.where(Sls[idsf] < 0)[0]
    print(f"manually removing {intersection_node.shape} nodes (intersection)")
    idsf = np.delete(idsf, intersection_node, 0)
    distance_from_epicenter_no_inters = np.delete(
        distance_from_epicenter, intersection_node, 0
    )

    # fig = plt.figure(figsize=(14, 8))
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    prepare_plot(fig, ax)
    plt.plot(distance_from_epicenter_no_inters, np.abs(Sls[idsf]), "r")

    if args.event[0] == "foreshock":
        loaded = np.load("ref/antoine_fore.npz")
        dist = loaded['dist1'][loaded['dist1']<-1.5][::-1]
        dist = np.append(dist, loaded['dist2'][::-1])
        offset = loaded['offest1'][loaded['dist1']<-1.5][::-1]
        offset = np.append(offset, loaded['offest2'][::-1])
    else:
        loaded = np.load("ref/antoine_main.npz")
        dist = loaded['dist'][::-1]
        offset = loaded['offest'][::-1]

    interpolatorOffset = sp_int.interp1d(dist, offset)
    offset_interpolated = interpolatorOffset(distance_from_epicenter_no_inters)

    plt.plot(dist, offset,'--')
    plt.plot(distance_from_epicenter_no_inters, offset_interpolated)

    plt.show()

    return offset_interpolated - np.abs(Sls[idsf])