import csv
import math
import matplotlib.pyplot as plt
import numpy as np

vicon_frame_time_s = 0.01

with open("data.csv", "rb") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",")

    # Skip headers
    csv_reader.next()

    # Read data and zip into columns
    data_columns = zip(*csv_reader)

    # Convert CSV columns to numpy
    frame = np.asarray(data_columns[0], dtype=int)
    vicon_t = np.vstack((np.asarray(data_columns[1], dtype=float),
                         np.asarray(data_columns[2], dtype=float)))
    vicon_rz = np.asarray(data_columns[6], dtype=float)
    magneto = np.asarray(data_columns[7], dtype=float)
    optical_flow = np.asarray(data_columns[7], dtype=float)

    # Because there are numerous duplicate Vicon frames, get unique frames and indices
    frame, frame_splits = np.unique(frame, return_index=True)
    vicon_t = vicon_t[:,frame_splits]
    vicon_rz = vicon_rz[frame_splits]

    # Calculate the time between Vicon frames
    frame_delta = (frame[1:] - frame[:-1]) * vicon_frame_time_s

    # Use this to calculate instantaneous velocity
    vicon_v = (vicon_t[:,1:] - vicon_t[:,:-1]) / frame_delta

    # Split magneto readings and optical flow into frames
    magneto_frames = np.split(magneto, frame_splits[1:])
    optical_flow = np.split(optical_flow, frame_splits[1:])

    # Take average of magneto reading and optical flow in each frame and recombine into numpy array
    # **NOTE** unlike matplotlib and VICON magneto angles go clockwise
    magneto = np.asarray([-np.average(m) for m in magneto_frames])
    optical_flow = np.asarray([np.average(o) for o in optical_flow])

    # Calculate shortest angle between each Vicon angle and each Magneto angle; and use this to align
    magneto_offset = np.arctan2(np.sin(vicon_rz - magneto), np.cos(vicon_rz - magneto))
    magneto_origin = np.average(magneto_offset)
    magneto += magneto_origin
    print("Magneto mean error:%f degrees, standard deviation:%f degrees" % (np.rad2deg(magneto_origin), np.std(np.rad2deg(magneto_offset))))

    # Calculate velocity vector from optical flow
    optical_v = (vicon_v / np.linalg.norm(vicon_v, axis=0)) * optical_flow[:-1]

    # Calculate vicon heading vector
    # **NOTE** quiver with array of angles doesn't REALLY seem to work
    vicon_hx = np.cos(vicon_rz)
    vicon_hy = np.sin(vicon_rz)

    magneto_hx = np.cos(magneto)
    magneto_hy = np.sin(magneto)

    # Create plot
    figure, axes = plt.subplots(1, 2, sharey=True)

    # Plot XY position from VICON
    axes[0].plot(vicon_t[0], vicon_t[1], marker="x", markevery=vicon_t.shape[1])
    #axes[1].plot(vicon_t[0], vicon_t[1], marker="x", markevery=vicon_t.shape[1])

    # Plot orientations from VICON and Magneto
    axes[0].quiver(vicon_t[0], vicon_t[1], vicon_hx, vicon_hy, angles="xy", scale=0.05, scale_units="xy", color="red", label="Vicon heading")
    axes[0].quiver(vicon_t[0], vicon_t[1], magneto_hx, magneto_hy, angles="xy", scale=0.05, scale_units="xy", color="blue", label="Magneto heading")

    # Plot velocity from VICON
    axes[1].quiver(vicon_t[0,:-1], vicon_t[1,:-1], vicon_v[0], vicon_v[1], angles="xy", scale=10.0, scale_units="xy", color="green", label="Vicon velocity")
    #axes[1].quiver(vicon_t[0,:-1], vicon_t[1,:-1], optical_v[0], optical_v[1], angles="xy", scale=0.1, scale_units="xy", color="gray", label="Optical flow velocity")

    axes[0].legend()

    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[1].set_xlabel("x [mm]")
    # Show plot
    plt.show()

