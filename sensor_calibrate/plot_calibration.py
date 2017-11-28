import csv
import math
import matplotlib.pyplot as plt
import numpy as np

with open("data.csv", "rb") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",")

    # Skip headers
    csv_reader.next()

    # Read data and zip into columns
    data_columns = zip(*csv_reader)

    # Convert CSV columns to numpy
    frame = np.asarray(data_columns[0], dtype=int)
    vicon_tx = np.asarray(data_columns[1], dtype=float)
    vicon_ty = np.asarray(data_columns[2], dtype=float)

    vicon_rz = np.asarray(data_columns[6], dtype=float)
    magneto = np.asarray(data_columns[7], dtype=float)
    optical_flow = np.asarray(data_columns[7], dtype=float)

    # Because there are numerous duplicate Vicon frames, get unique frames and indices
    frame, frame_splits = np.unique(frame, return_index=True)
    vicon_tx = vicon_tx[frame_splits]
    vicon_ty = vicon_ty[frame_splits]
    vicon_rz = vicon_rz[frame_splits]

    # Split magneto readings into frames
    magneto_frames = np.split(magneto, frame_splits[1:])

    # Take average of magneto reading in each frame and recombine into numpy array
    # **NOTE** unlike matplotlib and VICON these angles go clockwise
    magneto = np.asarray([-np.average(m) for m in magneto_frames])

    # Calculate shortest angle between each Vicon angle and each Magneto angle
    magneto_offset = np.arctan2(np.sin(vicon_rz - magneto), np.cos(vicon_rz - magneto))
    magneto_origin = np.average(magneto_offset)
    print("Magneto mean error:%f degrees, standard deviation:%f degrees" % (np.rad2deg(magneto_origin), np.std(np.rad2deg(magneto_offset))))
    magneto += magneto_origin


    # Calculate vicon heading vector
    # **NOTE** quiver with array of angles doesn't REALLY seem to work
    vicon_hx = np.cos(vicon_rz)
    vicon_hy = np.sin(vicon_rz)

    magneto_hx = np.cos(magneto)
    magneto_hy = np.sin(magneto)

    # Create plot
    figure, axis = plt.subplots()

    # Plot XY position from VICON
    axis.plot(vicon_tx, vicon_ty, marker="x", markevery=len(vicon_tx))

    # Plot orientations from VICON and Magneto
    axis.quiver(vicon_tx, vicon_ty, vicon_hx, vicon_hy, angles="xy", scale=0.05, scale_units="xy", color="red")
    axis.quiver(vicon_tx, vicon_ty, magneto_hx, magneto_hy, angles="xy", scale=0.05, scale_units="xy", color="blue")

    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    # Show plot
    plt.show()

