**Control and Analysis System**

This repository contains the source code developed for the "Design of a filtration and geometric parameter estimation algorithm for layer-by-layer robotic 3D printing control systems)" thesis. 
The code is designed to analyze 2D profilometer scanning data and control the Wire Additive Manufacturing (WAM) process.

**Project Structure**

***The code is divided into two main modules based on their application:***

IPE (In-Process Estimator): Scripts for real-time operation. Includes the PID controller logic to adjust the deposition speed on the fly based on scanning data.

PPE (Post-Process Estimator): Scripts for offline analysis. Includes a graphical user interface (GUI), tools for visualizing 3D layer profiles, applying filters (DBSCAN, SOR, Median), and calculating quality metrics.

Additionally, the .zip file contains full data cloud of points gained from the experimental print, described in the thesis document.
