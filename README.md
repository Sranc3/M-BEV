# M-BEV
### Demo for M-BEV ###
This is the implementation for the paper M-BEV: Masked BEV Perception for Robust Autonomous Driving.

M-BEV is a perception framework to improve the robustness for camera-based autonomous driving methods.
We develop a novel Masked View Reconstruction (MVR) module in our M-BEV. It mimics various missing cases by randomly masking features of different camera views, then leverages the original features of these views as self-supervision, and reconstructs the masked ones with the distinct spatio-temporal context across camera views. Via such a plug-and-play MVR, our M-BEV is capable of learning the missing
views from the resting ones, and thus well generalized for robust view recovery and accurate perception in the testing.
