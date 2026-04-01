from scipy.io import loadmat

# Load the .mat file
data = loadmat('/Users/anushka/Downloads/mpii_human_pose_v1_u12_2.mat')

# Display variable names and data
print(data.keys())
