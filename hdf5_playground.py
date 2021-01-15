import h5py

file = h5py.File('/Users/lucamasserano/Desktop/uni/cmu/ada/ada_code/1.hdf5', "r")

file.keys()

# Hits are stored in ‘E’ as a dense 4D tensor with dimensions (muon, z-cell, x-cell, y-cell)
hits = file['E']
hits

# The true energy is stored in ‘y’ as a vector
true_energy = file['y']
true_energy

# Hl feats are stored in ‘hl_x’ as a matrix with dimensions (muon, HL feats) --> obsolete, now done in c++
hl_feats = file['hl_x']
hl_feats

# Meta data about the muons, the mean and standard deviation of the HL feats
metadata = file['meta']
metadata
