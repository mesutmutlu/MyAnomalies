import numpy as np
from scipy.spatial import distance

f1 = np.array([1,1,2,2,0,5,5,5,0,0])
f1_ = np.array([1,1,1,1,0,1,1,1,0,0])
f2 = np.array([1,1,1,1,1,1,1,1,1,1])
f3 = np.array([0,1,0,1,1,1,1,0,0,1])

print("cosine, ""f1", "f2", 1-distance.cosine(f1,f2))
print("cosine, ""f1", "f3", 1-distance.cosine(f1,f3))
print("cosine, ""f2", "f3", 1-distance.cosine(f2,f3))
print("jaccard, ""f1", "f2", 1-distance.jaccard(f1_.tolist(),f2.tolist()))
print("jaccard, ""f1", "f3", 1-distance.jaccard(f1_.tolist(),f3.tolist()))
print("jaccard, ""f2", "f3", 1-distance.jaccard(f2.tolist(),f3.tolist()))

print(4/(2*np.sqrt(10)))