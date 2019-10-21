# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#################
###　 Results  ###
#################

#526339430
n_epoch = 20
accuracy_all = np.array([0.2114, 0.2208, 0.2925, 0.3567, 0.4004,
                         0.4668, 0.4229, 0.4275, 0.5273, 0.5645, 
                         0.554, 0.596, 0.6006, 0.6245, 0.6245,   
                         0.6714, 0.6904, 0.6895, 0.685, 0.6978])
plt.figure(figsize=(5,4))
plt.plot(np.arange(1, n_epoch+1), accuracy_all*100,
         color="k")
plt.xlabel("Epoch")
plt.ylabel("Train accuracy (%)")
plt.savefig("accuracy.pdf")