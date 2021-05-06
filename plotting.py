import matplotlib.pyplot as plt
import numpy as np
import time

x = np.linspace(0, 10*np.pi, 100)
y1 = np.sin(x)
y2 = np.sin(x-0.1)

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
line1, = ax1.plot(x, y1, 'b-')
line2, = ax2.plot(x, y2, 'b-')

for phase in np.linspace(0, 10*np.pi, 100):
    line1.set_ydata(np.sin(0.5 * x + phase))
    line2.set_ydata(np.sin(0.5 * (x-0.1) + phase))
    fig.canvas.draw()
    fig.canvas.flush_events()