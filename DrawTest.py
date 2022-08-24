import matplotlib.pyplot as plt

y = [1, 4, 9, 16, 25, 36, 49, 64]
x1 = [1, 16, 30, 42, 55, 68, 77, 88]
x2 = [1, 6, 12, 18, 28, 40, 52, 65]
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
l1 = ax.plot(x1, y, 'ys-')

fig2 = plt.figure()
ax2 = fig2.add_axes([0, 0, 1, 1])
ax2.grid(True)
l2 = ax2.plot(x2, y, 'go--')
ax.legend(labels=('tv', 'Smartphone'), loc='lower right')  # legend placed at lower right
ax.set_title("Advertisement effect on sales")
ax.set_xlabel('medium')
ax.set_ylabel('sales')
plt.show()
