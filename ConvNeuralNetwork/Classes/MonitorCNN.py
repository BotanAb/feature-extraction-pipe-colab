import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib; matplotlib.use("TkAgg")


class Animate:

    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.xs = []
        self.ys = []
        self.y_label = "Loss"
        self.x_label = "Epochs"

    def _update(self, i):

        self.xs = []
        self.ys = []

        graph_data = open('logloss.txt', 'r').read()
        lines = graph_data.split('\n')
        for line in lines:
            if len(line) > 1:
                x, y = line.split(',')
                self.xs.append(float(x))
                self.ys.append(float(y))

        self.ax.clear()
        self.ax.plot(self.xs, self.ys)

        plt.title(self.y_label + ' over time')
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)

    def start(self):

        open('logloss.txt', 'w').close()
        self.anim = animation.FuncAnimation(self.fig, self._update, interval=10)
        plt.show()