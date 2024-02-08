import panel as pn
import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from jax import random
from jax import numpy as jnp
import time
import param as pr
import threading as tr
import plotly.express as px

pn.extension("plotly")

NUM_AGENTS = 5 
MAX_AGENTS = 10
GRID_SIZE = 20 
NUM_STEPS = 50
VIZUALIZE = True
VIZ_DELAY = 0.000001
SEED = 0

shape = (100, 2)
shape3D = (100, 3)

class Data(pr.Parameterized):
    data = pr.Array(np.random.random(shape))
    data3D = pr.Array(np.random.random(shape3D))

    @pn.depends("data")
    def plot_data(self):
        fig = Figure()
        ax = fig.subplots()    

        ax.scatter(self.data[:,0], self.data[:,1])

        return fig
    
    @pn.depends("data3D")
    def plot_data_3D(self):
        fig = px.scatter_3d(self.data3D, x=0, y=1, z=2)
        return fig

def update_data():
    return np.random.random(shape)

def update_data_3D():
    return np.random.random(shape3D)

def thread_data():
    global data
    print("Thread started")
    while True:
        data.data = update_data()
        data.data3D = update_data_3D()
        time.sleep(1)


def main():
    global data
    data = Data()
    print(data)

    t1 = tr.Thread(target=thread_data, name="thread_data", daemon=True)
    if not t1.is_alive():
        t1.start()

    fig = pn.pane.Plotly()
    fig3D = pn.pane.Matplotlib()

    

    plot = pn.Column(data.plot_data, data.plot_data_3D)
    
    plot.servable()

    
main()