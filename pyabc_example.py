import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

import pyabc

pyabc.settings.set_figure_params('pyabc')  # for beautified plots

def model(parameter):
    return {"data": parameter["mu"] + 0.5 * np.random.randn()}

prior = pyabc.Distribution(mu=pyabc.RV("uniform", 0, 5))
def distance(x, x0):
    return abs(x["data"] - x0["data"])

abc = pyabc.ABCSMC(model, prior, distance, population_size=1000)

db_path = os.path.join(tempfile.gettempdir(), "test.db")
observation = 2.5
abc.new("sqlite:///" + db_path, {"data": observation})

history = abc.run(minimum_epsilon=0.1, max_nr_populations=10)