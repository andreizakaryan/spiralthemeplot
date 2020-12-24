import pandas as pd
from splot import SpiralPlot

df = pd.read_csv('patients.csv')
sp = SpiralPlot(df, 'date', 'disease', 'age')
sp.plot()