import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('heartbeat.csv')
plt.plot(df['time'], df['heartrate'])
plt.ylim(1.5, 1.7)
plt.savefig('del.png')