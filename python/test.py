import numpy as np
import pandas as pd

x = np.array(['1.1', '2.2', '3.3'])
y = x.astype(float)
df = pd.DataFrame(([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                            columns=['a', 'b', 'c'])
fig = df['a'].value_counts().plot(kind="bar").get_figure()
fig.savefig('test.png')
print(df)