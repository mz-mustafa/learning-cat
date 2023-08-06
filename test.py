import pandas as pd

df_test = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df_test = pd.core.frame.DataFrame.append(df_test, {'A': 5, 'B': 6}, ignore_index=True)
print(df_test)
