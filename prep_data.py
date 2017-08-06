import pandas as pd

from sklearn.datasets import load_digits

digits = load_digits()

df = pd.DataFrame(digits.data)
df.columns = ['digit_{}'.format(v) for v in df.columns]
df['class'] = digits.target
df.to_json('digits.json')
