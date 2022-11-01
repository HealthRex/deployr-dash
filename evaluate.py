from cosmos import read_predictions_and_labels
from evaluators import BinaryEvaluator

df = read_predictions_and_labels()

# Drop missing - complete case only
df = df[df['label'] != -1]

evalr = BinaryEvaluator(outdir='./data/HCT/')
evalr(df.label, df.prediction)