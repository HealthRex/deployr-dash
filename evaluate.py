from utils.cosmos import read_predictions_and_labels
from utils.evaluators import BinaryEvaluator
from utils.constants import *

df = read_predictions_and_labels()

# Drop missing - complete case only
df = df[df['label'] != -1]

evalr = BinaryEvaluator(outdir=HCT_OUTPUT_DIR)
evalr(df.label, df.prediction)

