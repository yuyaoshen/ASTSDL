print("Checking the version of all packages...")
import pickle
print("pickle: ",pickle.format_version)
import pandas as pd
print("pandas: ",pd.__version__)
import numpy
print("numpy: ",numpy.__version__)
import tensorflow
print("tensorflow: ",tensorflow.__version__)
from tensorflow import keras
print("keras: ",keras.__version__)
