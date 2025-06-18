from . import utils, Operators, States, Evolution, Metrics, QRC
import types

# Dynamically add all functions from mymodule to the package namespace
for name, obj in vars(utils).items():
    if isinstance(obj, types.FunctionType):
        globals()[name] = obj

for name, obj in vars(Operators).items():
    if isinstance(obj, types.FunctionType):
        globals()[name] = obj

for name, obj in vars(States).items():
    if isinstance(obj, types.FunctionType):
        globals()[name] = obj

for name, obj in vars(Evolution).items():
    if isinstance(obj, types.FunctionType):
        globals()[name] = obj

for name, obj in vars(Metrics).items():
    if isinstance(obj, types.FunctionType):
        globals()[name] = obj

for name, obj in vars(QRC).items():
    if isinstance(obj, types.FunctionType):
        globals()[name] = obj



