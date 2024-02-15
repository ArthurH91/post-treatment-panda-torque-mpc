import pathlib
import pickle

def load_pickle():
    pkl_path = pathlib.Path().resolve().parent / "pickle"
    print(pkl_path)
    b = {}
    for file in pkl_path.glob("*.pkl"):
        with open(file, "rb") as handle:
            b[file.name] = pickle.load(handle)
    return b