import pathlib
import pickle

import pandas as pd


class Cache:
    def __init__(self, cache_dir):
        self.cache_dir = pathlib.Path(cache_dir)

    def load_from_cache(self, cache_filename, load_cache_func):
        from_cache = load_cache_func(
            cache_filename
        )
        if from_cache is not None:
            return from_cache
        else:
            print(f'{cache_filename} not found in cache {self.cache_dir.resolve()}.')

    def load_cached_df(self, filename):
        path = self.cache_dir / filename
        if not pathlib.Path(path).exists():
            return None
        return pd.read_pickle(path)

    def pickle_dump(self, obj, filename):
        path = self.cache_dir / filename
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_load(self, filename):
        path = self.cache_dir / filename
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_df_to_cache(self, df: pd.DataFrame, filename):
        df.to_pickle(self.cache_dir / filename)
