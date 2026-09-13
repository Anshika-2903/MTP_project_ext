"""
Loaders for NEW datasets added for the ICLR heterogeneous-coarsening
extension, beyond IMDB/ACM/DBLP (data_loading_hetero.py). Unlike IMDB/ACM/
DBLP, these are LINK-PREDICTION-only datasets: no node-classification labels
or train/val/test masks, so `target_type=None` and only run_linkpred.py (not
eval_hetero_fast.py) applies to them.

Returns the SAME dict contract as data_loading_hetero.build_from_hetero_data
(A_global, X_dict, type_order, node_counts, relations, data, target_type), so
run_befgc_hetero / all three ablations / run_linkpred.py work UNMODIFIED.
Does not modify data_loading_hetero.py or any other existing file.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from data_loading_hetero import build_from_hetero_data


# ---------------------------------------------------------------- LastFM ----
def load_lastfm(root="./data/LastFM"):
    """PyG's built-in LastFM (user/artist/tag, user-artist listening + social
    ties, no classification labels -- standard link-prediction benchmark).
    type_order is read directly off the loaded data rather than hardcoded, so
    this is robust to whatever node-type set/order PyG's loader produces."""
    from torch_geometric.datasets import LastFM
    data = LastFM(root=root)[0]
    type_order = list(data.node_types)
    return build_from_hetero_data(data, type_order, target_type=None)


# ------------------------------------------------------------- MovieLens ----
def load_movielens(root="./data/MovieLens/raw/ml-latest-small"):
    """MovieLens-small (ml-latest-small): user/movie bipartite rating graph.

    Deliberately bypasses PyG's own MovieLens loader, which requires the
    `sentence_transformers` package (not installed in fgc_comp) to embed movie
    titles as text features -- unnecessary complexity for our purposes. We
    build movie features from the already-present multi-hot GENRE labels
    instead (19 standard MovieLens genres), which is simpler, needs no extra
    dependency, and is a completely standard feature choice for this dataset
    in the GNN literature. Users get NO explicit features -- data_loading_
    hetero.build_from_hetero_data's existing feature-propagation fallback
    (mean of rated movies' genre vectors) kicks in automatically, exactly as
    it already does for ACM's featureless "subject" type / DBLP's "conference"
    type, so no new code is needed for that part.

    All (user, movie) rating pairs are used as positive edges regardless of
    the rating VALUE -- the standard "implicit feedback" convention for using
    MovieLens as a link-prediction/recommendation graph (rating aggregated as
    "did this user interact with this movie", not "how much did they like
    it"). Ties into run_linkpred.py exactly like LastFM/IMDB/ACM/DBLP's
    relations do.
    """
    movies = pd.read_csv(os.path.join(root, "movies.csv"))
    ratings = pd.read_csv(os.path.join(root, "ratings.csv"))

    movie_ids = movies["movieId"].to_numpy()
    movie_idx = {mid: i for i, mid in enumerate(movie_ids)}
    n_movies = len(movie_ids)

    all_genres = sorted({g for gl in movies["genres"] for g in gl.split("|") if g != "(no genres listed)"})
    genre_idx = {g: i for i, g in enumerate(all_genres)}
    genre_x = torch.zeros(n_movies, len(all_genres))
    for i, gl in enumerate(movies["genres"]):
        for g in gl.split("|"):
            if g in genre_idx:
                genre_x[i, genre_idx[g]] = 1.0

    user_ids = np.sort(ratings["userId"].unique())
    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    n_users = len(user_ids)

    valid = ratings["movieId"].isin(movie_idx)
    ratings = ratings[valid]
    src = ratings["userId"].map(user_idx).to_numpy()
    dst = ratings["movieId"].map(movie_idx).to_numpy()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    data = HeteroData()
    data["movie"].x = genre_x
    data["user"].num_nodes = n_users
    data["user", "to", "movie"].edge_index = edge_index

    return build_from_hetero_data(data, ["user", "movie"], target_type=None)


NEW_LOADERS = {"LastFM": load_lastfm, "MovieLens": load_movielens}


def load_hetero_new(name):
    if name not in NEW_LOADERS:
        raise ValueError(f"Unknown dataset {name}; available: {list(NEW_LOADERS)}")
    return NEW_LOADERS[name]()
