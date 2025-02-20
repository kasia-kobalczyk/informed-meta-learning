# %%
import numpy as np
import pandas as pd


def sample_function(a, b, c, x):
    return a * x + np.sin(b * x) + c


# %%
N = 2000
curves = []
curve_ids = []
knowledge_ls = []

# Synthetic data
for i in range(N):
    a = np.random.uniform(-1, 1)
    b = np.random.uniform(0, 6)
    c = np.random.uniform(-1, 1)
    x = np.linspace(-2, 2, 100)
    y = sample_function(a, b, c, x)
    knowledge = (a, b, c)
    curves.append(y)
    curve_ids.append(i)
    knowledge_ls.append(knowledge)

curves_df = pd.DataFrame(np.stack(curves, axis=0))
curves_df["curve_id"] = curve_ids
knowledge_df = pd.DataFrame(knowledge_ls)
knowledge_df["curve_id"] = curve_ids
knowledge_df.columns = ["a", "b", "c", "curve_id"]


curves_df.to_csv("../data/trending-sinusoids/data.csv", index=False)
knowledge_df.to_csv("../data/trending-sinusoids/knowledge.csv", index=False)

# random splitting
train_ids = list(np.random.choice(curves_df.curve_id, 1000, replace=False))
val_ids = list(
    np.random.choice(
        curves_df[~curves_df.curve_id.isin(train_ids)].curve_id, 500, replace=False
    )
)
test_ids = list(curves_df[~curves_df.curve_id.isin(train_ids + val_ids)].curve_id)
split_df = pd.DataFrame({"curve_id": knowledge_df.curve_id})
split_df.loc[split_df.curve_id.isin(train_ids), "split"] = "train"
split_df.loc[split_df.curve_id.isin(val_ids), "split"] = "val"
split_df.loc[split_df.curve_id.isin(test_ids), "split"] = "test"
split_df.to_csv("../data/trending-sinusoids/splits.csv", index=False)


# %%

# Synthetic data with dist shift
curves = []
curve_ids = []
knowledge_ls = []

# Training and Eval curves
for i in range(1500):
    a = np.random.uniform(-1, 1)
    b = np.random.normal(2, 1)
    c = np.random.uniform(-1, 1)
    x = np.linspace(-2, 2, 100)
    y = sample_function(a, b, c, x)
    knowledge = (a, b, c)
    curves.append(y)
    curve_ids.append(i)
    knowledge_ls.append(knowledge)

# Testing curves
for i in range(1500, 2000):
    a = np.random.uniform(-1, 1)
    b = np.random.normal(3, 1)
    c = np.random.uniform(-1, 1)
    x = np.linspace(-2, 2, 100)
    y = sample_function(a, b, c, x)
    knowledge = (a, b, c)
    curves.append(y)
    curve_ids.append(i)
    knowledge_ls.append(knowledge)

splits = ["train"] * 1000 + ["val"] * 500 + ["test"] * 500

curves_df = pd.DataFrame(np.stack(curves, axis=0))
curves_df["curve_id"] = curve_ids
knowledge_df = pd.DataFrame(knowledge_ls)
knowledge_df["curve_id"] = curve_ids
knowledge_df.columns = ["a", "b", "c", "curve_id"]
split_df = pd.DataFrame({"curve_id": curve_ids, "split": splits})

curves_df.to_csv("../data/trending-sinusoids-dist-shift/data.csv", index=False)
knowledge_df.to_csv("../data/trending-sinusoids-dist-shift/knowledge.csv", index=False)
split_df.to_csv("../data/trending-sinusoids-dist-shift/splits.csv", index=False)
# %%
