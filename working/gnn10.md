```python
!pip3 install torch_geometric
```

    Collecting torch_geometric
      Downloading torch_geometric-2.6.1-py3-none-any.whl.metadata (63 kB)
    [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/63.1 kB[0m [31m?[0m eta [36m-:--:--[0m[2K     [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━[0m [32m61.4/63.1 kB[0m [31m2.5 MB/s[0m eta [36m0:00:01[0m[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.1/63.1 kB[0m [31m1.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: aiohttp in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (3.11.14)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (2025.3.0)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (3.1.6)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (2.0.2)
    Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (5.9.5)
    Requirement already satisfied: pyparsing in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (3.2.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (2.32.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (4.67.1)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (2.6.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (6.2.0)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (0.3.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (1.18.3)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch_geometric) (3.0.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (2025.1.31)
    Downloading torch_geometric-2.6.1-py3-none-any.whl (1.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m16.8 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: torch_geometric
    Successfully installed torch_geometric-2.6.1



```python
import warnings
from functools import partial
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv
from torch_geometric.utils import subgraph
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from google.colab import drive

warnings.filterwarnings("ignore")

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.min_rows", 10)
pd.set_option("display.max_rows", 10)
pd.set_option("display.width", None)

sns.set_theme(style="whitegrid")

drive_path = "/content/drive"
drive.mount(drive_path)
base_path = f"{drive_path}/My Drive/Colab Notebooks/gnn/input"
data_path = f"{base_path}/march-machine-learning-mania-2025"
gnn_path = f"{base_path}/gnn"

device = "cuda" if torch.cuda.is_available() else "cpu"
```

    Mounted at /content/drive



```python
def print_df(name, df, info=False):
  print(f"{name} {df.shape}")
  print(df)
  print()
  if info:
    df.info()
    print()
```


```python
def load_nodes(asc_or_des):
  nodes = pd.read_csv(f"{gnn_path}/nodes_{asc_or_des}.csv")
  nodes["Date"] = pd.to_datetime(nodes["Date"])

  nodes = pd.concat([
      # indentifying info, not passed to model
      nodes[["Index"]].astype("int32"),
      nodes[["Key"]],
      nodes[["Season"]].astype("int32"),
      nodes[["Date"]],
      nodes[["Le_TeamID", "Ri_TeamID"]].astype("int32"),
      nodes[["Le_TeamName", "Ri_TeamName"]],

      # target (scaled as Le_y)
      nodes[["Le_Margin"]].astype("int32"),

      # features (not scaled)
      nodes[["Men", "NCAATourney", "Le_Loc"]].astype("int32"),

      # # features (scaled)
      nodes[["SeasonsAgo", "DayNum", "NumOT"]].astype("int32"),
      nodes.loc[:, "Le_Score":].astype("int32"),
    ],
    axis=1,
  )

  nodes.index = nodes.index.astype("int32")

  return nodes
```


```python
def scale(scaler, df, cols=None):
  return pd.DataFrame(
    scaler.transform(df).astype("float32"),
    index=df.index,
    columns=df.columns if cols is None else cols,
  )


scaler_x = StandardScaler()
scaler_y = StandardScaler()


def scale_values(nodes):
  return pd.concat([
      nodes.loc[:, :"Le_Margin"],
      scale(scaler_y, nodes[["Le_Margin"]], ["Le_y"]),
      nodes.loc[:, "Men":"Le_Loc"].astype("float32"),
      scale(scaler_x, nodes.loc[:, "SeasonsAgo":]),
    ],
    axis=1,
  )
```


```python
nodes = [load_nodes(d) for d in ("asc", "des")]
nodes_doubled = pd.concat(nodes)
scaler_x.fit(nodes_doubled.loc[:, "SeasonsAgo":])
scaler_y.fit(nodes_doubled[["Le_Margin"]])
nodes = [scale_values(n) for n in nodes]
print_df("nodes[0]", nodes[0])
```

    nodes[0] (202866, 44)
             Index                 Key  Season       Date  Le_TeamID  Ri_TeamID   Le_TeamName     Ri_TeamName  Le_Margin      Le_y  Men  NCAATourney  Le_Loc  SeasonsAgo    DayNum     NumOT  Le_Score    Le_FGM    Le_FGA   Le_FGM3   Le_FGA3    Le_FTM    Le_FTA     Le_OR     Le_DR    Le_Ast     Le_TO    Le_Stl    Le_Blk     Le_PF  Ri_Score    Ri_FGM    Ri_FGA   Ri_FGM3   Ri_FGA3    Ri_FTM    Ri_FTA     Ri_OR     Ri_DR    Ri_Ast     Ri_TO    Ri_Stl    Ri_Blk     Ri_PF
    0            0  2003_010_1104_1328    2003 2002-11-14       1104       1328       Alabama        Oklahoma          6  0.364316  1.0          0.0     0.0    2.069352 -1.669631 -0.214915  0.021993  0.566312  0.102891 -1.084275 -0.794976 -0.364918 -0.108382  0.691001  0.002141 -0.003511  1.793540 -0.013958 -1.005587  0.935628 -0.435726 -0.414578 -0.539242 -1.411433 -1.427122  0.475545  0.406779 -0.208418 -0.373919 -1.106682  0.755983  0.592462 -0.567787  0.494275
    1            1  2003_010_1272_1393    2003 2002-11-14       1272       1393       Memphis        Syracuse          7  0.425036  1.0          0.0     0.0    2.069352 -1.669631 -0.214915  0.174566  0.370134  0.616597  0.551516  0.153242 -0.533011  0.020409  0.915856  0.754260  0.658391 -0.281574 -0.923589  0.307814  0.052923 -0.359440 -0.022222  1.258731 -0.102800  0.785388 -0.701103  0.149199  2.040129  0.190171 -1.327316 -0.489085  0.289252  1.183415 -0.388430
    2            2  2003_011_1186_1458    2003 2002-11-15       1186       1458  E Washington       Wisconsin        -26 -1.578703  1.0          0.0    -1.0    2.069352 -1.642327 -0.214915 -0.969732 -0.806934 -1.438229 -1.084275 -1.269085 -0.196825 -0.237172 -1.107837 -0.373919 -1.106682  0.963494 -0.923589 -0.129986  1.597657  1.013718  0.370134 -0.025536 -0.102800 -1.111049  1.652194  1.050730  0.241292  0.002141 -0.224145 -1.111619  0.592462 -0.129986  0.052923
    3            3  2003_011_1208_1400    2003 2002-11-15       1208       1400       Georgia           Texas         -6 -0.364316  1.0          0.0     0.0    2.069352 -1.642327 -0.214915  0.250853 -0.022222  0.616597 -0.102800 -0.478903  0.643638  1.050730  2.264984 -1.690127 -0.224145 -0.904108 -0.013958 -1.005587 -0.829782  0.708572  1.154847  0.488171 -0.102800 -0.794976 -0.364918 -0.752333  1.365565 -0.373919 -0.224145 -0.074062 -0.923589  0.307814  0.494275
    4            4  2003_011_1266_1437    2003 2002-11-15       1266       1437     Marquette       Villanova         12  0.728632  1.0          0.0     0.0    2.069352 -1.642327 -0.214915  0.403426 -0.022222  0.102891  0.551516 -0.162830  0.643638  1.308311  1.365565  0.378201  0.437757 -0.904108 -0.620379 -0.567787  1.597657 -0.512013 -0.414578  2.029291 -1.084275  1.101461  0.139360  0.535570  4.513531 -0.373919 -0.886047 -0.489085 -1.530009  0.745615  1.156304
    ...        ...                 ...     ...        ...        ...        ...           ...             ...        ...       ...  ...          ...     ...         ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
    202861  202861  2025_132_1276_1458    2025 2025-03-16       1276       1458      Michigan       Wisconsin          6  0.364316  1.0          0.0     0.0   -1.553550  1.661486 -0.214915 -0.664586 -1.003112  0.231318  0.878674  2.207716 -0.196825 -0.752333 -1.107837  1.882439  0.217123 -1.319131 -1.226799 -0.567787 -0.388430 -1.122305 -1.787825  1.387157  0.224358  3.155934  0.475545 -0.108382  0.915856  1.318350 -1.106682 -1.734154 -1.226799 -0.567787 -0.829782
    202862  202862  2025_132_3119_3250    2025 2025-03-16       3119       3250          Army          Lehigh        -12 -0.728632  0.0          0.0    -1.0   -1.553550  1.661486 -0.214915 -0.435726  0.173956 -0.153962 -0.102800 -0.320867 -1.205381 -1.138703 -0.658127 -2.066187 -0.665413 -0.904108 -0.620379 -1.443388  0.494275  0.479712  0.566312 -1.566656 -0.429959 -0.794976  0.307453 -0.237172 -1.332691  0.190171  0.437757  0.133449 -0.317169 -1.443388 -1.271135
    202863  202863  2025_132_3125_3293    2025 2025-03-16       3125       3293       Belmont       Murray St        -21 -1.275107  0.0          0.0     0.0   -1.553550  1.661486 -0.214915 -0.435726 -0.022222  1.387157 -1.411433  0.311279 -0.196825 -0.623542  0.241292 -0.373919 -0.444779 -1.526642 -0.620379 -1.443388 -0.388430  1.166291  0.762490 -0.410816  2.514465  1.417534 -0.028733 -0.494752 -1.332691  1.694409  1.761561 -0.281574 -1.530009 -0.129986 -0.609106
    202864  202864  2025_132_3144_3456    2025 2025-03-16       3144       3456      Campbell  William & Mary         -3 -0.182158  0.0          0.0     0.0   -1.553550  1.661486 -0.214915 -0.359440 -0.022222 -0.282389 -0.429959  0.153242 -0.533011 -0.881123 -0.658127  1.694409 -0.224145  0.340960 -0.923589 -0.567787 -1.271135 -0.130580  0.566312  0.745024  0.551516  0.627352 -1.541567 -1.525074 -1.332691  0.190171 -0.886047 -1.111619 -0.317169 -0.129986 -1.050459
    202865  202865  2025_132_3192_3476    2025 2025-03-16       3192       3476   F Dickinson       Stonehill         17  1.032229  0.0          0.0     1.0   -1.553550  1.661486 -0.214915 -0.130580 -0.218400 -0.282389 -1.084275  0.311279  0.643638 -0.108382 -0.208418 -0.373919 -0.444779 -1.111619  0.289252 -1.005587 -2.153840 -1.427451 -0.610756 -0.025536 -0.757117  0.153242 -1.709659 -1.911445  0.691001 -0.373919  0.217123  0.548472 -0.923589 -1.005587 -0.167754
    
    [202866 rows x 44 columns]
    



```python
edges = pd.read_csv(f"{gnn_path}/edges.csv", dtype="int32")
edges[["Direction", "Delta"]] = edges[["Direction", "Delta"]].astype("float32")
edges.index = edges.index.astype("int32")
print_df("edges", edges)
```

    edges (183746250, 9)
               SourceIndex  SourceSeason  SourceNCAATourney  TargetIndex  TargetSeason  TargetNCAATourney  Type  Direction  Delta
    0                    0          2003                  0            3          2003                  0     4        1.0    1.0
    1                    0          2003                  0           10          2003                  0     4        1.0    3.0
    2                    0          2003                  0           14          2003                  0     4        1.0    4.0
    3                    0          2003                  0           24          2003                  0     5        1.0    4.0
    4                    0          2003                  0           27          2003                  0     4        1.0    5.0
    ...                ...           ...                ...          ...           ...                ...   ...        ...    ...
    183746245       202865          2025                  0       202758          2025                  0     3       -1.0    3.0
    183746246       202865          2025                  0       202758          2025                  0     4       -1.0    3.0
    183746247       202865          2025                  0       202758          2025                  0     5       -1.0    3.0
    183746248       202865          2025                  0       202797          2025                  0     7       -1.0    2.0
    183746249       202865          2025                  0       202809          2025                  0     5       -1.0    2.0
    
    [183746250 rows x 9 columns]
    



```python
test = nodes[0][
  (2021 <= nodes[0]["Season"]) &
  (nodes[0]["Season"] <= 2024) &
  (nodes[0]["NCAATourney"] == 1)
].index

train = nodes[0][~nodes[0].index.isin(test)].index

print(test)
print()
print(train)
```

    Index([158848, 158849, 158850, 158851, 158852, 158853, 158854, 158855, 158856, 158857,
           ...
           191771, 191772, 191773, 191774, 191775, 191776, 191777, 191778, 191779, 191780],
          dtype='int32', length=531)
    
    Index([     0,      1,      2,      3,      4,      5,      6,      7,      8,      9,
           ...
           202856, 202857, 202858, 202859, 202860, 202861, 202862, 202863, 202864, 202865],
          dtype='int32', length=202335)



```python
def tensor(data):
  return torch.tensor(data.values, device=device, dtype=torch.float32)


def long_tensor(data):
  return torch.tensor(data.values, device=device, dtype=torch.long)


xs = [tensor(n.loc[:, "Men":]) for n in nodes]
y_trues = [tensor(n[["Le_y"]]) for n in nodes]
edge_index = long_tensor(edges[["SourceIndex", "TargetIndex"]].T)
edge_type = long_tensor(edges["Type"])
edge_attr = tensor(edges[["Direction", "Delta"]])


class Model(nn.Module):
  def __init__(self, layers, transforms):
    super().__init__()
    self.layers = nn.ModuleList(layers)
    self.transforms = transforms

  def forward(self, node_indices, x):
    y_pred = x[node_indices]

    ei, _, mask = subgraph(
        node_indices,
        edge_index,
        relabel_nodes=True,
        return_edge_mask=True,
    )

    for transform in self.transforms:
      y_pred = transform(
        y_pred,
        ei,
        edge_type[mask],
        edge_attr[mask],
      )

    return y_pred


def transform_rgat(layer, x, *edge_args):
  out = layer(x, *edge_args)
  out = F.leaky_relu(out)
  return F.dropout(out, training=layer.training)


def transform_linear(layer, x, *edge_args):
  return layer(x)


def initialize_model(layer_sizes, heads=4):
  layers = []
  transforms = []

  for i in range(len(layer_sizes) - 1):
    inp = layer_sizes[i] * (heads if i > 0 else 1)
    out = layer_sizes[i + 1]

    if i < len(layer_sizes) - 2:
      layer = RGATConv(
        inp,
        out,
        num_relations=edges["Type"].unique().shape[0],
        heads=heads,
        edge_dim=len(["Direction", "Delta"]),
      )

      transform = partial(transform_rgat, layer)

    else:
      layer = nn.Linear(inp, out)
      transform = partial(transform_linear, layer)

    layers.append(layer)
    transforms.append(transform)

  return Model(layers, transforms)


def brier_score(margin_pred, margin_true):
    win_prob_pred = 1 / (1 + np.exp(-margin_pred * 0.175))
    win_true = (margin_true > 0).astype("int32")
    return np.mean((win_prob_pred - win_true) ** 2)


def calculate_score(y_preds, train_or_test):
  score = 0

  for y_pred, n in zip(y_preds, nodes):
    margin_pred = scaler_y.inverse_transform(
      y_pred.cpu().numpy().reshape(-1, 1)
    ).flatten()

    score += brier_score(
        margin_pred[train_or_test],
        n.loc[train_or_test, "Le_Margin"]
    )

  return score / len(nodes)
```


```python
def train_models(
    hidden_layer_sizes=[64, 32, 16],
    n_epochs=10_000,
    patience=60,
  ):
  layer_sizes = [xs[0].shape[1]] + hidden_layer_sizes + [y_trues[0].shape[1]]
  kfold = KFold(shuffle=True, random_state=42)

  y_pred_oofs = [
    torch.zeros(y_true.shape[0], device=device, dtype=torch.float32)
    for y_true in y_trues
  ]

  state_dicts = []

  for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train), 1):
    print(f"  fold {fold_n}")
    start = datetime.now()
    i_fold = long_tensor(train[i_fold])
    i_oof = long_tensor(train[i_oof])
    model = initialize_model(layer_sizes)
    adam = torch.optim.Adam(model.parameters())

    for epoch_n in range(1, n_epochs + 1):
      model.train()
      y_pred_epoch_folds = [model.forward(i_fold, x) for x in xs]

      mse_epoch_folds = [
        F.mse_loss(y_pred_epoch_fold, y_true[i_fold])
        for y_pred_epoch_fold, y_true
        in zip(y_pred_epoch_folds, y_trues)
      ]

      adam.zero_grad()

      for mse in mse_epoch_folds:
        mse.backward()

      mse_epoch_fold = (sum(mse_epoch_folds) / len(mse_epoch_folds)).item()
      adam.step()
      model.eval()

      with torch.no_grad():
        y_pred_epoch_oofs = [model.forward(i_oof, x) for x in xs]

        mse_epoch_oof = (sum(
          F.mse_loss(y_pred_epoch_oof, y_true[i_oof])
          for y_pred_epoch_oof, y_true
          in zip(y_pred_epoch_oofs, y_trues)
        ) / len(y_trues)).item()

      if epoch_n == 1 or m_best[0] > mse_epoch_oof:
        m_best = (mse_epoch_oof, 0, model.state_dict())
      else:
        m_best = (m_best[0], m_best[1]+1, m_best[2])

      if ((epoch_n % (n_epochs // 100) == 0)
          or (epoch_n > (n_epochs - 3))
          or (m_best[1] > patience)):
        print(
          f"    epoch {epoch_n:>6}: "
          f"fold={mse_epoch_fold:.4f} "
          f"oof={mse_epoch_oof:.4f}"
        )

      if m_best[1] > patience:
        print(f"    out of patience: oof={m_best[0]:.4f}")
        break

    model.load_state_dict(m_best[2])
    model.eval()

    with torch.no_grad():
      for x, y_pred_oof in zip(xs, y_pred_oofs):
        y_pred_oof[i_oof] = model.forward(i_oof, x).flatten()

    state_dicts.append(model.state_dict())
    t = (datetime.now() - start).total_seconds()
    print(f"  done fold {fold_n} {t} seconds")

  score = calculate_score(y_pred_oofs, train)
  print(f"oof brier score: {score:.4f}")
  return layer_sizes, state_dicts
```


```python
def test_models(layer_sizes, state_dicts):
  y_preds = [
    torch.zeros(y_true.shape[0], device=device, dtype=torch.float32)
    for y_true in y_trues
  ]

  for state_dict in state_dicts:
    model = initialize_model(layer_sizes)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
      for x, y_pred in zip(xs, y_preds):
        y_pred += model.forward(long_tensor(test), x).flatten()

  for y_pred in y_preds:
    y_pred /= len(state_dicts)

  score = calculate_score(y_preds, test)
  print(f"test brier score: {score:.4f}")
```


```python
layer_sizes, state_dicts = train_models()
```

      fold 1



    ---------------------------------------------------------------------------

    OutOfMemoryError                          Traceback (most recent call last)

    <ipython-input-12-205e84709bc8> in <cell line: 0>()
    ----> 1 layer_sizes, state_dicts = train_models()
    

    <ipython-input-10-ff6b91ecf109> in train_models(hidden_layer_sizes, n_epochs, patience)
         24     for epoch_n in range(1, n_epochs + 1):
         25       model.train()
    ---> 26       y_pred_epoch_folds = [model.forward(i_fold, x) for x in xs]
         27 
         28       mse_epoch_folds = [


    <ipython-input-10-ff6b91ecf109> in <listcomp>(.0)
         24     for epoch_n in range(1, n_epochs + 1):
         25       model.train()
    ---> 26       y_pred_epoch_folds = [model.forward(i_fold, x) for x in xs]
         27 
         28       mse_epoch_folds = [


    <ipython-input-9-fdbd1c913add> in forward(self, node_indices, x)
         31 
         32     for transform in self.transforms:
    ---> 33       y_pred = transform(
         34         y_pred,
         35         ei,


    <ipython-input-9-fdbd1c913add> in transform_rgat(layer, x, *edge_args)
         42 
         43 def transform_rgat(layer, x, *edge_args):
    ---> 44   out = layer(x, *edge_args)
         45   out = F.leaky_relu(out)
         46   return F.dropout(out, training=layer.training)


    /usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1738         else:
    -> 1739             return self._call_impl(*args, **kwargs)
       1740 
       1741     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1748                 or _global_backward_pre_hooks or _global_backward_hooks
       1749                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1750             return forward_call(*args, **kwargs)
       1751 
       1752         result = None


    /usr/local/lib/python3.11/dist-packages/torch_geometric/nn/conv/rgat_conv.py in forward(self, x, edge_index, edge_type, edge_attr, size, return_attention_weights)
        348         # propagate_type: (x: Tensor, edge_type: OptTensor,
        349         #                  edge_attr: OptTensor)
    --> 350         out = self.propagate(edge_index=edge_index, edge_type=edge_type, x=x,
        351                              size=size, edge_attr=edge_attr)
        352 


    /tmp/torch_geometric.nn.conv.rgat_conv_RGATConv_propagate_a8bvqq0x.py in propagate(self, edge_index, x, edge_type, edge_attr, size)
        181     else:
        182 
    --> 183         kwargs = self.collect(
        184             edge_index,
        185             x,


    /tmp/torch_geometric.nn.conv.rgat_conv_RGATConv_propagate_a8bvqq0x.py in collect(self, edge_index, x, edge_type, edge_attr, size)
        101     elif isinstance(x, Tensor):
        102         self._set_size(size, i, x)
    --> 103         x_i = self._index_select(x, edge_index_i)
        104     else:
        105         x_i = None


    /usr/local/lib/python3.11/dist-packages/torch_geometric/nn/conv/message_passing.py in _index_select(self, src, index)
        265             return src.index_select(self.node_dim, index)
        266         else:
    --> 267             return self._index_select_safe(src, index)
        268 
        269     def _index_select_safe(self, src: Tensor, index: Tensor) -> Tensor:


    /usr/local/lib/python3.11/dist-packages/torch_geometric/nn/conv/message_passing.py in _index_select_safe(self, src, index)
        288                     f"your node feature matrix and try again.")
        289 
    --> 290             raise e
        291 
        292     def _lift(


    /usr/local/lib/python3.11/dist-packages/torch_geometric/nn/conv/message_passing.py in _index_select_safe(self, src, index)
        269     def _index_select_safe(self, src: Tensor, index: Tensor) -> Tensor:
        270         try:
    --> 271             return src.index_select(self.node_dim, index)
        272         except (IndexError, RuntimeError) as e:
        273             if index.numel() > 0 and index.min() < 0:


    OutOfMemoryError: CUDA out of memory. Tried to allocate 14.82 GiB. GPU 0 has a total capacity of 14.74 GiB of which 4.54 GiB is free. Process 29429 has 10.20 GiB memory in use. Of the allocated memory 9.21 GiB is allocated by PyTorch, and 899.87 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)



```python
test_models(layer_sizes, state_dicts)
```


```python

```
