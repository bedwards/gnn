```python
!pip3 install torch_geometric
```

    Requirement already satisfied: torch_geometric in /usr/local/lib/python3.11/dist-packages (2.6.1)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (3.11.14)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (2025.3.0)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (3.1.6)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (2.0.2)
    Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (5.9.5)
    Requirement already satisfied: pyparsing in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (3.2.1)
    Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (2.32.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (4.67.1)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (2.6.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (6.2.0)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (0.3.0)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->torch_geometric) (1.18.3)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch_geometric) (3.0.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->torch_geometric) (2025.1.31)



```python
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGATConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
sea = []

for gender in ["M", "W"]:
  sea_ = pd.read_csv(f"{data_path}/{gender}Seasons.csv", usecols=["Season", "DayZero"])
  sea_["DayZero"] = pd.to_datetime(sea_["DayZero"])
  sea_ = sea_.rename(columns={"DayZero": f"{gender}DayZero"})
  sea.append(sea_)

sea = pd.merge(sea[0], sea[1], on="Season", how="outer")
sea = sea.sort_values("Season").reset_index(drop=True)

print(f"sea {sea.shape}")
print(sea)
print()
sea.info()
```

    sea (41, 3)
        Season   MDayZero   WDayZero
    0     1985 1984-10-29        NaT
    1     1986 1985-10-28        NaT
    2     1987 1986-10-27        NaT
    3     1988 1987-11-02        NaT
    4     1989 1988-10-31        NaT
    ..     ...        ...        ...
    36    2021 2020-11-02 2020-11-02
    37    2022 2021-11-01 2021-11-01
    38    2023 2022-10-31 2022-10-31
    39    2024 2023-11-06 2023-11-06
    40    2025 2024-11-04 2024-11-04
    
    [41 rows x 3 columns]
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41 entries, 0 to 40
    Data columns (total 3 columns):
     #   Column    Non-Null Count  Dtype         
    ---  ------    --------------  -----         
     0   Season    41 non-null     int64         
     1   MDayZero  41 non-null     datetime64[ns]
     2   WDayZero  28 non-null     datetime64[ns]
    dtypes: datetime64[ns](2), int64(1)
    memory usage: 1.1 KB



```python
tea = pd.DataFrame()

for gender in ["M", "W"]:
  tea = pd.concat([
      tea,
      pd.read_csv(f"{data_path}/{gender}Teams.csv", usecols=["TeamID", "TeamName"]),
  ])

tea = tea.sort_values("TeamID").reset_index(drop=True)

print(f"tea {tea.shape}")
# print(tea)
```

    tea (758, 2)



```python
nodes = []

for gender in ["M", "W"]:
  for part in ["RegularSeason", "NCAATourney"]:
    nodes_gender_part = pd.read_csv(f"{data_path}/{gender}{part}DetailedResults.csv")
    nodes_gender_part["Men"] = gender == "M"
    nodes_gender_part["NCAATourney"] = part == "NCAATourney"
    nodes.append(nodes_gender_part)

nodes = pd.concat(nodes)
nodes["WLoc"] = nodes["WLoc"].map({"A": -1, "N": 0, "H": 1})
nodes["LLoc"] = nodes["WLoc"] * -1

for c in nodes:
  nodes[c] = nodes[c].astype("int32")

both = nodes[[c for c in nodes if c[0] not in ("W", "L")]]

def extract(W_or_L, Le_or_Ri):
  return nodes[[c for c in nodes if c[0] == W_or_L]].rename(columns={c: f"{Le_or_Ri}_{c[1:]}" for c in nodes})

nodes = pd.concat([
  pd.concat([both, extract("W", "Le"), extract("L", "Ri")], axis=1),
  pd.concat([both, extract("L", "Le"), extract("W", "Ri")], axis=1),
])

# Date
nodes = pd.merge(nodes, sea, on="Season")
daynum = pd.to_timedelta(nodes["DayNum"], unit="D")
nodes["Date"] = nodes["WDayZero"] + daynum
nodes.loc[nodes["Men"], "Date"] = nodes["MDayZero"] + daynum
nodes = nodes.drop(columns=["MDayZero", "WDayZero"])

# TeamName
def add_team_name(Le_or_Ri):
  return pd.merge(nodes, tea, left_on=f"{Le_or_Ri}_TeamID", right_on="TeamID"
    ).rename(columns={"TeamName": f"{Le_or_Ri}_TeamName"}
    ).drop(columns=["TeamID"])
nodes = add_team_name("Le")
nodes = add_team_name("Ri")

# Le_Margin
nodes["Le_Margin"] = nodes["Le_Score"] - nodes["Ri_Score"]

# SeasonsAgo
nodes["SeasonsAgo"] = 2025 - nodes["Season"]

# Le_Loc
nodes = nodes.drop(columns=["Ri_Loc"])

# Split ascending and descending TeamIDs
#   so model doesn't learn from noise in arbitrary order
ascending = nodes["Le_TeamID"] < nodes["Ri_TeamID"]
nodes_asc = nodes[ascending]
nodes_des = nodes[~ascending]
del nodes

# Key
def key(ascending=True):
  if ascending:
    df = nodes_asc
    lesser = "Le"
    greater = "Ri"
  else:
    df = nodes_des
    lesser = "Ri"
    greater = "Le"
  df["Key"] = (
    df["Season"].astype(str) + "_" +
    df["DayNum"].astype(str).str.zfill(3) + "_" +
    df[f"{lesser}_TeamID"].astype(str) + "_" +
    df[f"{greater}_TeamID"].astype(str)
  )
key()
key(ascending=False)

# Index
def index(df):
  df = df.sort_values("Key").reset_index(drop=True)
  df.index = df.index.astype("int32")
  return df.reset_index(names=["Index"])
nodes_asc = index(nodes_asc)
nodes_des = index(nodes_des)

def order_columns(df):
  cols = (
    ["Index", "Key", "Season", "Date"] +
    ["Le_TeamID", "Ri_TeamID", "Le_TeamName", "Ri_TeamName"] +
    ["Le_Margin"] +
    ["SeasonsAgo", "Men", "NCAATourney", "DayNum", "NumOT"] +
    ["Le_Loc"]
  )
  return df[cols + [c for c in df if c not in cols]]
nodes_asc = order_columns(nodes_asc)
nodes_des = order_columns(nodes_des)

print(f"nodes_asc {nodes_asc.shape}")
print(nodes_asc)
# print()
# nodes_asc.info()
print()

print(f"nodes_des {nodes_des.shape}")
print(nodes_des)
# print()
# nodes_des.info()

nodes_asc.to_csv(f"{gnn_path}/nodes_asc.csv", index=False)
nodes_des.to_csv(f"{gnn_path}/nodes_des.csv", index=False)
```

    nodes_asc (202866, 43)
             Index                 Key  Season       Date  Le_TeamID  Ri_TeamID   Le_TeamName     Ri_TeamName  Le_Margin  SeasonsAgo  Men  NCAATourney  DayNum  NumOT  Le_Loc  Le_Score  Le_FGM  Le_FGA  Le_FGM3  Le_FGA3  Le_FTM  Le_FTA  Le_OR  Le_DR  Le_Ast  Le_TO  Le_Stl  Le_Blk  Le_PF  Ri_Score  Ri_FGM  Ri_FGA  Ri_FGM3  Ri_FGA3  Ri_FTM  Ri_FTA  Ri_OR  Ri_DR  Ri_Ast  Ri_TO  Ri_Stl  Ri_Blk  Ri_PF
    0            0  2003_010_1104_1328    2003 2002-11-14       1104       1328       Alabama        Oklahoma          6          22    1            0      10      0       0        68      27      58        3       14      11      18     14     24      13     23       7       1     22        62      22      53        2       10      16      22     10     22       8     18       9       2     20
    1            1  2003_010_1272_1393    2003 2002-11-14       1272       1393       Memphis        Syracuse          7          22    1            0      10      0       0        70      26      62        8       20      10      19     15     28      16     13       4       4     18        63      24      67        6       24       9      20     20     25       7     12       8       6     16
    2            2  2003_011_1186_1458    2003 2002-11-15       1186       1458  E Washington       Wisconsin        -26          22    1            0      11      0      -1        55      20      46        3       11      12      17      6     22       8     19       4       3     25        81      26      57        6       12      23      27     12     24      12      9       9       3     18
    3            3  2003_011_1208_1400    2003 2002-11-15       1208       1400       Georgia           Texas         -6          22    1            0      11      0       0        71      24      62        6       16      17      27     21     15      12     10       7       1     14        77      30      61        6       14      11      13     17     22      12     14       4       4     20
    4            4  2003_011_1266_1437    2003 2002-11-15       1266       1437     Marquette       Villanova         12          22    1            0      11      0       0        73      24      58        8       18      17      29     17     26      15     10       5       2     25        61      22      73        3       26      14      23     31     22       9     12       2       5     23
    ...        ...                 ...     ...        ...        ...        ...           ...             ...        ...         ...  ...          ...     ...    ...     ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...
    202861  202861  2025_132_1276_1458    2025 2025-03-16       1276       1458      Michigan       Wisconsin          6           0    1            0     132      0       0        59      19      59        9       33      12      13      6     34      14      8       3       2     16        53      15      68        7       39      16      18     15     31       8      6       3       2     14
    202862  202862  2025_132_3119_3250    2025 2025-03-16       3119       3250          Army          Lehigh        -12           0    0            0     132      0      -1        62      25      56        6       17       6      10      8     13      10     10       5       0     20        74      27      45        5       14      15      17      5     25      15     15       6       0     12
    202863  202863  2025_132_3125_3293    2025 2025-03-16       3125       3293       Belmont       Murray St        -21           0    0            0     132      0       0        62      24      68        2       21      12      14     12     22      11      7       5       0     16        83      28      54       14       28      13      15      5     33      21     13       2       3     15
    202864  202864  2025_132_3144_3456    2025 2025-03-16       3144       3456      Campbell  William & Mary         -3           0    0            0     132      0       0        63      24      55        5       20      10      12      8     33      12     16       4       2     12        66      27      63        8       23       4       7      5     25       9      9       6       3     13
    202865  202865  2025_132_3192_3476    2025 2025-03-16       3192       3476   F Dickinson       Stonehill         17           0    0            0     132      0       1        66      23      55        3       21      17      18     10     22      11      9       8       1      8        49      21      57        4       20       3       4     14     22      14     17       4       1     17
    
    [202866 rows x 43 columns]
    
    nodes_des (202866, 43)
             Index                 Key  Season       Date  Le_TeamID  Ri_TeamID     Le_TeamName   Ri_TeamName  Le_Margin  SeasonsAgo  Men  NCAATourney  DayNum  NumOT  Le_Loc  Le_Score  Le_FGM  Le_FGA  Le_FGM3  Le_FGA3  Le_FTM  Le_FTA  Le_OR  Le_DR  Le_Ast  Le_TO  Le_Stl  Le_Blk  Le_PF  Ri_Score  Ri_FGM  Ri_FGA  Ri_FGM3  Ri_FGA3  Ri_FTM  Ri_FTA  Ri_OR  Ri_DR  Ri_Ast  Ri_TO  Ri_Stl  Ri_Blk  Ri_PF
    0            0  2003_010_1104_1328    2003 2002-11-14       1328       1104        Oklahoma       Alabama         -6          22    1            0      10      0       0        62      22      53        2       10      16      22     10     22       8     18       9       2     20        68      27      58        3       14      11      18     14     24      13     23       7       1     22
    1            1  2003_010_1272_1393    2003 2002-11-14       1393       1272        Syracuse       Memphis         -7          22    1            0      10      0       0        63      24      67        6       24       9      20     20     25       7     12       8       6     16        70      26      62        8       20      10      19     15     28      16     13       4       4     18
    2            2  2003_011_1186_1458    2003 2002-11-15       1458       1186       Wisconsin  E Washington         26          22    1            0      11      0       1        81      26      57        6       12      23      27     12     24      12      9       9       3     18        55      20      46        3       11      12      17      6     22       8     19       4       3     25
    3            3  2003_011_1208_1400    2003 2002-11-15       1400       1208           Texas       Georgia          6          22    1            0      11      0       0        77      30      61        6       14      11      13     17     22      12     14       4       4     20        71      24      62        6       16      17      27     21     15      12     10       7       1     14
    4            4  2003_011_1266_1437    2003 2002-11-15       1437       1266       Villanova     Marquette        -12          22    1            0      11      0       0        61      22      73        3       26      14      23     31     22       9     12       2       5     23        73      24      58        8       18      17      29     17     26      15     10       5       2     25
    ...        ...                 ...     ...        ...        ...        ...             ...           ...        ...         ...  ...          ...     ...    ...     ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...       ...     ...     ...      ...      ...     ...     ...    ...    ...     ...    ...     ...     ...    ...
    202861  202861  2025_132_1276_1458    2025 2025-03-16       1458       1276       Wisconsin      Michigan         -6           0    1            0     132      0       0        53      15      68        7       39      16      18     15     31       8      6       3       2     14        59      19      59        9       33      12      13      6     34      14      8       3       2     16
    202862  202862  2025_132_3119_3250    2025 2025-03-16       3250       3119          Lehigh          Army         12           0    0            0     132      0       1        74      27      45        5       14      15      17      5     25      15     15       6       0     12        62      25      56        6       17       6      10      8     13      10     10       5       0     20
    202863  202863  2025_132_3125_3293    2025 2025-03-16       3293       3125       Murray St       Belmont         21           0    0            0     132      0       0        83      28      54       14       28      13      15      5     33      21     13       2       3     15        62      24      68        2       21      12      14     12     22      11      7       5       0     16
    202864  202864  2025_132_3144_3456    2025 2025-03-16       3456       3144  William & Mary      Campbell          3           0    0            0     132      0       0        66      27      63        8       23       4       7      5     25       9      9       6       3     13        63      24      55        5       20      10      12      8     33      12     16       4       2     12
    202865  202865  2025_132_3192_3476    2025 2025-03-16       3476       3192       Stonehill   F Dickinson        -17           0    0            0     132      0      -1        49      21      57        4       20       3       4     14     22      14     17       4       1     17        66      23      55        3       21      17      18     10     22      11      9       8       1      8
    
    [202866 rows x 43 columns]



```python
def as_struct(nodes):
  cols = ["Index", "Date", "Season", "NCAATourney"]
  return np.array(
    list(nodes[cols].itertuples(index=False)),
    dtype=[(c, nodes[c].dtype) for c in cols]
  )


def create_edges(source, target, edge_type):
  source, target = [pd.DataFrame(n.flatten()) for n in np.meshgrid(source, target)]
  edges = pd.concat([
      source.rename(columns={c: f"Source{c}" for c in source}),
      target.rename(columns={c: f"Target{c}" for c in target}),
  ], axis=1)
  edges["Type"] = edge_type
  edges["Type"] = edges["Type"].astype("int32")
  edges["Delta"] = ((edges["TargetDate"] - edges["SourceDate"]).dt.days).astype("int32")
  edges.insert(9, "Direction", np.sign(edges["Delta"]).astype("int32"))
  edges["Delta"] = np.abs(edges["Delta"])
  edges = edges.drop(columns=["SourceDate", "TargetDate"])
  return edges


edges = []

for season in range(nodes_asc["Season"].min(), nodes_asc["Season"].max() + 1):
  print(f"Processing {season}")
  nodes_season = nodes_asc[nodes_asc["Season"] == season]

  for Le_TeamID, Le_nodes in nodes_season.groupby("Le_TeamID"):
    Le_struct = as_struct(Le_nodes)
    edges.append(create_edges(Le_struct, Le_struct, 0))
    Ri_nodes = nodes_season[nodes_season["Ri_TeamID"] == Le_TeamID]
    Ri_struct = as_struct(Ri_nodes)
    edges.append(create_edges(Le_struct, Ri_struct, 1))

    for Ri_TeamID in Le_nodes["Ri_TeamID"].unique():
      opp_Le_nodes = nodes_season[nodes_season["Le_TeamID"] == Ri_TeamID]
      opp_Le_struct = as_struct(opp_Le_nodes)
      edges.append(create_edges(Le_struct, opp_Le_struct, 4))
      opp_Ri_nodes = nodes_season[(nodes_season["Ri_TeamID"] == Ri_TeamID) & (nodes_season["Le_TeamID"] != Le_TeamID)]
      opp_Ri_struct = as_struct(opp_Ri_nodes)
      edges.append(create_edges(Le_struct, opp_Ri_struct, 5))

  for Ri_TeamID, Ri_nodes in nodes_season.groupby("Ri_TeamID"):
    Ri_struct = as_struct(Ri_nodes)
    edges.append(create_edges(Ri_struct, Ri_struct, 2))
    Le_nodes = nodes_season[nodes_season["Le_TeamID"] == Ri_TeamID]
    Le_struct = as_struct(Le_nodes)
    edges.append(create_edges(Ri_struct, Le_struct, 3))

    for Le_TeamID in Ri_nodes["Le_TeamID"].unique():
      opp_Ri_nodes = nodes_season[nodes_season["Ri_TeamID"] == Le_TeamID]
      opp_Ri_struct = as_struct(opp_Ri_nodes)
      edges.append(create_edges(Ri_struct, opp_Ri_struct, 6))
      opp_Le_nodes = nodes_season[(nodes_season["Le_TeamID"] == Le_TeamID) & (nodes_season["Ri_TeamID"] != Ri_TeamID)]
      opp_Le_struct = as_struct(opp_Le_nodes)
      edges.append(create_edges(Ri_struct, opp_Le_struct, 7))

edges = pd.concat(edges)
edges = edges[edges["SourceIndex"] != edges["TargetIndex"]]
edges = edges.sort_values(["SourceIndex", "TargetIndex", "Type"]).reset_index(drop=True)
edges.index = edges.index.astype("int32")

print(f"edges {edges.shape}")
print(edges)
print()
edges.info()

edges.to_csv(f"{gnn_path}/edges.csv", index=False)
```

    Processing 2003
    Processing 2004
    Processing 2005
    Processing 2006
    Processing 2007
    Processing 2008
    Processing 2009
    Processing 2010
    Processing 2011
    Processing 2012
    Processing 2013
    Processing 2014
    Processing 2015
    Processing 2016
    Processing 2017
    Processing 2018
    Processing 2019
    Processing 2020
    Processing 2021
    Processing 2022
    Processing 2023
    Processing 2024
    Processing 2025
    edges (183746250, 9)
               SourceIndex  SourceSeason  SourceNCAATourney  TargetIndex  TargetSeason  TargetNCAATourney  Type  Direction  Delta
    0                    0          2003                  0            3          2003                  0     4          1      1
    1                    0          2003                  0           10          2003                  0     4          1      3
    2                    0          2003                  0           14          2003                  0     4          1      4
    3                    0          2003                  0           24          2003                  0     5          1      4
    4                    0          2003                  0           27          2003                  0     4          1      5
    ...                ...           ...                ...          ...           ...                ...   ...        ...    ...
    183746245       202865          2025                  0       202758          2025                  0     3         -1      3
    183746246       202865          2025                  0       202758          2025                  0     4         -1      3
    183746247       202865          2025                  0       202758          2025                  0     5         -1      3
    183746248       202865          2025                  0       202797          2025                  0     7         -1      2
    183746249       202865          2025                  0       202809          2025                  0     5         -1      2
    
    [183746250 rows x 9 columns]
    
    <class 'pandas.core.frame.DataFrame'>
    Index: 183746250 entries, 0 to 183746249
    Data columns (total 9 columns):
     #   Column             Dtype
    ---  ------             -----
     0   SourceIndex        int32
     1   SourceSeason       int32
     2   SourceNCAATourney  int32
     3   TargetIndex        int32
     4   TargetSeason       int32
     5   TargetNCAATourney  int32
     6   Type               int32
     7   Direction          int32
     8   Delta              int32
    dtypes: int32(9)
    memory usage: 6.8 GB



```python

```
