```python
!pip3 install torch_geometric
```

    Collecting torch_geometric
      Downloading torch_geometric-2.6.1-py3-none-any.whl.metadata (63 kB)
    [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/63.1 kB[0m [31m?[0m eta [36m-:--:--[0m[2K     [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━[0m [32m61.4/63.1 kB[0m [31m2.2 MB/s[0m eta [36m0:00:01[0m[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.1/63.1 kB[0m [31m1.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: aiohttp in /usr/local/lib/python3.11/dist-packages (from torch_geometric) (3.11.14)
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
    Downloading torch_geometric-2.6.1-py3-none-any.whl (1.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m15.7 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: torch_geometric
    Successfully installed torch_geometric-2.6.1



```python
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
sea = []

for gender in ["M", "W"]:
  sea_ = pd.read_csv(f"{data_path}/{gender}Seasons.csv", usecols=["Season", "DayZero"])
  sea_["DayZero"] = pd.to_datetime(sea_["DayZero"])
  sea_ = sea_.rename(columns={"DayZero": f"{gender}DayZero"})
  sea.append(sea_)

sea = pd.merge(sea[0], sea[1], on="Season", how="outer")
sea = sea.sort_values("Season").reset_index(drop=True)

print(f"sea {sea.shape}")
# print(sea)
# print()
# sea.info()
```

    sea (41, 3)



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
res = pd.DataFrame()

for part in ["RegularSeason", "NCAATourney"]:
  for gender in ["M", "W"]:
    res_part_gender = pd.read_csv(f"{data_path}/{gender}{part}DetailedResults.csv")
    res_part_gender["Women"] = gender == "W"
    res_part_gender["Women"] = res_part_gender["Women"].astype("int32")
    res_part_gender["NCAATourney"] = part == "NCAATourney"
    res_part_gender["NCAATourney"] = res_part_gender["NCAATourney"].astype("int32")
    res = pd.concat([res, res_part_gender])

res = res.sort_values(["Season", "DayNum"]).reset_index(drop=True)

res.insert(0, "WKey", res["Season"].astype(str) + "_" + res["DayNum"].astype(str).str.zfill(3) + "_" + res["WTeamID"].astype(str) + "_" + res["LTeamID"].astype(str))
res.loc[res["LTeamID"] < res["WTeamID"], "WKey"] = res["Season"].astype(str) + "_" + res["DayNum"].astype(str).str.zfill(3) + "_" + res["LTeamID"].astype(str) + "_" + res["WTeamID"].astype(str)
res.insert(1, "LKey", res["WKey"])

res.insert(2, "WSeason", res.pop("Season"))
res.insert(3, "LSeason", res["WSeason"])

res.insert(4, "WDayNum", res.pop("DayNum"))
res.insert(5, "LDayNum", res["WDayNum"])

res.insert(7, "LTeamID", res.pop("LTeamID"))

res.insert(8, "WOppID", res["LTeamID"])
res.insert(9, "LOppID", res["WTeamID"])

res.insert(10, "WMargin", (res["WScore"] - res["LScore"]).astype("int32"))
res.insert(11, "LMargin", -res["WMargin"])

res.insert(12, "WWomen", res.pop("Women"))
res.insert(13, "LWomen", res["WWomen"])

res.insert(14, "WNCAATourney", res.pop("NCAATourney"))
res.insert(15, "LNCAATourney", res["WNCAATourney"])

res.insert(16, "WLoc", res.pop("WLoc").map({"A": -1, "N": 0, "H": 1}))
res.insert(17, "LLoc", -res["WLoc"])

res.insert(18, "WNumOT", res.pop("NumOT"))
res.insert(19, "LNumOT", res["WNumOT"])

res = pd.concat([
    res[[c for c in res if c[0] == "W"]].rename(columns={c: c[1:] for c in res if c[0] == "W"}),
    res[[c for c in res if c[0] == "L"]].rename(columns={c: c[1:] for c in res if c[0] == "L"}),
]).reset_index(drop=True)

for c in res.loc[:, "Season":"OppID"]:
  res[c] = res[c].astype("int32")

# sy = StandardScaler()
# cy = "Margin"
# res[cy] = sy.fit_transform(res[[cy]]).astype("float32")

cf = "Loc"
for c in res.loc[:, cf:]:
  res[c] = res[c].astype("float32")

res.loc[:, cf:] = StandardScaler().fit_transform(res.loc[:, cf:])

res = pd.merge(res, sea, on="Season")
res.insert(1, "Date", res["MDayZero"] + pd.to_timedelta(res["DayNum"], unit="D"))
res.loc[res["TeamID"]>=3000, "Date"] = res["WDayZero"] + pd.to_timedelta(res["DayNum"], unit="D")
res = res.drop(columns=["MDayZero", "WDayZero"])

res = pd.merge(res, tea, on="TeamID")
res = pd.merge(res, tea, left_on="OppID", right_on="TeamID", suffixes=["", "_"]).rename(columns={"TeamName_": "OppName"})
res = res.drop(columns=["TeamID_"])
res.insert(6, "TeamName", res.pop("TeamName"))
res.insert(7, "OppName", res.pop("OppName"))

res = res.sort_values(["Key", "TeamID"]).reset_index(drop=True)

print(f"res {res.shape}")
print(res)
print()
res.info()
```

    res (405732, 27)
                           Key       Date  Season  DayNum  TeamID  OppID        TeamName         OppName  Margin  Women  NCAATourney      Loc     NumOT     Score       FGM       FGA      FGM3      FGA3       FTM       FTA        OR        DR       Ast        TO       Stl       Blk        PF
    0       2003_010_1104_1328 2002-11-14    2003      10    1104   1328         Alabama        Oklahoma       6      0            0  0.00000 -0.214915  0.021993  0.566312  0.102891 -1.084275 -0.794976 -0.364918 -0.108382  0.691001  0.002141 -0.003511  1.793540 -0.013958 -1.005587  0.935628
    1       2003_010_1104_1328 2002-11-14    2003      10    1328   1104        Oklahoma         Alabama      -6      0            0  0.00000 -0.214915 -0.435726 -0.414578 -0.539242 -1.411433 -1.427122  0.475545  0.406779 -0.208418 -0.373919 -1.106682  0.755983  0.592462 -0.567787  0.494275
    2       2003_010_1272_1393 2002-11-14    2003      10    1272   1393         Memphis        Syracuse       7      0            0  0.00000 -0.214915  0.174566  0.370134  0.616597  0.551516  0.153242 -0.533011  0.020409  0.915856  0.754260  0.658391 -0.281574 -0.923589  0.307814  0.052923
    3       2003_010_1272_1393 2002-11-14    2003      10    1393   1272        Syracuse         Memphis      -7      0            0  0.00000 -0.214915 -0.359440 -0.022222  1.258731 -0.102800  0.785388 -0.701103  0.149199  2.040129  0.190171 -1.327316 -0.489085  0.289252  1.183415 -0.388430
    4       2003_011_1186_1458 2002-11-15    2003      11    1186   1458    E Washington       Wisconsin     -26      0            0 -1.05761 -0.214915 -0.969732 -0.806934 -1.438229 -1.084275 -1.269085 -0.196825 -0.237172 -1.107837 -0.373919 -1.106682  0.963494 -0.923589 -0.129986  1.597657
    ...                    ...        ...     ...     ...     ...    ...             ...             ...     ...    ...          ...      ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...
    405727  2025_132_3125_3293 2025-03-16    2025     132    3293   3125       Murray St         Belmont      21      1            0  0.00000 -0.214915  1.166291  0.762490 -0.410816  2.514465  1.417534 -0.028733 -0.494752 -1.332691  1.694410  1.761561 -0.281574 -1.530009 -0.129986 -0.609106
    405728  2025_132_3144_3456 2025-03-16    2025     132    3144   3456        Campbell  William & Mary      -3      1            0  0.00000 -0.214915 -0.359440 -0.022222 -0.282389 -0.429959  0.153242 -0.533011 -0.881123 -0.658127  1.694410 -0.224145  0.340960 -0.923589 -0.567787 -1.271135
    405729  2025_132_3144_3456 2025-03-16    2025     132    3456   3144  William & Mary        Campbell       3      1            0  0.00000 -0.214915 -0.130580  0.566312  0.745024  0.551516  0.627352 -1.541567 -1.525074 -1.332691  0.190171 -0.886047 -1.111619 -0.317169 -0.129986 -1.050459
    405730  2025_132_3192_3476 2025-03-16    2025     132    3192   3476     F Dickinson       Stonehill      17      1            0  1.05761 -0.214915 -0.130580 -0.218400 -0.282389 -1.084275  0.311279  0.643638 -0.108382 -0.208418 -0.373919 -0.444779 -1.111619  0.289252 -1.005587 -2.153840
    405731  2025_132_3192_3476 2025-03-16    2025     132    3476   3192       Stonehill     F Dickinson     -17      1            0 -1.05761 -0.214915 -1.427452 -0.610756 -0.025536 -0.757117  0.153242 -1.709659 -1.911445  0.691001 -0.373919  0.217123  0.548472 -0.923589 -1.005587 -0.167754
    
    [405732 rows x 27 columns]
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 405732 entries, 0 to 405731
    Data columns (total 27 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   Key          405732 non-null  object        
     1   Date         405732 non-null  datetime64[ns]
     2   Season       405732 non-null  int32         
     3   DayNum       405732 non-null  int32         
     4   TeamID       405732 non-null  int32         
     5   OppID        405732 non-null  int32         
     6   TeamName     405732 non-null  object        
     7   OppName      405732 non-null  object        
     8   Margin       405732 non-null  int32         
     9   Women        405732 non-null  int32         
     10  NCAATourney  405732 non-null  int32         
     11  Loc          405732 non-null  float32       
     12  NumOT        405732 non-null  float32       
     13  Score        405732 non-null  float32       
     14  FGM          405732 non-null  float32       
     15  FGA          405732 non-null  float32       
     16  FGM3         405732 non-null  float32       
     17  FGA3         405732 non-null  float32       
     18  FTM          405732 non-null  float32       
     19  FTA          405732 non-null  float32       
     20  OR           405732 non-null  float32       
     21  DR           405732 non-null  float32       
     22  Ast          405732 non-null  float32       
     23  TO           405732 non-null  float32       
     24  Stl          405732 non-null  float32       
     25  Blk          405732 non-null  float32       
     26  PF           405732 non-null  float32       
    dtypes: datetime64[ns](1), float32(16), int32(7), object(3)
    memory usage: 48.0+ MB



```python
nodes = res

# nodes = pd.DataFrame()

# even_indices = np.arange(0, len(res), 2)
# odd_indices = np.arange(1, len(res), 2)
# even_them_indices = np.minimum(even_indices + 1, len(res) - 1)
# odd_them_indices = odd_indices - 1

# nodes = pd.concat([
#     pd.concat([res.iloc[even_indices].reset_index(drop=True),
#                res.iloc[even_them_indices].add_suffix('_them').reset_index(drop=True)], axis=1),
#     pd.concat([res.iloc[odd_indices].reset_index(drop=True),
#                res.iloc[odd_them_indices].add_suffix('_them').reset_index(drop=True)], axis=1)
# ]).sort_values(["Key", "TeamID"]).reset_index(drop=True).reset_index()

# nodes = nodes.drop(columns=["Key_them", "Date_them", "Season_them", "DayNum_them", "TeamID_them", "OppID_them", "TeamName_them", "OppName_them", "Margin_them", "Loc_them", "NumOT_them"])

# print(f"nodes {nodes.shape}")
# print(nodes)
# print()
# nodes.info()

nodes.to_csv(f"{gnn_path}/nodes.csv", index=False)
```


```python
# indices_ex1 = nodes[(nodes["Season"]==2025) & (nodes["TeamID"]==1196)].index.values
# print(indices_ex1)
# print(len(indices_ex1))
# print()

# indices = nodes.groupby(['Season', 'TeamID']).indices
# indices_ex2 = indices[(2025, 1196)]
# print(indices_ex2)
# print(len(indices_ex2))
# print()

# source_ex2, target_ex2 = np.meshgrid(indices_ex2, indices_ex2)

# edges_ex2 = pd.DataFrame({
#   "source": source_ex2.flatten(),
#   "target": target_ex2.flatten(),
# })

# print(edges_ex2)

# assert edges_ex2.shape[0] == len(indices_ex2)**2
```


```python
# edges_1 = [
#   np.meshgrid(i_season_teamid, i_season_teamid)
#   for i_season_teamid in indices.values()
# ]

# edges_1 = pd.DataFrame({
#   "source": np.concatenate([s.flatten() for s, _ in edges_1]),
#   "target": np.concatenate([t.flatten() for _, t in edges_1]),
# })
```


```python
edges_team = []

for _, df in nodes.groupby(["Season", "TeamID"])[["index", "Date"]]:

  index_date = np.array(
    list(df.itertuples(index=False)),
    dtype=[("index", "int32"), ("Date", "datetime64[ns]")]
  )

  source, target = np.meshgrid(index_date, index_date)

  df = pd.concat([
      pd.DataFrame(source.flatten()).rename(columns={"index": "Source", "Date": "SourceDate"}),
      pd.DataFrame(target.flatten()).rename(columns={"index": "Target", "Date": "TargetDate"}),
  ], axis=1)

  df["Type"] = 0
  df["Delta"] = ((df["TargetDate"] - df["SourceDate"]).dt.days).astype("int32")
  df.insert(5, "Direction", df["Delta"] // np.maximum(1, np.abs(df["Delta"])))
  df["Delta"] = np.abs(df["Delta"])
  df = df.drop(columns=["SourceDate", "TargetDate"])
  edges_team.append(df)

edges_team = pd.concat(edges_team)

print(f"edges_team {edges_team.shape}")
print(edges_team)
print()

edges_team.to_csv(f"{gnn_path}/edges_team.csv", index=False)
```

    edges_team (12263084, 5)
         Source  Target  Type  Direction  Delta
    0       220     220     0          0      0
    1       492     220     0         -1      3
    2       678     220     0         -1      6
    3       920     220     0         -1      8
    4      1224     220     0         -1     12
    ..      ...     ...   ...        ...    ...
    779  401771  404547     0          1     15
    780  402145  404547     0          1     13
    781  402975  404547     0          1      8
    782  403749  404547     0          1      6
    783  404547  404547     0          0      0
    
    [12263084 rows x 5 columns]
    



```python
edges_opp = []

for _, df in nodes.groupby(["Season", "TeamID"])[["index", "Date"]]:

  index_date = np.array(
    list(df.itertuples(index=False)),
    dtype=[("index", "int32"), ("Date", "datetime64[ns]")]
  )

  index_opp = df["index"].copy()
  index_opp[index_opp % 2 == 0] += 1
  index_opp[index_opp % 2 == 1] -= 1

  df_opp = nodes.loc[index_opp, ["index", "Date"]]

  index_date_opp = np.array(
    list(df_opp.itertuples(index=False)),
    dtype=[("index", "int32"), ("Date", "datetime64[ns]")]
  )

  source, target = np.meshgrid(index_date, index_date_opp)

  df = pd.concat([
      pd.DataFrame(source.flatten()).rename(columns={"index": "Source", "Date": "SourceDate"}),
      pd.DataFrame(target.flatten()).rename(columns={"index": "Target", "Date": "TargetDate"}),
  ], axis=1)

  df["Type"] = 1
  df["Delta"] = ((df["TargetDate"] - df["SourceDate"]).dt.days).astype("int32")
  df.insert(5, "Direction", df["Delta"] // np.maximum(1, np.abs(df["Delta"])))
  df["Delta"] = np.abs(df["Delta"])
  df = df.drop(columns=["SourceDate", "TargetDate"])
  edges_opp.append(df)

edges_opp = pd.concat(edges_opp)

print(f"edges_opp {edges_opp.shape}")
print(edges_opp)
print()

edges_opp.to_csv(f"{gnn_path}/edges_opp.csv", index=False)
```

    edges_opp (12263084, 5)
         Source  Target  Type  Direction  Delta
    0       220     220     1          0      0
    1       492     220     1         -1      3
    2       678     220     1         -1      6
    3       920     220     1         -1      8
    4      1224     220     1         -1     12
    ..      ...     ...   ...        ...    ...
    779  401771  404546     1          1     15
    780  402145  404546     1          1     13
    781  402975  404546     1          1      8
    782  403749  404546     1          1      6
    783  404547  404546     1          0      0
    
    [12263084 rows x 5 columns]
    



```python
edges_sos = []

for (Season, TeamID), df in nodes.groupby(["Season", "TeamID"])[["index", "Date", "OppID"]]:

  index_date = np.array(
    [(index, Date) for index, Date, _ in df.itertuples(index=False)],
    dtype=[("index", "int32"), ("Date", "datetime64[ns]")]
  )

  df_sos = nodes.loc[
    (nodes["Season"] == Season) &
      (nodes["TeamID"].isin(df["OppID"])) &
      (nodes["OppID"] != TeamID),
    ["index", "Date"]
  ]

  index_date_sos = np.array(
    list(df_sos.itertuples(index=False)),
    dtype=[("index", "int32"), ("Date", "datetime64[ns]")]
  )

  source, target = np.meshgrid(index_date, index_date_sos)

  df = pd.concat([
      pd.DataFrame(source.flatten()).rename(columns={"index": "Source", "Date": "SourceDate"}),
      pd.DataFrame(target.flatten()).rename(columns={"index": "Target", "Date": "TargetDate"}),
  ], axis=1)

  df["Type"] = 2
  df["Delta"] = ((df["TargetDate"] - df["SourceDate"]).dt.days).astype("int32")
  df.insert(5, "Direction", df["Delta"] // np.maximum(1, np.abs(df["Delta"])))
  df["Delta"] = np.abs(df["Delta"])
  df = df.drop(columns=["SourceDate", "TargetDate"])
  edges_sos.append(df)

edges_sos = pd.concat(edges_sos)

print(f"edges_sos {edges_sos.shape}")
print(edges_sos)
print()

edges_sos.to_csv(f"{gnn_path}/edges_sos.csv", index=False)
```

    edges_sos (252999442, 5)
           Source  Target  Type  Direction  Delta
    0         220      14     2         -1      7
    1         492      14     2         -1     10
    2         678      14     2         -1     13
    3         920      14     2         -1     15
    4        1224      14     2         -1     19
    ...       ...     ...   ...        ...    ...
    15731  401771  405697     2          1     23
    15732  402145  405697     2          1     21
    15733  402975  405697     2          1     16
    15734  403749  405697     2          1     14
    15735  404547  405697     2          1      8
    
    [252999442 rows x 5 columns]
    



```python
# conv = RGATConv(
#     in_channels=30,
#     out_channels=1,
#     num_relations=3,
# )

# conv.forward(
#   x,
#   edge_index=,
#   edge_type=,
#   edge_attr=,
#   y=,
# )
```


```python

```
