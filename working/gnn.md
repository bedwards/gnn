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
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch_geometric
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from google.colab import drive

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_columns", None)
pd.set_option("display.min_rows", 50)
pd.set_option("display.max_rows", 50)
pd.set_option("display.width", None)

drive_path = "/content/drive"
drive.mount(drive_path)
data_path = f"{drive_path}/My Drive/Colab Notebooks/gnn/input/march-machine-learning-mania-2025"
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
sea = pd.read_csv(f"{data_path}/MSeasons.csv", usecols=["Season", "DayZero"])
sea["DayZero"] = pd.to_datetime(sea["DayZero"])
print(f"sea {sea.shape}")
print(sea)
print()
sea.info()
```

    sea (41, 2)
        Season    DayZero
    0     1985 1984-10-29
    1     1986 1985-10-28
    2     1987 1986-10-27
    3     1988 1987-11-02
    4     1989 1988-10-31
    5     1990 1989-10-30
    6     1991 1990-10-29
    7     1992 1991-11-04
    8     1993 1992-11-02
    9     1994 1993-11-01
    10    1995 1994-10-31
    11    1996 1995-10-30
    12    1997 1996-10-28
    13    1998 1997-10-27
    14    1999 1998-10-26
    15    2000 1999-11-01
    16    2001 2000-10-30
    17    2002 2001-10-29
    18    2003 2002-11-04
    19    2004 2003-11-03
    20    2005 2004-11-01
    21    2006 2005-10-31
    22    2007 2006-10-30
    23    2008 2007-11-05
    24    2009 2008-11-03
    25    2010 2009-11-02
    26    2011 2010-11-01
    27    2012 2011-10-31
    28    2013 2012-11-05
    29    2014 2013-11-04
    30    2015 2014-11-03
    31    2016 2015-11-02
    32    2017 2016-10-31
    33    2018 2017-10-30
    34    2019 2018-11-05
    35    2020 2019-11-04
    36    2021 2020-11-02
    37    2022 2021-11-01
    38    2023 2022-10-31
    39    2024 2023-11-06
    40    2025 2024-11-04
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41 entries, 0 to 40
    Data columns (total 2 columns):
     #   Column   Non-Null Count  Dtype         
    ---  ------   --------------  -----         
     0   Season   41 non-null     int64         
     1   DayZero  41 non-null     datetime64[ns]
    dtypes: datetime64[ns](1), int64(1)
    memory usage: 788.0 bytes



```python
tea = pd.read_csv(f"{data_path}/MTeams.csv", usecols=["TeamID", "TeamName"])
print(f"tea {tea.shape}")
print(tea)
```

    tea (380, 2)
         TeamID          TeamName
    0      1101       Abilene Chr
    1      1102         Air Force
    2      1103             Akron
    3      1104           Alabama
    4      1105       Alabama A&M
    5      1106        Alabama St
    6      1107       SUNY Albany
    7      1108         Alcorn St
    8      1109      Alliant Intl
    9      1110     American Univ
    10     1111    Appalachian St
    11     1112           Arizona
    12     1113        Arizona St
    13     1114   Ark Little Rock
    14     1115    Ark Pine Bluff
    15     1116          Arkansas
    16     1117       Arkansas St
    17     1118      Armstrong St
    18     1119              Army
    19     1120            Auburn
    20     1121           Augusta
    21     1122       Austin Peay
    22     1123           Ball St
    23     1124            Baylor
    24     1125           Belmont
    ..      ...               ...
    355    1456    William & Mary
    356    1457          Winthrop
    357    1458         Wisconsin
    358    1459           Wofford
    359    1460         Wright St
    360    1461           Wyoming
    361    1462            Xavier
    362    1463              Yale
    363    1464     Youngstown St
    364    1465       Cal Baptist
    365    1466     North Alabama
    366    1467         Merrimack
    367    1468        Bellarmine
    368    1469         Utah Tech
    369    1470       Tarleton St
    370    1471      UC San Diego
    371    1472      St Thomas MN
    372    1473        Lindenwood
    373    1474         Queens NC
    374    1475  Southern Indiana
    375    1476         Stonehill
    376    1477    East Texas A&M
    377    1478          Le Moyne
    378    1479        Mercyhurst
    379    1480      West Georgia
    
    [380 rows x 2 columns]



```python
res = pd.DataFrame()

for type_ in ["RegularSeason", "NCAATourney"]:
  res = pd.concat([res, pd.read_csv(f"{data_path}/M{type_}DetailedResults.csv")])

res = res.reset_index(drop=True)

res.insert(0, "WID", res["Season"].astype(str) + "_" + res["DayNum"].astype(str).str.zfill(3) + "_" + res["WTeamID"].astype(str) + "_" + res["LTeamID"].astype(str))
res.loc[res["WTeamID"] < res["LTeamID"], "WID"] = res["Season"].astype(str) + "_" + res["DayNum"].astype(str).str.zfill(3) + "_" + res["LTeamID"].astype(str) + "_" + res["WTeamID"].astype(str)
res.insert(1, "LID", res["WID"])

res.insert(2, "WSeason", res.pop("Season"))
res.insert(3, "LSeason", res["WSeason"])

res.insert(4, "WDayNum", res.pop("DayNum"))
res.insert(5, "LDayNum", res["WDayNum"])

res.insert(7, "LTeamID", res.pop("LTeamID"))

res.insert(8, "WOppID", res["LTeamID"])
res.insert(9, "LOppID", res["WTeamID"])

res.insert(10, "WMargin", res["WScore"] - res["LScore"])
res.insert(11, "LMargin", -res["WMargin"])

res.insert(12, "WLoc", res.pop("WLoc").map({"A": -1, "N": 0, "H": 1}))
res.insert(13, "LLoc", -res["WLoc"])

res.insert(14, "WNumOT", res.pop("NumOT"))
res.insert(15, "LNumOT", res["WNumOT"])

res = pd.concat([
    res[[c for c in res if c[0] == "W"]].rename(columns={c: c[1:] for c in res if c[0] == "W"}),
    res[[c for c in res if c[0] == "L"]].rename(columns={c: c[1:] for c in res if c[0] == "L"}),
]).sort_values("ID").reset_index(drop=True)

for c in res.loc[:, "Season":"OppID"]:
  res[c] = res[c].astype("int32")

sy = StandardScaler()
cy = "Margin"
res[cy] = sy.fit_transform(res[[cy]]).astype("float32")

cf = "Loc"
for c in res.loc[:, cf:]:
  res[c] = res[c].astype("float32")

res.loc[:, cf:] = StandardScaler().fit_transform(res.loc[:, cf:])

res = pd.merge(res, sea, on="Season")
res.insert(1, "Date", res['DayZero'] + pd.to_timedelta(res["DayNum"], unit="D"))
res = res.drop(columns=["Season", "DayZero", "DayNum"])

res = pd.merge(res, tea, on="TeamID")
res = pd.merge(res, tea, left_on="OppID", right_on="TeamID", suffixes=["", "_"]).rename(columns={"TeamName_": "OppName"})
res.insert(4, "TeamName", res.pop("TeamName"))
res.insert(5, "OppName", res.pop("OppName"))

print(f"res {res.shape}")
print(res)
print()
res.info()
```

    res (240528, 24)
                            ID       Date  TeamID  OppID        TeamName         OppName    Margin       Loc     NumOT     Score       FGM       FGA      FGM3      FGA3       FTM       FTA        OR        DR       Ast        TO       Stl       Blk        PF  TeamID_
    0       2003_010_1328_1104 2002-11-14    1104   1328         Alabama        Oklahoma  0.397053  0.000000 -0.225257 -0.153137  0.490802  0.226886 -1.252690 -0.951573 -0.476284 -0.233940  0.862797  0.073893 -0.012935  2.304793  0.173707 -1.023865  0.846855     1328
    1       2003_010_1328_1104 2002-11-14    1328   1104        Oklahoma         Alabama -0.397053  0.000000 -0.225257 -0.634959 -0.537384 -0.438796 -1.583592 -1.611310  0.346381  0.273258 -0.095526 -0.317623 -1.148182  1.129768  0.848504 -0.583745  0.399831     1104
    2       2003_010_1393_1272 2002-11-14    1393   1272        Syracuse         Memphis -0.463229  0.000000 -0.225257 -0.554655 -0.126110  1.425113 -0.259983  0.697769 -0.805351  0.019659  2.300281  0.269650 -1.375231 -0.280261  0.511106  1.176735 -0.494217     1272
    3       2003_010_1393_1272 2002-11-14    1272   1393         Memphis        Syracuse  0.463229  0.000000 -0.225257  0.007471  0.285165  0.759431  0.401821  0.038032 -0.640818 -0.107141  1.102378  0.856923  0.668213 -0.045256 -0.838489  0.296495 -0.047193     1393
    4       2003_011_1400_1208 2002-11-15    1208   1400         Georgia           Texas -0.397053  0.000000 -0.225257  0.087774 -0.126110  0.759431 -0.259983 -0.621705  0.510914  0.907256  2.539862 -1.687925 -0.239984 -0.750271  0.173707 -1.023865 -0.941241     1400
    5       2003_011_1400_1208 2002-11-15    1400   1208           Texas         Georgia  0.397053  0.000000 -0.225257  0.569596  1.107713  0.626295 -0.259983 -0.951573 -0.476284 -0.867938  1.581539 -0.317623 -0.239984  0.189749 -0.838489  0.296495  0.399831     1208
    6       2003_011_1437_1266 2002-11-15    1266   1437       Marquette       Villanova  0.794106  0.000000 -0.225257  0.248382 -0.126110  0.226886  0.401821 -0.291836  0.510914  1.160855  1.581539  0.465408  0.441164 -0.750271 -0.501090 -0.583745  1.517391     1437
    7       2003_011_1437_1266 2002-11-15    1437   1266       Villanova       Marquette -0.794106  0.000000 -0.225257 -0.715262 -0.537384  2.223930 -1.252690  1.027637  0.017315  0.400057  4.935669 -0.317623 -0.921132 -0.280261 -1.513286  0.736615  1.070367     1266
    8       2003_011_1457_1296 2002-11-15    1457   1296        Winthrop      N Illinois -0.397053  0.000000 -0.225257 -1.598603 -1.359932 -0.971341 -0.259983  0.367900 -0.969884 -0.614339  1.581539 -0.709137 -0.921132  1.364774 -0.838489 -0.143625  1.070367     1296
    9       2003_011_1457_1296 2002-11-15    1296   1457      N Illinois        Winthrop  0.397053  0.000000 -0.225257 -1.116781 -1.359932 -2.435840 -1.252690 -1.776244  0.510914  1.414454 -1.053849 -0.904895 -0.467034 -0.280261  2.535497 -0.583745 -0.047193     1457
    10      2003_011_1458_1186 2002-11-15    1458   1186       Wisconsin    E Washington  1.720564  1.062083 -0.225257  0.890811  0.285165  0.093750 -0.259983 -1.281442  1.498112  0.907256  0.383635  0.073893 -0.239984 -0.985276  0.848504 -0.143625 -0.047193     1186
    11      2003_011_1458_1186 2002-11-15    1186   1458    E Washington       Wisconsin -1.720564 -1.062083 -0.225257 -1.197084 -0.948658 -1.370750 -1.252690 -1.446376 -0.311751 -0.360740 -1.053849 -0.317623 -1.148182  1.364774 -0.838489 -0.143625  1.517391     1458
    12      2003_012_1194_1156 2002-11-16    1194   1156     FL Atlantic    Cleveland St  0.330878  0.000000 -0.225257  0.087774  0.696439  0.226886 -0.590885 -1.446376 -0.640818 -0.233940 -0.335107 -0.317623 -0.921132  0.894764  0.848504 -0.583745  1.070367     1156
    13      2003_012_1194_1156 2002-11-16    1156   1194    Cleveland St     FL Atlantic -0.330878  0.000000 -0.225257 -0.313744 -0.126110 -0.571932 -0.259983 -0.291836 -0.311751  0.907256  0.623216  0.465408 -0.012935  2.774803  0.511106 -0.583745 -0.047193     1194
    14      2003_012_1236_1161 2002-11-16    1161   1236     Colorado St             PFW  1.191160  1.062083 -0.225257  0.810507 -0.331747 -0.172523 -1.583592 -1.941178  2.978909  2.428851  0.623216 -1.100653  0.214114  0.894764  1.523301 -1.023865  1.517391     1236
    15      2003_012_1236_1161 2002-11-16    1236   1161             PFW     Colorado St -1.191160 -1.062083 -0.225257 -0.634959 -1.154295 -2.036431 -0.921788 -0.786639  1.004513  1.034055 -0.335107 -0.513380 -0.467034  3.949829  1.185903  0.296495  2.187927     1161
    16      2003_012_1457_1186 2002-11-16    1186   1457    E Washington        Winthrop  0.926457  0.000000 -0.225257  0.408989  0.696439  0.759431 -0.921788 -0.951573  0.181848  0.146458  0.623216  2.227225  1.349361  1.364774  0.173707 -0.583745  0.623343     1457
    17      2003_012_1457_1186 2002-11-16    1457   1186        Winthrop    E Washington -0.926457  0.000000 -0.225257 -0.715262 -0.948658  0.360022 -0.921788 -0.456771  0.510914  0.400057 -0.574688  0.269650 -0.694083  0.424754  2.535497  2.056974 -0.047193     1186
    18      2003_012_1458_1296 2002-11-16    1458   1296       Wisconsin      N Illinois  1.852915  1.062083 -0.225257  1.131722  1.518987  1.425113 -0.590885 -0.456771  0.181848 -0.107141  0.862797 -0.317623 -0.467034 -1.690291  1.860700 -1.463985 -1.164753     1296
    19      2003_012_1458_1296 2002-11-16    1296   1458      N Illinois       Wisconsin -1.852915 -1.062083 -0.225257 -1.116781 -0.331747 -0.571932 -1.252690 -0.951573 -1.134417 -0.994738 -0.335107 -0.121865 -0.694083  1.129768 -1.850684 -0.143625 -0.047193     1458
    20      2003_013_1202_1106 2002-11-17    1106   1202      Alabama St          Furman -0.066176  0.000000 -0.225257  0.248382  0.902076  0.892568  1.063626  0.367900 -1.463483 -1.882335  0.623216 -1.492168  0.441164 -0.280261 -0.163691 -0.583745 -1.388265     1202
    21      2003_013_1202_1106 2002-11-17    1202   1106          Furman      Alabama St  0.066176  0.000000 -0.225257  0.328685  0.902076 -0.705068  0.070919 -1.116507 -0.805351 -1.121537 -1.053849 -0.513380  1.122311  0.424754  0.173707 -1.023865 -2.952849     1106
    22      2003_013_1237_1135 2002-11-17    1135   1237           Brown           IUPUI -0.066176  0.000000 -0.225257 -0.394048 -0.126110 -0.039387 -0.259983 -0.126902 -0.476284 -0.360740  0.862797 -0.513380  0.895262  1.129768  0.511106  0.296495 -1.164753     1237
    23      2003_013_1237_1135 2002-11-17    1237   1135           IUPUI           Brown  0.066176  0.000000 -0.225257 -0.313744  0.285165  1.291976 -0.590885 -0.126902 -0.805351 -0.867938  2.539862 -0.121865  0.441164  0.894764  1.860700 -0.143625 -0.270705     1135
    24      2003_013_1323_1125 2002-11-17    1125   1323         Belmont      Notre Dame -1.852915 -1.062083 -0.225257 -1.759210 -1.359932  1.025704  0.401821  0.697769 -1.628016 -1.501936  0.862797  0.465408 -0.239984  0.894764  1.185903 -1.463985 -0.270705     1323
    ...                    ...        ...     ...    ...             ...             ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...       ...      ...
    240503  2025_131_1397_1120 2025-03-15    1397   1120       Tennessee          Auburn  0.330878  0.000000 -0.225257  0.007471 -0.948658 -1.237613 -0.590885 -0.126902  1.827178  0.907256 -0.095526 -0.317623 -1.375231 -0.750271 -0.501090 -0.583745 -0.047193     1120
    240504  2025_131_1408_1272 2025-03-15    1408   1272          Tulane         Memphis -0.066176  0.000000 -0.225257  0.569596  0.490802  0.493159  0.732723  1.522440  0.017315  0.146458 -1.293430  0.269650  0.668213 -0.280261  0.848504  0.296495  1.964415     1272
    240505  2025_131_1408_1272 2025-03-15    1272   1408         Memphis          Tulane  0.066176  0.000000 -0.225257  0.649900  0.079527  1.158840 -0.590885  0.532834  1.498112  2.302051  2.300281  0.073893  0.668213 -0.045256  0.848504  0.736615 -0.270705     1408
    240506  2025_131_1412_1317 2025-03-15    1317   1412     North Texas             UAB -0.661755  0.000000 -0.225257 -1.116781 -0.948658  0.759431 -0.921788 -0.291836 -0.311751 -0.487540  1.581539 -0.709137 -0.921132 -0.985276 -0.163691 -0.583745 -0.941241     1412
    240507  2025_131_1412_1317 2025-03-15    1412   1317             UAB     North Texas  0.661755  0.000000 -0.225257 -0.313744 -0.126110 -0.039387 -0.921788 -0.621705  0.017315 -0.107141  0.623216 -0.513380 -0.012935 -1.220281 -0.838489  0.736615 -0.941241     1317
    240508  2025_131_1430_1213 2025-03-15    1430   1213     Utah Valley    Grand Canyon -0.463229  1.062083 -0.225257  0.971115  0.490802  0.226886  0.732723 -0.126902  0.839980  1.034055  0.144054 -1.100653 -0.012935 -0.515266 -1.175887 -1.023865  1.517391     1213
    240509  2025_131_1430_1213 2025-03-15    1213   1430    Grand Canyon     Utah Valley  0.463229 -1.062083 -0.225257  1.533240  0.696439 -0.305659 -0.259983 -0.456771  2.156244  1.668053 -1.053849 -0.709137 -0.694083 -1.220281  0.173707  1.616855  0.846855     1430
    240510  2025_131_1433_1260 2025-03-15    1260   1433  Loyola-Chicago             VCU -0.463229  0.000000 -0.225257 -1.197084 -0.948658  1.558249 -0.590885  0.862703 -0.640818 -0.487540  1.821120 -0.121865 -0.694083 -0.750271  0.173707  0.296495 -0.047193     1433
    240511  2025_131_1433_1260 2025-03-15    1433   1260             VCU  Loyola-Chicago  0.463229  0.000000 -0.225257 -0.634959 -0.537384 -0.705068 -0.259983  0.697769 -0.311751  0.146458 -0.574688  0.856923 -0.467034 -0.750271 -0.163691  1.176735 -0.494217     1260
    240512  2025_131_1458_1277 2025-03-15    1458   1277       Wisconsin     Michigan St  0.198527  0.000000 -0.225257  0.569596 -0.331747 -0.305659  0.732723  1.522440  1.333579  1.034055 -1.293430  0.661165  0.441164 -1.455286 -1.175887  0.296495  0.623343     1277
    240513  2025_131_1458_1277 2025-03-15    1277   1458     Michigan St       Wisconsin -0.198527  0.000000 -0.225257  0.328685 -0.126110  0.360022  0.070919 -0.126902  0.839980  0.526857 -0.574688  0.856923 -0.012935 -1.455286 -1.850684 -0.583745  0.399831     1458
    240514  2025_131_1463_1343 2025-03-15    1463   1343            Yale       Princeton  0.132351  0.000000 -0.225257 -0.875870 -0.537384 -0.971341 -1.252690 -0.786639 -0.311751 -0.233940 -0.814269  0.465408 -1.375231 -1.220281 -1.175887 -0.143625 -2.058801     1343
    240515  2025_131_1463_1343 2025-03-15    1343   1463       Princeton            Yale -0.132351  0.000000 -0.225257 -1.036477 -0.537384  0.093750  1.063626  2.512045 -1.792549 -2.135934 -1.293430 -0.709137  0.214114 -1.455286 -0.501090 -0.583745 -0.270705     1463
    240516  2025_131_1471_1414 2025-03-15    1414   1471       UC Irvine    UC San Diego -0.926457  0.000000 -0.225257 -0.715262 -0.948658  0.093750  0.401821  1.522440 -0.147218 -0.867938 -1.533011 -0.513380  0.214114 -0.280261  0.848504  0.296495 -1.388265     1471
    240517  2025_131_1471_1414 2025-03-15    1471   1414    UC San Diego       UC Irvine  0.926457  0.000000 -0.225257  0.408989  0.490802 -0.172523  1.725430  1.522440 -0.805351 -0.867938 -1.293430  0.661165  2.030509 -0.280261  0.511106 -0.143625 -1.164753     1414
    240518  2025_132_1397_1196 2025-03-16    1397   1196       Tennessee         Florida -0.595580  0.000000 -0.225257  0.569596 -0.537384 -0.971341  0.401821  1.027637  1.827178  1.541254 -1.293430 -0.709137  0.441164 -0.985276  0.173707  0.736615  0.623343     1196
    240519  2025_132_1397_1196 2025-03-16    1196   1397         Florida       Tennessee  0.595580  0.000000 -0.225257  1.292329  0.285165  0.626295  0.732723  1.192571  1.827178  1.034055  1.102378  0.073893 -0.467034 -1.220281 -0.163691 -1.023865  1.517391     1397
    240520  2025_132_1412_1272 2025-03-16    1412   1272             UAB         Memphis -0.794106  0.000000 -0.225257  0.168078 -0.126110  1.691385  0.070919  1.192571  0.510914  0.400057  2.539862 -0.121865 -0.694083  0.424754 -0.501090  0.296495  0.623343     1272
    240521  2025_132_1412_1272 2025-03-16    1272   1412         Memphis             UAB  0.794106  0.000000 -0.225257  1.131722  1.518987  1.558249 -0.259983 -1.116507  0.017315  0.400057  1.821120  0.661165 -0.012935 -0.280261  1.185903  1.616855 -0.047193     1412
    240522  2025_132_1433_1206 2025-03-16    1206   1433    George Mason             VCU -0.330878  0.000000 -0.225257 -0.554655 -1.154295 -0.039387  0.732723  0.202966  0.346381 -0.107141  0.862797 -0.904895 -0.012935 -0.515266  1.523301 -0.143625  0.846855     1433
    240523  2025_132_1433_1206 2025-03-16    1433   1206             VCU    George Mason  0.330878  0.000000 -0.225257 -0.153137 -1.154295 -1.237613  0.401821  0.862703  1.333579  0.907256  0.144054  0.269650 -0.694083  0.659759  0.173707  2.497094  0.399831     1206
    240524  2025_132_1458_1276 2025-03-16    1458   1276       Wisconsin        Michigan -0.397053  0.000000 -0.225257 -1.357692 -1.976844  1.558249  0.070919  3.171781  0.346381 -0.233940  1.102378  1.444195 -1.148182 -1.690291 -1.175887 -0.583745 -0.941241     1276
    240525  2025_132_1458_1276 2025-03-16    1276   1458        Michigan       Wisconsin  0.397053  0.000000 -0.225257 -0.875870 -1.154295  0.360022  0.732723  2.182176 -0.311751 -0.867938 -1.053849  2.031468  0.214114 -1.220281 -1.175887 -0.583745 -0.494217     1458
    240526  2025_132_1463_1165 2025-03-16    1463   1165            Yale         Cornell  0.397053  0.000000 -0.225257  1.613544  1.313350  0.759431  2.056332  0.532834  0.181848  0.146458 -0.814269 -0.709137  0.441164 -2.395306 -0.501090 -1.463985 -0.270705     1165
    240527  2025_132_1463_1165 2025-03-16    1165   1463         Cornell            Yale -0.397053  0.000000 -0.225257  1.131722  0.902076 -0.571932  1.394528  1.192571  0.181848 -0.487540 -1.533011  0.465408  0.441164 -0.515266 -1.513286 -0.583745 -0.494217     1463
    
    [240528 rows x 24 columns]
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 240528 entries, 0 to 240527
    Data columns (total 24 columns):
     #   Column    Non-Null Count   Dtype         
    ---  ------    --------------   -----         
     0   ID        240528 non-null  object        
     1   Date      240528 non-null  datetime64[ns]
     2   TeamID    240528 non-null  int32         
     3   OppID     240528 non-null  int32         
     4   TeamName  240528 non-null  object        
     5   OppName   240528 non-null  object        
     6   Margin    240528 non-null  float32       
     7   Loc       240528 non-null  float32       
     8   NumOT     240528 non-null  float32       
     9   Score     240528 non-null  float32       
     10  FGM       240528 non-null  float32       
     11  FGA       240528 non-null  float32       
     12  FGM3      240528 non-null  float32       
     13  FGA3      240528 non-null  float32       
     14  FTM       240528 non-null  float32       
     15  FTA       240528 non-null  float32       
     16  OR        240528 non-null  float32       
     17  DR        240528 non-null  float32       
     18  Ast       240528 non-null  float32       
     19  TO        240528 non-null  float32       
     20  Stl       240528 non-null  float32       
     21  Blk       240528 non-null  float32       
     22  PF        240528 non-null  float32       
     23  TeamID_   240528 non-null  int64         
    dtypes: datetime64[ns](1), float32(17), int32(2), int64(1), object(3)
    memory usage: 26.6+ MB



```python
device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_kwargs = dict(dtype=torch.float32, device=device)
x = torch.tensor(res.loc[:, "Loc":].values, **tensor_kwargs)
print(f"x {x.shape}")
print(x)
```

    x torch.Size([240528, 17])
    tensor([[ 0.0000e+00, -2.2526e-01, -1.5314e-01,  ..., -1.0239e+00,
              8.4685e-01,  1.3280e+03],
            [ 0.0000e+00, -2.2526e-01, -6.3496e-01,  ..., -5.8375e-01,
              3.9983e-01,  1.1040e+03],
            [ 0.0000e+00, -2.2526e-01, -5.5466e-01,  ...,  1.1767e+00,
             -4.9422e-01,  1.2720e+03],
            ...,
            [ 0.0000e+00, -2.2526e-01, -8.7587e-01,  ..., -5.8375e-01,
             -4.9422e-01,  1.4580e+03],
            [ 0.0000e+00, -2.2526e-01,  1.6135e+00,  ..., -1.4640e+00,
             -2.7071e-01,  1.1650e+03],
            [ 0.0000e+00, -2.2526e-01,  1.1317e+00,  ..., -5.8375e-01,
             -4.9422e-01,  1.4630e+03]])



```python
# Get team-date index mapping
team_date_idx = dict(zip(zip(res['TeamID'], res['Date']), range(len(res))))

# Create unidirectional temporal edges (later game -> earlier game)
df_team_time = res[['TeamID', 'Date']].sort_values(['TeamID', 'Date'])
df_team_time['curr_idx'] = df_team_time.index
df_team_time['prev_idx'] = df_team_time.groupby('TeamID')['curr_idx'].shift(1)  # Previous game
df_team_time['next_idx'] = df_team_time.groupby('TeamID')['curr_idx'].shift(-1)  # Next game

# Create temporal edges (from current to previous AND from next to current)
temporal_edges_prev = df_team_time.dropna(subset=['prev_idx'])[['curr_idx', 'prev_idx']].values.astype(int)
temporal_edges_next = df_team_time.dropna(subset=['next_idx'])[['next_idx', 'curr_idx']].values.astype(int)
temporal_edges = np.vstack([temporal_edges_prev, temporal_edges_next])

# Create bidirectional game edges
game_edges = []
for nodes in game_to_nodes.values():
    if len(nodes) == 2:
        game_edges.extend([(nodes[0], nodes[1]), (nodes[1], nodes[0])])

# Combine all edges and create edge attributes
game_edges = np.array(game_edges)
edge_index = torch.tensor(np.vstack([temporal_edges, game_edges]).T, **tensor_kwargs)
edge_attr = torch.cat([
    torch.zeros(len(temporal_edges), 1, **tensor_kwargs),
    torch.ones(len(game_edges), 1, **tensor_kwargs)
], dim=0)

# Add time feature from Date
time = torch.tensor((res['Date'] - res['Date'].min()).dt.total_seconds().values, **tensor_kwargs).unsqueeze(1)

# Create kwargs with ID, TeamID, and OppID
kwargs = {
    'id_lookup': res['ID'].tolist(),  # Keep as list for indexing
    'team_id': torch.tensor(res['TeamID'].values, dtype=torch.int32),  # Keep as tensor
    'opp_id': torch.tensor(res['OppID'].values, dtype=torch.int32),  # Keep as tensor
    'res_df': res  # Store reference to original DataFrame
}

# Create final Data object
y = torch.tensor(res['Margin'].values, **tensor_kwargs).unsqueeze(1)
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, time=time, **kwargs)

# Print number of nodes
print(f"Number of nodes: {data.num_nodes}")

# Print number of edges
print(f"Number of edges: {data.num_edges}")

# Print shape of each attribute
print(f"x shape: {data.x.shape}")
print(f"edge_index shape: {data.edge_index.shape}")
print(f"edge_attr shape: {data.edge_attr.shape}")
print(f"y shape: {data.y.shape}")
print(f"time shape: {data.time.shape}")

# Print other properties
print(f"All keys: {data.keys()}")
```

    Number of nodes: 240528
    Number of edges: 720842
    x shape: torch.Size([240528, 17])
    edge_index shape: torch.Size([2, 720842])
    edge_attr shape: torch.Size([720842, 1])
    y shape: torch.Size([240528, 1])
    time shape: torch.Size([240528, 1])
    All keys: ['id_lookup', 'edge_index', 'edge_attr', 'team_id', 'res_df', 'y', 'opp_id', 'time', 'x']



```python
def visualize_team_season(data, team_id=1345, season_year=2024):
    # Get tensor data
    team_ids = data.team_id.cpu().numpy()
    times = data.time.cpu().numpy().flatten()

    # Create index lookup
    indices = np.where(team_ids == team_id)[0]

    # Use these indices to get data from original DataFrame
    team_games = data.res_df.iloc[indices].copy()

    # Continue with date filtering and visualization
    season_start = datetime(season_year-1, 11, 1)
    season_end = datetime(season_year, 4, 30)
    team_games = team_games[(team_games['Date'] >= season_start) &
                            (team_games['Date'] <= season_end)]

    # Sort by date
    team_games = team_games.sort_values('Date')

    team_name = team_games['TeamName'].iloc[0]

    if len(team_games) == 0:
        return None, f"No games found for {team_name} in {season_year-1}-{season_year} season"

    # Unnormalize margins if needed (using inverse_transform)
    # team_games['Margin'] = sy.inverse_transform(team_games[['Margin']])

    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot margins
    wins = team_games['Margin'] > 0
    losses = ~wins

    ax.scatter(team_games.loc[wins, 'Date'], team_games.loc[wins, 'Margin'],
               color='green', s=80, label='Win')
    ax.scatter(team_games.loc[losses, 'Date'], team_games.loc[losses, 'Margin'],
               color='red', s=80, label='Loss')

    # Add horizontal line at margin=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Format plot
    ax.set_title(f'Team {team_name} Games in {season_year-1}-{season_year} Season', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Margin (Normalized)', fontsize=12)

    # Format dates on x-axis
    date_fmt = mdates.DateFormatter('%b %d')
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    # Annotate with opponent names but limit text length
    for _, game in team_games.iterrows():
        # Truncate opponent name if too long
        opp_name = game['OppName']
        if len(opp_name) > 12:
            opp_name = opp_name[:10] + '..'

        # Position annotation based on margin
        y_offset = 5 if game['Margin'] > 0 else -10

        ax.annotate(f"vs. {opp_name}",
                   (game['Date'], game['Margin']),
                   xytext=(0, y_offset),
                   textcoords='offset points',
                   ha='center', fontsize=8,
                   rotation=90)

    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, f"Created visualization for {team_name} in {season_year-1}-{season_year} season"

# Call the function with Purdue's team ID
team_id = 1345  # Purdue
fig, msg = visualize_team_season(data, team_id=team_id, season_year=2024)
plt.show()
```


    
![png](gnn_files/gnn_7_0.png)
    



```python

```
