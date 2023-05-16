import csv
import warnings  # 运行这个代码可以让Python不显示warnings
warnings.filterwarnings("ignore")
import math
import numpy
import numpy as np
import pandas as pd
import torch

# df=pd.read_csv('data/poi_all_deal1.csv',encoding='UTF-8')
# print(df)
# print(df.describe())
# file_path = 'data/air quality1'
# files = glob.glob(os.path.join(file_path, "*.csv"))  # 文件列表
# files.sort()  # 文件列表按名称排序
# print(files)
# df2 = pd.DataFrame()
# cnt = 0
# for file in files:
#     cnt += 1
#     if cnt > 365:
#         break
#     df = pd.read_csv(file, index_col=0)
#     df.reset_index(inplace=True)
#     df1 = df.interpolate(method='linear', axis=0)
#     df2 = df2.append(df1, ignore_index=False)
# print(df2)
#
# df3=df2[['date','hour','type','1431A','1432A','1433A','1434A','1435A','1437A','1438A']]
# print(df3)
df = pd.read_csv('data/air quality2.csv', encoding='UTF-8')
df3 = df[['date', 'hour', 'type', '1431A', '1432A', '1433A', '1434A', '1435A', '1437A', '1438A']]
# print(df3)
print(df3.describe())
# df3.to_csv("data/air quality2.csv",index=True,sep=',')
# df1=df.drop(['PM2.5_24h','PM10_24h','O3_24h','O3_8h_24h','CO_24h'],axis=0)
#
# print(df1)

df1 = pd.read_csv('data/poi_all_deal1.csv', encoding='UTF-8')
# print(df)
print(df1.describe())
LON1 = 103.497579
LON2 = 104.600359
LAT1 = 30.297549
LAT2 = 31.032446


def generalID(lon, lat, column_num, row_num):
    # 若在范围外的点，返回-1
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    # 把经度范围根据列数切割
    column = (LON2 - LON1) / column_num
    # 把纬度范围根据行数切割
    row = (LAT2 - LAT1) / row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((lon - LON1) / column) + 1 + int((lat - LAT1) / row) * column_num

alldata=[]
pointdata = pd.read_csv('data/point.csv', encoding='UTF-8')
print(pointdata)
idx = 0
for i in range(24 * 30):  # 时间范围
    lon = []
    lat = []
    PM25 = []
    PM10 = []
    O3 = []
    CO = []
    SO2 = []
    AQI = []
    NO2 = []
    for j in range(7):
        lon.append(round(float(pointdata.iat[j, 3]), 2))
        lat.append(round(float(pointdata.iat[j, 4]), 2))
        AQI.append(round(float(df3.iat[idx, 3 + j]), 1))
        PM25.append(round(float(df3.iat[idx + 1, 3 + j]), 1))
        PM10.append(round(float(df3.iat[idx + 3, 3 + j]), 1))
        SO2.append(round(float(df3.iat[idx + 5, 3 + j]), 1))
        NO2.append(round(float(df3.iat[idx + 7, 3 + j]), 1))
        O3.append(round(float(df3.iat[idx + 9, 3 + j]), 1))
        CO.append(round(float(df3.iat[idx + 12, 3 + j]), 1))
    idx = idx + 15
    if idx == 10440:
        break;

    c = {'lon': lon,
         'lat': lat,
         'AQI': AQI,
         'PM2.5': PM25,
         'PM10': PM10,
         'SO2': SO2,
         'NO2': NO2,
         'O3': O3,
         'CO': CO
         }
    print(c)
    data = numpy.zeros((15, 15, 7), dtype=float)
    for j in range(7):
        id = generalID(lon[j], lat[j], 15, 15)
        # print(id)
        x = int(id / 15)
        y = id % 15
        # print(x, y)
        data[x, y, 0] = AQI[j]
        data[x, y, 1] = PM25[j]
        data[x, y, 2] = PM10[j]
        data[x, y, 3] = SO2[j]
        data[x, y, 4] = NO2[j]
        data[x, y, 5] = O3[j]
        data[x, y, 6] = CO[j]
    # print(data)
    alldata.append(torch.from_numpy(data))
new_data = torch.stack(alldata, dim=0)
print(new_data.shape)
np.savez('data/new_data.npz', data=new_data)  # 存储张量

