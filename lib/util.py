import os.path

import pandas as pd

LON1 = 103.497579
LON2 = 104.600359
LAT1 = 30.297549
LAT2 = 31.032446


def generalID(lon, lat, column_num=15, row_num=15):
    # 若在范围外的点，返回-1
    if lon <= LON1 or lon >= LON2 or lat <= LAT1 or lat >= LAT2:
        return -1
    # 把经度范围根据列数切割
    column = (LON2 - LON1) / column_num
    # 把纬度范围根据行数切割
    row = (LAT2 - LAT1) / row_num
    # 二维矩阵坐标索引转换为一维ID，即： （列坐标区域（向下取整）+ 1） + （行坐标区域 * 列数）
    return int((lon - LON1) / column) + 1 + int((lat - LAT1) / row) * column_num


def get_air_quality_stations(path):
    pointdata = pd.read_csv(path, encoding='UTF-8')
    stations = []
    for i in range(7):
        stations.append((pointdata.iat[i, 3], pointdata.iat[i, 4]))
    return stations


def get_latlon_by_id(id):
    column = (LON2 - LON1) / 15
    row = (LAT2 - LAT1) / 15
    x = id // 15
    y = id % 15
    return LAT1 + row * x + row / 2, LON1 + column * y + column / 2


def get_id_by_idx(x, y):
    return x * 15 + y


if __name__ == '__main__':
    get_air_quality_stations()
