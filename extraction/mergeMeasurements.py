import os
import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime as dt
from extractBuoy import download

def create(begin: int):

    # Prepare nc file

    ds = nc.Dataset('./Data/measurements.nc', mode='w', format='NETCDF4')
    ds.description = 'SoDa file and buoy data merged'
    ds.loc = 'Site latitude (positive means North) : 43.38 ; site longitude (positive means East) : 7.83'

    # Dimensions

    ds.createDimension('time', size=24)
    ds.createDimension('date', size=153*(2022 - begin))

    date = ds.createVariable('date', 'str', dimensions=('date'))
    date.units = 'YY-MM-DD'
    time = ds.createVariable('time', 'int', dimensions=('time'))
    time.units = 'h'

    # SoDa variables

    ghi = ds.createVariable('ghi', 'float', dimensions=('date', 'time'))
    ghi.units = 'W/m^2'
    ghi.long_name = 'Global horizontal irradiance (satellite)'
    cs = ds.createVariable('cs', 'float', dimensions=('date', 'time'))
    cs.units = 'W/m^2'
    cs.long_name = 'Clear-sky GHI (model)'
    t_rean = ds.createVariable('t_rean', 'float', dimensions=('date', 'time'))
    t_rean.units = 'K'
    t_rean.long_name = '2 meters above the ground temperature (reanalyses)'
    ws_rean = ds.createVariable('ws_rean', 'float', dimensions=('date', 'time'))
    ws_rean.units = 'm/s'
    ws_rean.long_name = '10 meters above the ground wind speed (reanalyses)'
    wd_rean = ds.createVariable('wd_rean', 'float', dimensions=('date', 'time'))
    wd_rean.units = '° (0° => North, 90° => East...)'
    wd_rean.long_name = '10 meters above the ground wind direction (reanalyses)'

    # Buoy variables

    t = ds.createVariable('t', 'float', dimensions=('date', 'time'))
    t.units = 'K'
    t.long_name = '2 meters above the ground temperature'
    ws = ds.createVariable('ws', 'float', dimensions=('date', 'time'))
    ws.units = 'm/s'
    ws.long_name = '10 meters above the ground wind speed'
    wd = ds.createVariable('wd', 'float', dimensions=('date', 'time'))
    wd.units = '° (0° => North, 90° => East...)'
    wd.long_name = '10 meters above the ground wind direction'
    shww = ds.createVariable('shww', 'float', dimensions=('date', 'time'))
    shww.units = 'm'
    shww.long_name = 'Wind waves significant height'
    mpww = ds.createVariable('mpww', 'float', dimensions=('date', 'time'))
    mpww.units = 's'
    mpww.long_name = 'Wind waves mean period'

    # Copernicus variables

    shww_rean = ds.createVariable('shww_rean', 'float', dimensions=('date', 'time'))
    shww_rean.units = 'm'
    shww_rean.long_name = 'Wind waves significant height (reanalyses)'
    mpww_rean = ds.createVariable('mpww_rean', 'float', dimensions=('date', 'time'))
    mpww_rean.units = 's'
    mpww_rean.long_name = 'Wind waves mean period (reanalyses)'

    return ds

def addSoDa(ds: nc.Dataset, begin: int):

    # Read file

    df = pd.read_csv(
        './Data/SoDa/SoDa_buoy.csv',
        sep=';',
        comment='#',
        usecols=[
            'Date',
            'Time',
            'Global Horiz',
            'Clear-Sky',
            'Temperature',
            'Wind speed',
            'Wind direction'
        ]
    )

    # Removing unnecessary elements

    dates = pd.DatetimeIndex(df['Date'])
    df.drop(
        labels=df[(dates.year < begin) | (dates.year > 2021) | (dates.month < 5) | (dates.month > 9)].index,
        axis=0,
        inplace=True
    )
    dates = np.array(df['Date'], dtype='str')

    # Fill nc file

    newshape = (len(df.index) // 24, 24)
    ds['date'][:] = dates[::24]
    ds['time'][:] = np.arange(1, 25)
    ds['ghi'][:] = np.array(df['Global Horiz']).reshape(newshape)
    ds['cs'][:] = np.array(df['Clear-Sky']).reshape(newshape)
    ds['t_rean'][:] = np.array(df['Temperature']).reshape(newshape)
    ds['ws_rean'][:] = np.array(df['Wind speed']).reshape(newshape)
    ds['wd_rean'][:] = np.array(df['Wind direction']).reshape(newshape)

def oneMonthBuoy(year, month):

    duration = {5: 31, 6: 30, 7: 31, 8: 31, 9: 30}
    hADay = set(range(24))

    # Read file 

    fileName = f'./Data/Buoy/marine.{year}0{month}.csv.gz'
    db = pd.read_csv(
        fileName, 
        sep=';',
        usecols=['numer_sta', 'date', 't', 'dd', 'ff', 'HwaHwa', 'PwaPwa'],
        compression='gzip'
    )

    # Clean file

    right_sta = pd.Series(db['numer_sta'], dtype='str').str.fullmatch(r'6100(00)?1')
    db = db[right_sta]
    # db.drop(
    #     labels=db[np.array(db['numer_sta'], dtype='str') != '6100001'].index,
    #     axis=0,
    #     inplace=True
    # )

    # Date format

    buoyDate, buoyHour = [], []
    for date in np.array(db['date'], dtype='str'):
        assert date[:4] == f'{year}'
        assert date[4:6] == f'0{month}'
        buoyDate.append(f'{year}-0{month}-' + date[6:8])
        buoyHour.append(date[8:10])
    db['date'] = np.array(buoyDate, dtype='str')
    db['hour'] = np.array(buoyHour, dtype='int')
    
    db[db[['t', 'dd', 'ff', 'HwaHwa', 'PwaPwa']].isin(['mq'])] = np.nan

    db = db.astype(
        dtype={
            't': 'float',
            'dd': 'float',
            'ff': 'float',
            'HwaHwa': 'float',
            'PwaPwa': 'float'
        }
    )

    db = db.groupby(['date', 'hour']).mean().reset_index()

    # Complete gaps in file
    for d in range(1, duration[month] + 1):
        date = f'{year}-0{month}-' + '0'*(1 - int(d > 9)) + str(d)
        hours = set(db.loc[db['date'] == date, 'hour'])
        missingHours = hADay.difference(hours)
        
        for h in missingHours:
            newLine = pd.DataFrame(
                data=[[date, h, np.nan, np.nan, np.nan, np.nan, np.nan]],
                columns=['date', 'time', 't', 'dd', 'ff', 'HwaHwa', 'PwaPwa']
            )
            db = pd.concat([db, newLine], ignore_index=True)
    return db.sort_values(by=['date', 'hour'], axis=0, ignore_index=True)

def addBuoy(ds: nc.Dataset, begin: int):

    duration = {5: 31, 6: 30, 7: 31, 8: 31, 9: 30}

    # Check all files are downloaded

    wanted = set(f'{i}0{j}' for i in range(begin, 2022) for j in range(5, 10))
    downloaded = set(fileName[7:13] for fileName in os.listdir('./Data/Buoy'))
    download(wanted.difference(downloaded))

    # Adding to the dataset

    cnt = 0
    for year in range(begin, 2022):
        for month in range(5, 10):
            db = oneMonthBuoy(year, month)
            ds['t'][cnt:cnt + duration[month], :] = np.array(db['t']).reshape((duration[month], 24))
            ds['ws'][cnt:cnt + duration[month], :] = np.array(db['ff']).reshape((duration[month], 24))
            ds['wd'][cnt:cnt + duration[month], :] = np.array(db['dd']).reshape((duration[month], 24))
            ds['shww'][cnt:cnt + duration[month], :] = np.array(db['HwaHwa']).reshape((duration[month], 24))
            ds['mpww'][cnt:cnt + duration[month], :] = np.array(db['PwaPwa']).reshape((duration[month], 24))
            cnt += duration[month]

def nanPadding(a: np.ndarray, wrapperSize: int, shape=None):
    l = a.shape[0]
    shape = (wrapperSize,) if shape is None else shape
    if l > wrapperSize:
        raise ValueError('Too long array')
    else:
        padder = np.empty(shape=(wrapperSize - l,), dtype=a.dtype)
        padder[:] = np.nan
        return np.concatenate((a, padder)).reshape(shape)

def addCopernicus(ds: nc.Dataset, begin: int):

    cop = nc.Dataset('./Data/longCopernicus.nc', mode='r')
    mask = []
    for uxt in cop['time'][:].data:
        explicitDate = str(dt.datetime.fromtimestamp(int(uxt)))
        year = int(explicitDate[:4])
        month = int(explicitDate[5:7])
        if year < begin:
            mask.append(False)
        elif 4 < month < 10 and not explicitDate.endswith('05-01 00:00:00'):
            mask.append(True)
        elif explicitDate.endswith('10-01 00:00:00'):
            mask.append(True)
        else:
            mask.append(False)

    mask = np.array(mask, dtype='bool')
    l = np.count_nonzero(mask)
    ds['shww_rean'][:] = nanPadding(cop['VHM0_WW'][mask, 1, 1], (2022 - begin)*153*24, ((2022 - begin)*153, 24))
    ds['mpww_rean'][:] = nanPadding(cop['VHM0_WW'][mask, 1, 1], (2022 - begin)*153*24, ((2022 - begin)*153, 24))

    return l
            
def main(begin: int):

    assert 2014 < begin < 2022

    ds = create(begin)
    print('File created')
    addSoDa(ds, begin)
    print('SoDa data added')
    addBuoy(ds, begin)
    print('Buoy data added')
    l = addCopernicus(ds, begin)
    print(f'Copernicus data added : {l} values')
    ds.close()

if __name__ == '__main__':
    main(2016)
