import netCDF4 as nc
import numpy as np
import os

def state(beg, n):
    os.system('clear')
    print(f'Processed {n} / {153*(2022 - beg)}')

def main(beg):

    state(beg, 0)
    
    # Prepare nc file

    ds = nc.Dataset('./Data/nwp.nc', mode='w', format='NETCDF4')
    ds.source = 'ECMWF'
    ds.description = 'NWP from 2018 to 2021'

    ds.createDimension('date', size=153*(2022 - beg))
    ds.createDimension('time', size=25)
    ds.createDimension('lat', size=33)
    ds.createDimension('lon', size=33)

    dateVar = ds.createVariable('date', 'str', dimensions=('date'))
    dateVar.units = 'YY-MM-DD'
    time = ds.createVariable('time', 'int', dimensions=('time'))
    time.units = 'h'
    time.long_name = 'Hour of the day'
    lat = ds.createVariable('lat', 'float', dimensions=('lat'))
    lat.units = '°'
    lat.long_name = 'latitude'
    lon = ds.createVariable('lon', 'float', dimensions=('lon'))
    lon.units = '°'
    lon.long_name = 'longitude'

    ghi = ds.createVariable('ghi', 'float', dimensions=('date', 'time', 'lat', 'lon'))
    ghi.units = 'W/m**2'
    ghi.long_name = 'Global horizontal irradiance'
    t = ds.createVariable('t', 'float', dimensions=('date', 'time', 'lat', 'lon'))
    t.units = 'K'
    t.long_name = '2 meters above the ground temperature'
    shww = ds.createVariable('shww', 'float', dimensions=('date', 'time', 'lat', 'lon'))
    shww.units = 'm'
    shww.long_name = 'Significant height of wind waves'
    mpww = ds.createVariable('mpww', 'float', dimensions=('date', 'time', 'lat', 'lon'))
    mpww.units = 's'
    mpww.long_name = 'Mean period of wind waves'
    ws = ds.createVariable('ws', 'float', dimensions=('date', 'time', 'lat', 'lon'))
    ws.units = 'm/s'
    ws.long_name = '10 meters above the ground wind speed'
    wd = ds.createVariable('wd', 'float', dimensions=('date', 'time', 'lat', 'lon'))
    wd.units = '°'
    wd.long_name = '10 meters above the ground wind direction'

    # Filling dimensions

    datesOfYear = []
    ds['time'][:] = np.arange(25)
    ds['lat'][:] = np.linspace(46, 42, 33)
    ds['lon'][:] = np.linspace(5.5, 9.5, 33)

    # Filling variables

    duration = [31, 30, 31, 31, 30]

    date = 0
    for year in range(beg, 2022):
        for m, d in enumerate(duration):
            month = m + 5
            for day in range(1, d + 1):
                daystr = '0'*int(day < 10) + str(day)
                datesOfYear.append(f'{year}-0{month}-' + daystr)
                fileName = f'./Data/ECMWF/ECMWF_Monaco/ECMWF_Monaco_{year}0{month}' + daystr + '.nc'

                dpd = nc.Dataset(
                    fileName,
                    mode='r',
                    format='NETCDF3'
                )

                ds['ghi'][date, :, :, :] = dpd['ssrd'][:]
                ds['t'][date, :, :, :] = dpd['t2m'][:]
                ds['shww'][date, :, :, :] = dpd['shww'][:]
                ds['mpww'][date, :, :, :] = dpd['mpww'][:]
                ds['wd'][date, :, :, :] = dpd['dwi'][:]
                ds['ws'][date, :, :, :] = dpd['wind'][:]

                date += 1
                state(beg, date)

    ds['date'][:] = np.array(datesOfYear)

    # Closing nc file

    ds.close()
    print('Success')

if __name__ == '__main__':
    main(2016)
