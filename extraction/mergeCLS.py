import numpy as np
import pandas as pd
import netCDF4 as nc

def main():

    # Prepare nc file

    ds = nc.Dataset('./Data/cls.nc', mode='w', format='NETCDF4')
    ds.description = 'Clear sky GHI for each hour of the day (computed for 2020)'

    ds.createDimension('lat', size=33)
    ds.createDimension('lon', size=33)
    ds.createDimension('date', size=366)
    ds.createDimension('time', size=24)

    lat = ds.createVariable('lat', 'float', dimensions=('lat'))
    lat.units = '°'
    lat.long_name = 'latitude'
    lon = ds.createVariable('lon', 'float', dimensions=('lon'))
    lon.units = '°'
    lon.long_name = 'longitude'
    ds.createVariable('date', 'str', dimensions=('date'))
    time = ds.createVariable('time', 'int', dimensions=('time'))
    time.units = 'h'
    time.long_name = 'Hour of the day'
    cls = ds.createVariable('GHIcls', 'float', dimensions=('lat', 'lon', 'date', 'time'))
    cls.units = 'W/m**2'
    cls.long_name = 'Clear sky GHI'

    # Open csv file

    df = pd.read_csv('./Data/ECMWF/GHI_CLS_1h_2020.csv', sep=';')

    # Write dates

    duration = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    datesOfYear = []
    for m, d in enumerate(duration):
        ms = str(m + 1)
        for k in range(d):
            datesOfYear.append(ms + '-' + str(k + 1))

    # Fill nc file

    ds['lat'][:] = np.array(df.loc[::33, 'Lat'], dtype='float')
    ds['lon'][:] = np.array(df.loc[df.index[:33], 'Lon'], dtype='float')
    ds['date'][:] = np.array(datesOfYear, dtype='str')
    ds['time'][:] = np.arange(1, 25, dtype='int')

    data = df.to_numpy()
    ds['GHIcls'][:] = data[:, 2:].reshape((33, 33, 366, 24))

    # Closing nc file

    ds.close()

if __name__ == '__main__':
    main()
