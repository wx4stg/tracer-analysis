import warnings
warnings.simplefilter('ignore')

import sys

import pandas as pd
import pyrcel

USE_DASK = True
if USE_DASK:
    from dask.distributed import Client

def compute_active_ccn(row_to_pyrcel, desired_supersat, w0=1, T0=280, P0=100000, S0=-0.02):
    aerosols = {}
    for i in range(1, 5):
        this_aerosol = pyrcel.AerosolSpecies(f'mode{i}',
                                            pyrcel.Lognorm(mu=row_to_pyrcel[f'Dg{i}']/2000, sigma=row_to_pyrcel[f'Sigma{i}'], N=row_to_pyrcel[f'N{i}']),
                                            kappa=row_to_pyrcel[f'K{i}'], bins=100)
        aerosols[f'mode{i}'] = this_aerosol
    mod = pyrcel.ParcelModel(aerosols.values(), w0, T0, S0, P0, console=False, accom=1)
    ptrace, aer_out = mod.run(100/w0, 1/w0, solver='cvode', output='dateframes', terminate=True)
    n = len(ptrace['S']) - 1
    activated_amounts = []
    for mode_key, aer_df in aer_out.items():
        mode_value = aerosols[mode_key]
        f, _, _, _ = pyrcel.binned_activation(desired_supersat/100, ptrace['T'].iloc[n],
                                            aer_df.iloc[n], mode_value)
        print(f'{mode_key} activation fraction: {f}')
        activated_amounts.append(f * mode_value.total_N)
        print(f'Activated amount = {f * mode_value.total_N}')
    
    total_activated_amount = sum(activated_amounts)
    return total_activated_amount


if __name__ == '__main__':
    path_to_input = sys.argv[1]
    input_data = pd.read_excel(path_to_input, skiprows=2, header=None, names=['timestamp',
                                                                            'N1', 'Dg1', 'Kmore1', 'Kless1', 'Fmore1', 'Sigma1',
                                                                            'N2', 'Dg2', 'Kmore2', 'Kless2', 'Fmore2', 'Sigma2',
                                                                            'N3', 'Dg3', 'Kmore3', 'Kless3', 'Fmore3', 'Sigma3',
                                                                            'N4', 'Dg4', 'Sigma4', 'K4']).set_index('timestamp', drop=True)
    for i in range(1, 4):
        input_data[f'K{i}'] = input_data.apply(lambda row: row[f'Kmore{i}'] if row[f'Fmore{i}'] > 0.5 else row[f'Kless{i}'], axis=1)

    sam_pyrcel_out = pd.DataFrame(columns=['timestamp', '0.1SS', '0.4SS', '0.6SS', '1.0SS'])
    if USE_DASK:
        client = Client('tcp://127.0.0.1:8786')
        futures_0_1 = []
        futures_0_4 = []
        futures_0_6 = []
        futures_1_0 = []
        for i, row in input_data.iterrows():
            print('Scheduling', i)
            futures_0_1.append(client.submit(compute_active_ccn, row, 0.1))
            futures_0_4.append(client.submit(compute_active_ccn, row, 0.4))
            futures_0_6.append(client.submit(compute_active_ccn, row, 0.6))
            futures_1_0.append(client.submit(compute_active_ccn, row, 1.0))
        print('Gathering!')
        results_0_1 = client.gather(futures_0_1)
        results_0_4 = client.gather(futures_0_4)
        results_0_6 = client.gather(futures_0_6)
        results_1_0 = client.gather(futures_1_0)
        
        for i, row in enumerate(input_data.index):
            sam_pyrcel_out.loc[row] = [row, results_0_1[i], results_0_4[i], results_0_6[i], results_1_0[i]]
    else:
        for i, row in input_data.iterrows():
            print('Processing', i)
            sam_pyrcel_out.loc[i] = [i, compute_active_ccn(row, 0.1), compute_active_ccn(row, 0.4), compute_active_ccn(row, 0.6), compute_active_ccn(row, 1.0)]
    print('Writing CSV...')
    sam_pyrcel_out.to_csv(sys.argv[2])