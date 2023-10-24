# This is a sample Python script.

import mat73
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def split_dataframe(df, chunk_size = 10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        print("on chunk number " +str(i) + " : " + str(round((i/num_chunks), 3)*100) + "%")
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

##bin into 10 ms bins and 1 if there is at least one spike in the bin 0 otherwise
def bin_data(df):
    df_split = split_dataframe(df, chunk_size=3)
    return df_split

def plot_data(data):
    for idx in range(1, len(data)):
        if idx%1000 == 0:
            neural_data = np.array(data.iloc[idx:idx+1])[0]
            print(neural_data)
            spike_times = [i for i, x in enumerate(neural_data) if x == 1]
            fig, ax = plt.subplots()
            ax.vlines(spike_times, 0, 1)
            plt.show()

def create_pandas_df_transpose(data):
    #print(data)
    df = pd.DataFrame(data)
    #print(df.T)
    return df.T

def create_pandas_df(data):
    #print(data)
    df = pd.DataFrame(data)
    #print(df.T)
    return df

def assign_spike_values_to_bins(binned_data):
    out_df = None
    for i, chunk in enumerate(binned_data):
        #print(((chunk > 0).any(axis=0)))
        if i == 0:
            out_df = pd.DataFrame(((chunk > 0).any(axis=0))).transpose()
        else:
            out_df.loc[i] = ((chunk > 0).any(axis=0))
    print(out_df.astype(int))
    return out_df.astype(int)

def simMI():
    print()

def compute_conmi():
    print()

def main():
    data_dict = mat73.loadmat('/media/macleanlab/DATA/IMAGING_DATA/caimanoutput_20231019-172438/evaluation/evaluated_IMAGING_DATA_20231020-164721.mat')
    #print(data_dict)
    neural_data = data_dict['neuron']['C']
    df = create_pandas_df_transpose(neural_data)
    binned_data = bin_data(df)
    spiked_binned_data = assign_spike_values_to_bins(binned_data)
    plot_data(spiked_binned_data)

if __name__ == '__main__':
    main()

