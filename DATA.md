# Dataset Notes

This repository does not redistribute the datasets used in the paper.

## STEAD

STEAD is used as the source-domain pretraining dataset. Please download it from
the official data provider and convert or organize it into the HDF5/CSV format
expected by `v3.dataset_v3.STEADDatasetV3`.

## Mine Microseismic Dataset

The mine microseismic dataset used in the transfer-learning experiments may
contain industrial monitoring records and is not included in this repository.
To reproduce the workflow with your own data, prepare:

- waveform HDF5 file;
- metadata CSV file;
- three-component waveform records;
- optional background-noise segments.

## Industrial Non-Natural Seismic Dataset

The industrial non-natural seismic dataset is also excluded. The same HDF5/CSV
interface can be used for adaptation experiments.

## Recommended Local Layout

```text
data/
  stead/
    events.hdf5
    events.csv
    noise.hdf5
    noise.csv
  mining/
    data.hdf5
    data.csv
  nonnat/
    data.hdf5
    data.csv
```

The `data/` directory is ignored by git.
