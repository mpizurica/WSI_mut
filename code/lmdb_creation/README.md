# Tile storage

In order to more efficiently store tiles, the [lmdb_converter.py](https://github.com/mpizurica/WSI_mut/blob/master/code/lmdb_creation/lmdb_converter.py) can be used. It stores all tiles of a patient in an LMDB database. In the training script, you can indicate whether your file structure contains separate tiles for slides or whether you use an LMDB to store tiles of a patient.
