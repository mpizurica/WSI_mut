import os
import cv2
import glob
import lmdb
import logging
import pyarrow
import lz4framed
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import tee
import argparse
import pickle
from PIL import Image
from os import walk

logging.basicConfig(level=logging.INFO,
                    format= '[%(asctime)s] [%(pathname)s:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def list_files_in_folder(folder_path: str, format: str='jpeg'): 
    GS_subfolders = os.listdir(folder_path)
    files = []
    for GS_folder in GS_subfolders:
        files_temp = os.listdir(folder_path+'/'+GS_folder)
        files += [folder_path+'/'+GS_folder+'/'+i for i in files_temp]

    return sorted(files)

def read_image_safely(image_file_name: str) -> np.array:
    try:
    #return cv2.imread(image_file_name) --> reads in BGR!!
        with open(image_file_name, 'rb') as f:
            img = Image.open(f)
            img.load()
        return np.array(img.convert('RGB'))
    except:
        print('Read image with PIL went wrong')
        print(image_file_name)
        return np.array([], dtype=np.uint8)

def serialize_and_compress(obj):
    return lz4framed.compress(pickle.dumps(obj))

def extract_image_name(image_path: str) -> str:
    return image_path.split('/')[-1]

def convert(image_folder: str, lmdb_output_path: str, format: str, write_freq: int=5000):
    assert os.path.isdir(image_folder), f"Image folder '{image_folder}' does not exist"
    assert not os.path.isfile(lmdb_output_path), f"LMDB store '{lmdb_output_path} already exists"
    assert not os.path.isdir(lmdb_output_path), f"LMDB store name should a file, found directory: {lmdb_output_path}"
    assert write_freq > 0, f"Write frequency should be a positive number, found {write_freq}"

    #logger.info(f"Creating LMDB store: {lmdb_output_path}")
    
    list_of_images = list_files_in_folder(image_folder, format)

    map_size = (len(list_of_images) + 100) * 3 * 512 * 512
    lmdb_connection = lmdb.open(lmdb_output_path, subdir=False,
                                map_size=int(map_size), readonly=False,
                                meminit=False, map_async=True)

    lmdb_txn = lmdb_connection.begin(write=True)
    total_records = 0
    try:
        for idx, img_path in enumerate(tqdm(list_of_images)):
            img_arr = read_image_safely(img_path)
            img_idx: bytes = u"{}".format(idx).encode('ascii')
            img_name: str = extract_image_name(image_path=img_path)
            img_name: bytes = u"{}".format(img_name).encode('ascii')
            if idx < 5:
                logger.debug(img_idx, img_name, img_arr.size, img_arr.shape)
            try:
                lmdb_txn.put(img_idx, serialize_and_compress((img_name, img_arr.tobytes(), img_arr.shape)))
            except:
                print('Something went wrong!!')
                continue
            total_records += 1
            if idx % write_freq == 0:
                lmdb_txn.commit()
                lmdb_txn = lmdb_connection.begin(write=True)
    except TypeError:
        logger.error(traceback.format_exc())
        lmdb_connection.close()
        print('TypeError')
        #raise
        return

    lmdb_txn.commit()

    logger.info("Finished writing image data. Total records: {}".format(total_records))

    logger.info("Writing store metadata")
    image_keys__list = [u'{}'.format(k).encode('ascii') for k in range(total_records)]
    with lmdb_connection.begin(write=True) as lmdb_txn:
        lmdb_txn.put(b'__keys__', serialize_and_compress(image_keys__list))

    logger.info("Flushing data buffers to disk")
    lmdb_connection.sync()
    lmdb_connection.close()

    # -- store the order in which files were inserted into LMDB store -- #
    #pd.Series(image_file__iter_c3).apply(extract_image_name).to_csv(os.path.join(lmdb_output_path, 'loc.csv'),
    #                                                                index=False, header=False)
    logger.info("Finished creating LMDB store")

def convert_folders(root_image_folder, format):
    for dir in os.listdir(root_image_folder):
        lmdb_name = dir.replace('.svs', '.db')
        lmdb_path = os.path.join(root_image_folder, dir, lmdb_name)
        if not os.path.exists(lmdb_path):
            convert(os.path.join(root_image_folder, dir), lmdb_path, format)
        #delete(os.path.join(root_image_folder, dir), format)

def read_ldmb(path):
    lmdb_connection = lmdb.open(path,subdir=False, readonly=True, lock=False, readahead=False, meminit=False)

    with lmdb_connection.begin(write=False) as lmdb_txn:
        length = lmdb_txn.stat()['entries'] - 1
        keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))

        for key in keys:
            val = lmdb_txn.get(key)
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(val))
            print(img_name)

        import pdb; pdb.set_trace()
    with lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(keys[0])
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
            image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
    import pdb; pdb.set_trace()

def get_length(path):
    """
    Get length of all databases in path
    """
    all_db = [i for i in os.listdir(path) if '-lock' not in i]
    length = 0
    for db in all_db:
        lmdb_connection = lmdb.open(path+db,subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with lmdb_connection.begin(write=False) as lmdb_txn:
            length += lmdb_txn.stat()['entries'] - 1

    return length

def get_image_name_index(path):
    """
    Returns dataframe with indices for every tile within database
    """
    all_db = [i for i in os.listdir(path) if '-lock' not in i]
    dfs = []
    for db in tqdm(all_db):
        lmdb_connection = lmdb.open(path+db,subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        
        with lmdb_connection.begin(write=False) as lmdb_txn:
            keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            img_names_ind = {} #img_name:ind
            indices = []
            if len(keys) > 0:
                for i, key in enumerate(keys):
                    val = lmdb_txn.get(key)
                    img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(val))
                    img_names_ind[img_name.decode()] = i
                df = pd.DataFrame(img_names_ind.keys(), img_names_ind.values()).reset_index()
                df.columns = ['index', 'img_name']
            
                dfs.append(df)

    total_df = pd.concat(dfs)
    total_df.to_csv('img_name_to_index.csv')
    return total_df


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Convert images form files to LMDB.')
    # parser.add_argument('--image_folder', type=str, help='Folder containing the images')
    # parser.add_argument('--lmdb_output_path', type=str, help='Folder to save the database')
    # parser.add_argument('--format', default='png', type=str, help='Folder to save the database')
    # args = parser.parse_args()
    # convert_folders(args.image_folder, args.format)
    
    path = '/labs/gevaertlab/data/prostate_cancer/TCGA_tiles/db_tiles_512px/'
    db = 'TCGA-J4-A67L-01Z-00-DX1.4B2B89CD-B390-488F-AE3F-9E81E6D860AD.db'
    #read_ldmb(path+db)
    #print(get_length(path))
    print(get_image_name_index(path))
