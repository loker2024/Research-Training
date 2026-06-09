import os, numpy as np

ecl_dir = 'data/processed/ECL'
for h in [96, 168, 336]:
    h_dir = os.path.join(ecl_dir, f'h{h}')
    for split in ['train', 'val', 'test']:
        fpath = os.path.join(h_dir, f'{split}.npz')
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        try:
            data = np.load(fpath)
            x_shape = data['X'].shape
            y_shape = data['Y'].shape
            print(f'  h{h}/{split}.npz: {size_mb:.1f}MB  X={x_shape} Y={y_shape}  OK')
            del data
        except Exception as e:
            print(f'  h{h}/{split}.npz: {size_mb:.1f}MB  ERROR: {type(e).__name__}')
