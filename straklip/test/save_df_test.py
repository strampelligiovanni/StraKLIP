import pandas as pd
import numpy as np

# # Create a DataFrame
# df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
#
# # Add metadata to the DataFrame
# df.attrs['description'] = 'Example DataFrame'
# df.attrs['scale'] = 0.1
# df.attrs['offset'] = 0.5
#
# # Apply scale and offset to the DataFrame
# df_scaled = (df * df.attrs['scale']) + df.attrs['offset']
#
# # Save the metadata to an HDF5 file
# with pd.HDFStore('/Users/gstrampelli/PycharmProjects/Giovanni/src/straklip/straklip/tests/out/NGC1976/crossmatch_ids.h5') as store:
#    store.put('data', df_scaled, format='table')
#    store.get_storer('data').attrs.metadata = df.attrs

# Read the metadata and DataFrame from the HDF5 file
with pd.HDFStore('/tests/out/NGC1976/crossmatch_ids.h5') as store:
   # try:
   metadata = store.get_storer('crossmatch_ids').attrs
   # except:
   #    pass
   df = store.get('crossmatch_ids')

df.attrs['description'] = 'Example DataFrame'
# Save the metadata to an HDF5 file
with pd.HDFStore('/tests/out/NGC1976/crossmatch_ids.h5') as store:
   store.put('crossmatch_ids', df, format='table')
   store.get_storer('crossmatch_ids').attrs.metadata = df.attrs

with pd.HDFStore('/tests/out/NGC1976/crossmatch_ids.h5') as store:
   metadata_read = store.get_storer('crossmatch_ids').attrs.metadata
   df_read = store.get('crossmatch_ids')

print()
# Retrieve the scale and offset from the metadata
# scale = metadata['scale']
# offset = metadata['offset']
#
# # Apply scale and offset to the DataFrame
# df_unscaled = (df_read - offset) / scale
#
# # Print the unscaled DataFrame
# print(df_unscaled)