# -*- coding: utf-8 -*-
"""
Created on Mon November 11 2019

@author: fourndo@gmail.com


Batch Inversion

Run a series of inversions calling PF_Inversion_XXXXXXX.py

"""

import os
import json
import sys
from shutil import copyfile

dsep = os.path.sep
batch_file = sys.argv[1]
inv_script = sys.argv[2]
input_file = sys.argv[3]

if input_file is not None:
    workDir = dsep.join(
                os.path.dirname(os.path.abspath(input_file)).split(dsep)
            )
    if len(workDir) > 0:
        workDir += dsep

else:

    assert input_file is not None, "The input file is missing: 'python PFinversion.py input_file.json'"

# Read input json file
with open(input_file, 'r') as f:
    driver = json.load(f)

input_dict = dict((k.lower(), driver[k]) for k in list(driver.keys()))

###############################################################################
# Read json batch file and overwrite defaults
with open(batch_file, 'r') as f:
    driver = json.load(f)

batch_dict = dict((k.lower(), driver[k]) for k in list(driver.keys()))

if "result_folder" in list(input_dict.keys()):
    root = os.path.commonprefix([input_dict["result_folder"], workDir])
    outDir = workDir + os.path.relpath(input_dict["result_folder"], root) + dsep
else:
    outDir = workDir + dsep + "SimPEG_PFInversion" + dsep

# Loop through the batch variables
for ii, (key, parameters) in enumerate(batch_dict.items()):

    # Loop trough the variations
    for jj, parameter in enumerate(parameters):
        input_dict[key] = parameter

        with open(input_file, 'w') as outfile:

            # outfile.write(
            #     '[' +
            #     ',\n'.join(json.dumps(i) for i in input_dict.items()) +
            #     ']\n'
            # )

            json.dump(input_dict, outfile)

        os.system('python ' + inv_script + " " + input_file)

        # Once the inversion is done, copy all in a new directry
        save_dir = outDir + "Batch_" + str(ii) + "_" + str(jj) + dsep
        os.system('mkdir ' + save_dir)
        for file in os.listdir(outDir):

            if ("zarr" not in file) and (os.path.isfile(outDir + file)):

                copyfile(outDir + file, save_dir + file)

        # Copy the input file for reference
        copyfile(input_file, save_dir + os.path.basename(input_file))
