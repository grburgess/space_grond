from setuptools import setup
import os

import versioneer


# Create list of data files
def find_stan_files(directory):

    paths = []

    for (path, directories, filenames) in os.walk(directory):

        for filename in filenames:

            if ".stan" in filename:
            
                paths.append(os.path.join("..", path, filename))

    return paths



setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
