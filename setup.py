# Basic setup.py to support testing and installation.
# This does not use the desiutil installation infrastructure so that desicmx
# does not depend upon any of the offline desidata desi* packages.  If we
# decide to bring in those dependencies, this could be updated too.
#
# Supports:
# - python setup.py install
# - python setup.py test
#
# Does not support:
# - python setup.py version  (edit py/desicmx/_version.py instead)

import os, glob, re
from setuptools import setup, find_packages

def _get_version():
    line = open('py/desicmx/_version.py').readline().strip()
    m = re.match("__version__\s*=\s*'(.*)'", line)
    if m is None:
        print('ERROR: Unable to parse version from: {}'.format(line))
        version = 'unknown'
    else:
        version = m.groups()[0]

    return version

#- Basic info
setup_keywords = dict(
    name='desicmx',
    version=_get_version(),
    description='DESI commissioning tests and analyses',
    author='DESI Collaboration',
    author_email='desi-commiss@desi.lbl.gov',
    license='BSD',
    url='https://github.com/desihub/desicmx',
)

#- boilerplate, not sure if this is needed
setup_keywords['zip_safe'] = False
setup_keywords['use_2to3'] = False

#- What to install
setup_keywords['packages'] = find_packages('py')
setup_keywords['package_dir'] = {'':'py'}

#- Treat everything in bin/ as a script to be installed
setup_keywords['scripts'] = glob.glob(os.path.join('bin', '*'))

#- Data to include
setup_keywords['package_data'] = {
    'desicmx': ['data/*',],
    'desicmx.test': ['data/*',],
}

#- Testing
setup_keywords['test_suite'] = 'desicmx.test.test_suite'

#- Go!
setup(**setup_keywords)
