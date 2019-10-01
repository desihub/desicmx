# desicmx
DESI commissioning code

## Directory organization

`bin/` : generically useful scripts that multiple people may want to run

`py/desicmx/` : python code specific to CMX that you want to to be able to
    access via `from desicmx import blah`.
    
`doc/` : documentation of results; create subdirectories as needed.

`analysis/` : wild west dumping ground, create one directory per analysis.

`etc/` : for Data Systems to add module file definition, etc.

i.e. `bin/` and `py/desicmx/` are for sharing scripts and code that you want
others to use for their analyses; `doc/` is for documenting results; and
`analysis/` is a freeform place to dump code because it is better to put it
somewhere than nowhere.

## Ground rules

We won't enforce standard desihub code quality, unit tests, etc. requirements
for this repo, but please note that this repository is for code and
documentation, not for data files.
Analysis data files should go at NERSC in subdirectories of
`/global/project/projectdirs/desi/cmx/` instead.

