from setuptools import setup, find_packages

setup(
    name='asnlib',
    version='0.0.1',
    description='Library of processing functions for ambient seismic noise purposes',
    author='Will Pizii',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'numpy>=1.18',
        'pandas>=1.1',
        'obspy',
        'matplotlib',
	    'tqdm'
    ],
    project_urls={
        'Source': 'https://github.com/willpizii/assortedseismicscripts'
    },
)