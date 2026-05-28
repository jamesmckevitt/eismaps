from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name('README.md').read_text(encoding='utf-8')

setup(
    name='eismaps',
    version='0.2.3',
    author='James McKevitt',
    author_email='jm2@mssl.ucl.ac.uk',
    description='A toolkit for producing level 3 maps from Hinode/EIS spacecraft data.',
    license='CC-BY-NC-SA-4.0',
    license_files=('LICENSE',),
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/jamesmckevitt/eismaps',
    packages=find_packages(),
    include_package_data=True,  # for non-code files in MANIFEST.in
    package_data={
        'eismaps': [
            '*.dat',
            'calibration_data/*.sav',
            'calibration_data/EIS_EffArea_*',
            'calibration_data/*.json',
        ]
    },
    install_requires=[
        'eispac>=0.99.3',
        'numpy>=1.24',
        'sunkit-image>=0.5.1',
    ],
    python_requires='>=3.9',
    classifiers=[
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={
        'console_scripts': [
            'eismaps-sync-calibration=eismaps.calibration:sync_solarsoft_calibration_data_cli',
        ],
    },
)
