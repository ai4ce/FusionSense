import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'fusion_sense'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'fusion_sense_resources'), glob(os.path.join('fusion_sense_resources', '**'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='irving',
    maintainer_email='zichuanfang2015@yahoo.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pipeline = fusion_sense.pipeline:main',
            'next_best_touch = fusion_sense.next_best_touch:main',
        ],
    },
)
