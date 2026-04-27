from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'smart_university'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'simulation'),
            glob('simulation/*.sdf')),
        (os.path.join('share', package_name, 'model'),
            glob('model/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marvin',
    maintainer_email='marvin@todo.todo',
    description='Smart University Project',
    license='TODO',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lidar_controller = smart_university.lidar_controller:main',
            # add others here
        ],
    },
)
