from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'smart_city'


def collect_files(folder):
    result = []

    if not os.path.exists(folder):
        return result

    for path, _, files in os.walk(folder):
        if not files:
            continue

        result.append(
            (
                os.path.join('share', package_name, path),
                [os.path.join(path, file) for file in files]
            )
        )

    return result


model_files = collect_files('model')
config_files = collect_files('config')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')
        ),
        (
            os.path.join('share', package_name, 'simulation'),
            glob('simulation/*.sdf')
        ),
        *model_files,
        *config_files
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marvin',
    maintainer_email='marvin@todo.todo',
    description='ROS 2 distributed smart city transport simulation',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bus_path_manager = smart_city.bus_path_manager:main',
            'navigation_executor = smart_city.navigation_executor:main',
            'traffic_light_manager = smart_city.traffic_light_manager:main',
            'taxi_coordinator = smart_city.taxi_coordinator:main',
            'taxi_request_manager = smart_city.taxi_request_manager:main',

            'bus_booking_generator = smart_city.bus_booking_generator:main',
            'taxi_request_generator = smart_city.taxi_request_generator:main',
            'private_car_simulator_node = smart_city.private_car_simulator_node:main',
        ],
    },
)