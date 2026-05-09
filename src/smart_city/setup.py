from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'smart_city'

model_files = []
for path, _, files in os.walk('model'):
    for file in files:
        model_files.append(
            (
                os.path.join('share', package_name, path),
                [os.path.join(path, file)]
            )
        )

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
        *model_files
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marvin, abigail',
    maintainer_email='marvin@todo.todo, abigail@todo.todo',
    description='ROS 2 multi-robot smart university cleaning simulation',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'environment_perception_node = smart_city.environment_perception_node:main',
            'shelf_state_node = smart_city.shelf_state_node:main',
            'task_allocator_node = smart_city.task_allocator_node:main',
            'world_state_adapter_node = smart_city.world_state_adapter_node:main',
            'mock_gazebo_world_state_node = smart_city.mock_gazebo_world_state_node:main',

            'robot_state_node = smart_city.robot_state_node:main',
            'battery_manager_node = smart_city.battery_manager_node:main',
            'navigator_node = smart_city.navigator_node:main',
            'executor_node = smart_city.executor_node:main',
            'cleaning_action_server = smart_city.cleaning_action_server:main',
            'charging_action_server = smart_city.charging_action_server:main',
        ],
    },
)