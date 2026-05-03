from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'smart_university'

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
            'environment_perception_node = smart_university.environment_perception_node:main',
            'shelf_state_node = smart_university.shelf_state_node:main',
            'task_allocator_node = smart_university.task_allocator_node:main',
            'world_state_adapter_node = smart_university.world_state_adapter_node:main',
            'mock_gazebo_world_state_node = smart_university.mock_gazebo_world_state_node:main',

            'robot_state_node = smart_university.robot_state_node:main',
            'battery_manager_node = smart_university.battery_manager_node:main',
            'navigator_node = smart_university.navigator_node:main',
            'executor_node = smart_university.executor_node:main',
            'cleaning_action_server = smart_university.cleaning_action_server:main',
            'charging_action_server = smart_university.charging_action_server:main',
        ],
    },
)