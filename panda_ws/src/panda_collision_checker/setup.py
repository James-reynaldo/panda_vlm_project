from setuptools import find_packages, setup

package_name = 'panda_collision_checker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'launch/all_nodes.launch.py']),
    ],
    package_data={
        package_name: ['assets/*.xml'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='reynaldo',
    maintainer_email='reynaldo@campus.tu-berlin.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'collision_checker_node = panda_collision_checker.CollisionCheckerNode:main',
            'mujoco_node = panda_collision_checker.mujoco_node:main',
            'task_planner_node = panda_collision_checker.task_planner_node:main',
        ],
    },
)
