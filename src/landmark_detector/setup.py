from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'landmark_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # Ensure packages are found inside src/
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='Landmark detection package using ROS2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_detection = landmark_detector.image_detection:main',
            'robot_controller = landmark_detector.robot_controller:main',
            'landmark_marker_publisher = landmark_detector.landmark_marker_publisher:main',
            'amcl_pose_to_tf = landmark_detector.amcl_pose_to_tf:main',
        ],
    },
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/msg', glob('msg/*.msg')),
    ],
)