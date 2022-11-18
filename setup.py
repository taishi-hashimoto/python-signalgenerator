from setuptools import setup
from setuptools import find_packages


package_dir = "src"

setup(
    name='signalgenerator',
    version="v1.1.2",
    description="Simple signal generator (emulator).",
    package_dir={
        "": package_dir
    },
    packages=find_packages(package_dir),
    package_data={},
    install_requires=[
        "numpy",
        "scipy",
        "xarray",
    ],
    entry_points={
        "console_scripts": [
        ]
    },
    author="Taishi Hashimoto",
    author_email="hashimoto.taishi@outlook.com")
