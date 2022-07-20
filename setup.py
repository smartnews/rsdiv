from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

with (SETUP_DIRECTORY / "README.md").open() as ifs:
    LONG_DESCRIPTION = ifs.read()

install_requires = (
    [
        "numpy>=1.23.1",
        "pandas>=1.4.3",
        "scipy>=1.8.1",
        "lightfm>=1.16",
        "scikit-learn>=1.1.1",
        "matplotlib>=3.5.2",
        "plotly>=5.6.0",
    ],
)

setup(
    name="rsdiv",
    version="0.2.2",
    author="Yin Cheng",
    author_email="yin.sjtu@gmail.com",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/smartnews/rsdiv",
    python_requires=">=3.6",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    include_package_data=True,
)
