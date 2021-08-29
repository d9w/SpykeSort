import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skpyesort",
    version="0.0.1",
    author="Dennis G Wilson",
    author_email="d9w@pm.me",
    description="A python library for neural MEA data treatment and spike sorting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/d9w/SpykeSort",
    project_urls={
        "Bug Tracker": "https://github.com/d9w/SpykeSort/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License version 2.0",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["numpy",
                      "matplotlib",
                      "sklearn",
                      "pandas",
                      "scipy"],
    python_requires=">=3.6",
)
