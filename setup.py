import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
        name="rlalgs",
        version="0.0.6.4",
        author="Jhonas Prado Moura",
        author_email="jhonaspradomoura@gmail.com",
        description="A RL algorithms API",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jhonasiv/rl-algorithms",
        project_urls={
                "Bug Tracker": "https://github.com/jhonasiv/rl-algorithms/issues",
                },
        classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: POSIX :: Linux",
                "Development Status :: 2 - Pre-Alpha",
                "Intended Audience :: Science/Research",
                "Natural Language :: English",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Typing :: Typed"
                ],
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        python_requires=">=3.6",
        extras_requires={'gym': ['gym']}
        )

