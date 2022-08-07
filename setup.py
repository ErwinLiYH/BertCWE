import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="BertCWE",
    version="2.1.3",
    author="ErwinLi",
    author_email="1779599839@qq.com",
    description="Bert contextual word embedding",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/ErwinLiYH/BertCWE",
    project_urls={
        "Bug Tracker": "https://github.com/ErwinLiYH/BertCWE/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
