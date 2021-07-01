from distutils.core import setup

setup(
    name="vlgpax",
    version="2021.6.21",
    packages=["vlgpax"],
    url="https://github.com/yuanz271/vlgpax",
    license="MIT",
    author="yuan",
    author_email="yuanz271@gmail.com",
    description="variational Latent Gaussian Process",
    python_requires=">=3.8.0",
    install_requires=["jax", "jaxlib", "scikit-learn", "typer"],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
