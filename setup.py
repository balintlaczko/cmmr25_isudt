from setuptools import setup, find_packages

setup(
    name='cmmr25_isudt',
    version='0.1.0',
    author='Balint Laczko',
    author_email='balint.laczko@imv.uio.no',
    description='Code repository for our paper "Image Sonification as Unsupervised Domain Transfer", presented on CMMR 2025, London.',
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "lightning",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "tqdm",
        "python-osc",
        "wandb"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)