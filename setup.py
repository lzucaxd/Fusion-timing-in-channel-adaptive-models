from setuptools import setup, find_packages

setup(
    name='channel-adaptive-pipeline',
    version='0.1.0',

    url='https://github.com/lzucaxd/Fusion-timing-in-channel-adaptive-models',
    author='CHAMMI Team',
    author_email='zitong@broadinstitute.org',

    packages=find_packages(),
    install_requires=[
    #'faiss-gpu==1.7.2',
    'matplotlib==3.5.3',
    'numpy==1.22.0',
    'pandas==1.5.1',
    'scikit_image==0.19.2',
    'scikit_learn==1.2.2',
    'seaborn==0.12.2',
    'torch==2.0.0',
    'torchvision==0.15.1',
    'umap_learn==0.5.3',
    'tqdm==4.66.4',
    'timm'

    ]
    
)
