from setuptools import setup, find_packages

setup(
    name='lunar_soil_analysis',
    version='0.1.0',
    description='AI-Driven Lunar Soil Composition Analysis',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow',
        'torch',
        'scikit-learn',
        'opencv-python',
        'numpy',
        'matplotlib',
        'openai',
        'clip-openai',
        'blip',
        'plotly',
        'folium',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
