from distutils.core import setup


setup(
    name='rankboost_plus',
    version='0.1.0',
    packages=['rankboost', 'rankboost.algorithms'],
    license='LICENSE.txt',
    description='Implementations of various ranking-by-boosting algorithms including Rankboost+.',
    install_requires=[
        "numpy >= 1.13.0",
        "scipy >= 1.0.0",
    ],
)

