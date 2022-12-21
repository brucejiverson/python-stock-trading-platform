from setuptools import setup

setup(
   name='beasttrader',
   version='1.0.0',
   author='Bruce Iverson',
   author_email='brucejamesiverson@gmail.com',
   packages=['beasttrader', 'beasttrader.test'],
   scripts=[],
   url='http://pypi.python.org/pypi/beasttrader/',
   license='LICENSE.txt',
   description='A python platform for backtesting trading strategies with polygon data source and TD Ameritrade integration.',
   long_description=open('README.md').read(),
   install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "td-ameritrade-python-api",
        "sklearn",
        "asyncio",
        "mplfinance",
        "pandas",
        "pandas-ta",
        "polygon-api-client",
   ],
)
