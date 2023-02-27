# What is this package?
I wanted to explore genetic algorithms, a set of reinforcement learning algorithms that requires many agents to play at the same time. That is a feature that is not available in most backtesting software, and I wanted to see if I could implement it.


# Current status
This package is currently in development, and is largely made as a learning project. I likely will never use it to run real strategies as finding profitable algorithms in todays highly efficient markets is very difficult. 

However, this project can currently run backtests with a decent amount of configurability, and I've found it to be pretty fast to be able to implement and test new ideas. If you have recommendations shoot me a message.



# How to Use This Package
## Install initial tools and pyenv
```
sudo apt update && sudo apt upgrade
sudo apt install curl

# These are required for pyenv
sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

curl https://pyenv.run | bash
```


## Configure pyenv
The following is required for pyenv to work.

Add the following to the top of the .bashrc file:

```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
```
Add the following to the top of the .bashrc file:
```
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```
Now, restart the terminal, and run `pyenv` as initial test that pyenv was install


## Configure the environment and set up poetry

Ensure at this time that you have your pyproject.toml file in the root of the project and configured correctly with the poetry standards. If you're just trying to run the tests or examples, you can use the pyproject.toml file in this repo. Otherwise, you should have cloned this project, and you can define the path to this package in your pyproject.toml file.

Run the following commands to set up the environment and install poetry:
```
pyenv 3.10.6
pyenv virtualenv 3.10.6 <env_name>
pyenv local <env_name>
pip install poetry
poetry lock
poetry install
```

## Connecting to TDAmeritrade and Polygon APIs (brokerage and backtesting data)
The main function for getting data from polygon, "get_candle_data", takes the API key as a function parameter.

For TDAmeritrade, the set up is a little involved. Follow the Authentication Workflow defined at the [repository](https://github.com/areed1192/td-ameritrade-python-api).

## Running
Scripts can now be run with 
```poetry run python <path to script>```


<!-- To propogate changes to dependent modules: appears incrementing version number and then poetry update is the only way -->