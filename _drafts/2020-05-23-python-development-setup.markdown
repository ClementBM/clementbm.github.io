---
layout: post
title:  "A basic python development setup"
excerpt: "Development setup with vscode, git, pipenv, mypy and pytest"
date:   2020-05-23
categories: [python, vscode]
---

# Requirements
* Cross plateform (windows, linux)
* Works well with vscode
* One configuration per project (folder)
* Default configuration ?

## Prerequisite
### Brew
https://brew.sh/
(Add path in ~/.profile)
source ~/.profile
/!\ source is not sufficient for vscode terminal
/!\ vscode share the same terminal instance ?
~/.bashrc

### Configuration
```shell
python run import sys; print(sys.version); print(sys.executable)
```

**Configure bashrc**
```bashrc
alias python=python3
alias pip=pip3
```

**Show installed package**
```shell
pip show <package_name>
```

## Pipenv
### Pipenv
```shell
pip3 install pipenv
```
Not working ...
https://github.com/pypa/pip/issues/5599

```shell
brew install pipenv
```

This installs it globally. Since pipenv can manage even different python versions via pyenv, it's preferable to have it set up like this instead of installing it only for a specific python version using pip.

### Open virtual env
```shell
pipenv shell
```

### Exit virtual env
```shell
exit
```

### Update environment
```shell
pipenv --rm
pipenv install
```

Better way ?

### Update lock file
```shell
pipenv lock
```

### Install from Pipfile.lock
```shell
pipenv sync
```

on docker
```shell
pipenv install --system --deploy --ignore-pipfile
```

### Import dependencies from requirement.txt
```shell
pipenv install -r <path_to_requirement.txt>
```
/!\ the locking part may take a lot of time
```shell
Locking [packages] dependencies...
Locking ...
```

### Display dependency in the requirement.txt fashion
```shell
pipenv lock -r
```

### Export requirement.txt ?

### For dev pakages
```shell
pipenv install <package> --dev
```

### Pylint
```shell
pipenv install pylint --dev
```

### Mypy
```shell
pipenv install mypy --dev
```

### Pytest
```shell
pipenv install pytest --dev
```

## For Python interactive inside vscode
### Jupyter
```shell
pipenv install jupyter --dev
```

### ipykernel
```shell
pipenv install ipykernel --dev
```

### Notebook
```shell
pipenv install notebook --dev
```

## Git
### Folder hierarchy
* Avoid space

```shell
git remote add origin <>
git branch -u origin/master
git fetch
git rebase
...
```

## Continous integration
build, test, package, release, or deploy 
GitHub Actions powers GitHub's built-in continuous integration service
Published Docker container image ?

# Sources
* https://www.youtube.com/watch?v=zDYL22QNiWk
* https://pipenv.pypa.io/en/latest/
* https://code.visualstudio.com/docs/python/testing

* https://help.github.com/en/actions/getting-started-with-github-actions/about-github-actions
* https://help.github.com/en/actions/getting-started-with-github-actions/core-concepts-for-github-actions

Famous repo
1| scikit-learn.
2| Flask.
3| Keras.
4| Sentry.
5| Django.
6| Ansible.
7| Tornado.
8| Pandas.