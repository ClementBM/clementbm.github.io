---
layout: post
title:  "A basic python development setup"
excerpt: "Development setup with vscode, git, pipenv, mypy and pytest"
date:   2020-05-23
categories: [python, vscode]
---

In this post we'll go through the entire setup of basic python project.
It will cover:
- IDE `vscode` which integrate really well with python
- `git` as a revision control system
- Virtual Environments with `pipenv`
- Enforce coding rules through static code analysis tools
   - `Pylint` looking for programming errors, helps enforcing a coding standard, sniffs for code smells and offers simple refactoring suggestions
   - `mypy`: a static type checker
- Code formatter `black`
- Unit tests with `pytest`
- Continuous integration with github `action`
- Debugging with vscode and make use of th python interactive windows in vscode
- Prepare project for packaging
- Documentation with `sphinx`

# My requirements
* Cross plateform (windows, linux)
* Works well with vscode
* One configuration per project (folder)
* Default configuration ?

Not a full setup environment, not SOLID, but could be assess in a next post.

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

**Show information about installed package**
```shell
pip show <package_name>
```

## IDE: VSCode
Advantages: cross platform
Advanced settings
Already built-in functionnalities

### Renaming:
F2

### Snippets:

### Debugging tools
https://code.visualstudio.com/docs/editor/debugging

### Workspace settings
Configuration
* settings.json

* configuration python path
PIPENV_VENV_IN_PROJECT
```shell
export PIPENV_VENV_IN_PROJECT=1
```
https://pipenv-fork.readthedocs.io/en/latest/advanced.html
https://github.com/pypa/pipenv/issues/178

### Platform-specific properties
https://code.visualstudio.com/docs/editor/debugging#_platformspecific-properties

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
pipenv install --dev
```

```shell
pipenv update ?
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

## Formatter

## Coding rules ?

flake8, pep8 ?
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

## Editing
https://code.visualstudio.com/docs/python/editing
### Formatter: black
https://github.com/psf/black
```shell
pipenv install black --dev --pre
```
https://github.com/Microsoft/vscode-python/issues/5171

## For Python interactive inside vscode
Visual Sudio Code supports working with Jupyter Notebooks natively.

* Work with Jupyter-like code cells
Run code in the Python Interactive Window
* View, inspect, and filter variables using the Variable explorer and data viewer
* Debug a Jupyter notebook
* Export a Jupyter notebook

https://code.visualstudio.com/docs/python/jupyter-support-py

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

### Configuration settings YAML
```shell
pipenv install pyyaml
```

## Documentation
Sphinx
https://www.sphinx-doc.org/en/master/
Latex ?

Model design with **GraphViz** (integrated to Sphinx ?)

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

- .github
   - ISSUE_TEMPLATE.md
   - PULL_REQUEST_TEMPLATE.md

### .gitignore


## Continous integration
build, test, package, release, or deploy 
GitHub Actions powers GitHub's built-in continuous integration service
Published Docker container image ?
Travis, Jenkins ? but github action simpler, as I just want a really basic CI
TOX ?

to go further: Artifact and publishing package
https://help.github.com/en/actions/publishing-packages-with-github-actions

### Add a badge
https://help.github.com/en/actions/configuring-and-managing-workflows/configuring-a-workflow#adding-a-workflow-status-badge-to-your-repository

![](https://github.com/<OWNER>/<REPOSITORY>/workflows/<WORKFLOW_FILE_PATH>/badge.svg)

# Sources
* https://docs.brew.sh/Homebrew-on-Linux

**Pipenv**
* https://www.youtube.com/watch?v=zDYL22QNiWk
* https://pipenv.pypa.io/en/latest/
* https://code.visualstudio.com/docs/python/testing

**CI in github**
* https://help.github.com/en/actions/getting-started-with-github-actions/about-github-actions
* https://help.github.com/en/actions/getting-started-with-github-actions/core-concepts-for-github-actions
* https://github.com/marketplace?type=actions

**Configuration files**
* https://martin-thoma.com/configuration-files-in-python/

Famous repo
1| scikit-learn
2| Flask
3| Keras
4| Sentry
5| Django
6| Ansible
7| Tornado
8| Pandas