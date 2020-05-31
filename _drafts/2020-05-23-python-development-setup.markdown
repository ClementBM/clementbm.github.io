---
layout: post
title:  "A basic python development setup"
excerpt: "Development setup with vscode, git, pipenv, mypy and pytest"
date:   2020-05-23
categories: [python, vscode]
---

In this post we'll go through the entire setup of basic python project.
It will cover:
- Directory structure
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
- Code coverage with `coverage.py`

I took inspiration on one a the current famous python repositories like [scikit-learn](https://github.com/scikit-learn/scikit-learn), [Flask](https://github.com/pallets/flask), [Keras](https://github.com/keras-team/keras), [Sentry](https://github.com/getsentry/sentry), [Django](https://github.com/django/django), [Ansible](https://github.com/ansible/ansible), [Tornado](https://github.com/tornadoweb/tornado), [Pandas](https://github.com/pandas-dev/pandas), and also from this repository [darker](https://github.com/akaihola/darker). Hoping that the tools their using are durable and scale well to most python projects.

This post is not a complete walk through tutorial, its aim is to give you a starter point, if you are relatively new to python and you look for good practices on how to structure a python project.

## My requirements
My requirements might not be yours, there are mines:

* Cross platform (Windows, macOS and Linux)
* Works well with vscode
* One configuration per project (folder)
* Default configuration ?

Not a full setup environment, not SOLID, but could be assessed in a next post.

## Prerequisites
I personnaly used [`brew`](https://brew.sh/) to install `pipenv`, and thus to manage `python` too.
I tested it on ubuntu `bionic` distribution, but it also works is macOS and Windows WSL. In my case, I had to add some paths to `~/.bashrc` file, but it's certainly better to read the complete [installation procedure](https://docs.brew.sh/Installation).

:warning: with the integrated `vscode` terminal, it appears that sourcing the `~/.bashrc` file or equivalent is not sufficient. It also seems that `vscode` share the same terminal instance across windows.

With [Brewfile](https://homebrew-file.readthedocs.io/en/latest/usage.html), `brew` enables configuration, that make easy first setup installation. Here is an example [file](https://github.com/getsentry/sentry/blob/master/Brewfile).

> You can check version and installation path of your current python installation by running
```shell
python3 -c $'import sys; print(sys.version); print(sys.executable)'
```

> You also might want to create aliases in your `~/.bashrc` or equivalent to run python3 by default:
```bashrc
alias python=python3
alias pip=pip3
```

## Directory structure
A basic git python project looks somethings like this:
```
.
├── .github/workflows
    └── test.yaml
├── .vscode
    └── extensions.json
    └── launch.json
    └── settings.json
├── docs/
    └── Makefile
    └── conf.py
    └── index.rst
├── src/
    └── mypkg/
        └── __init__.py
        └── app.py
        └── view.py
├── tests/
    └── __init__.py
    └── foo/
        └── __init__.py
        └── test_view.py
    └── bar/
        └── __init__.py
        └── test_view.py
├── .gitignore
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── setup.py
├── setup.cfg
```

As you might suppose, none of the file or directory is random. You'll know more about these choices reading the post. It's worth noting that this structure might be familiar for most programers familiar with github and python. Major inspiration for the structure comes from [pytest good practices](https://docs.pytest.org/en/latest/goodpractices.html), other from famous repositories previously listed.

> Just one reminder when naming your files ans directories, avoid spaces !

## Visual Studio Code
[Visual studio code](https://github.com/microsoft/vscode) is a versatile code editor, which natively integrates with `python`.

One of its advantages are:
* Cross platform (Windows, macOS and Linux)
* Already built-in pythonic functionnalities
* Advanced customization settings
* Common programing task like renaming, code snippets, and other [editing](https://code.visualstudio.com/docs/python/editing) sugar
* Also integrates with [`liveshare`](https://docs.microsoft.com/en-us/visualstudio/liveshare/), which make remote pair programing possible !

We present two more features that's worth noting: debugging and workspace settings.
### Visual Studio Code: Debugging tools
Apart from the great [vscode debugging](https://code.visualstudio.com/docs/editor/debugging) support, `vscode` also supports working with Jupyter Notebooks natively, through the [Python interactive windows](https://code.visualstudio.com/docs/python/jupyter-support-py), which enables you to:
* Work with Jupyter-like code cells
Run code in the Python Interactive Window
* View, inspect, and filter variables using the Variable explorer and data viewer
* Debug a Jupyter notebook
* Export a Jupyter notebook

### Visual Studio Code: Workspace settings
Workspace settings makes settings specific to a project, they make the development process easier and easily shareable with others.
Configurations is made through file located in the .vscode folder at root. Here we present two
* settings.json
* launch.json

`settings.json` gather all general settings specific to the current project.
`launch.json` specify the type of debugging scenarios. One cool thing about `launch.json` is that it has [Platform-specific properties](https://code.visualstudio.com/docs/editor/debugging#_platformspecific-properties) which means you can have specific launch commands depending on your OS.

## Pipenv
[Pipenv](https://github.com/pypa/pipenv) automatically creates and manages a virtualenv for your projects, as well as adds/removes packages from your Pipfile as you install/uninstall packages. It also generates the ever-important Pipfile.lock, which is used to produce deterministic builds.

### Installation
For installation you may refer to the [official procedure](https://github.com/pypa/pipenv#installation).
I personnaly make a global installation with `brew`:
```shell
brew install pipenv
```
Since `pipenv` can manage different python versions via pyenv, it's preferable to have it set up globally instead of installing it only for a specific python version using pip.

I think it can also work well with the official `sudo apt install pipenv` on Ubuntu.

In my case I wanted the virtual environment package to be in the project directory. For that, pipenv offer a configuration which can be activated via the `PIPENV_VENV_IN_PROJECT` environment variable. Just set it in your `~/.bashrc` file or equivalent.

```shell
export PIPENV_VENV_IN_PROJECT=1
```
See [doc](https://pipenv-fork.readthedocs.io/en/latest/advanced.html) for details, and this [issue](https://github.com/pypa/pipenv/issues/178).

```json
"python.pythonPath": ".venv/bin/python"
```

### Some basic pipenv commands
**Open virtual environment**
```shell
cd <project_directory>
pipenv shell
```

**Exit virtual environment**
```shell
exit
```

**First setup, install all packages**
```shell
pipenv install --dev
```

**Add package**
```shell
pipenv install <package> --dev
```

### Dependencies
**Import dependencies from requirement.txt**
```shell
pipenv install -r <path_to_requirement.txt>
```

:warning: the locking part may take a lot of time
```shell
Locking [packages] dependencies...
Locking ...
```

**Install dependencies from Pipfile.lock**
```shell
pipenv sync
```

on docker
```shell
pipenv install --system --deploy --ignore-pipfile
```

**Update environment**
```shell
pipenv --rm
pipenv install --dev # install dev packages
```

```shell
pipenv update
```

**Update lock file**
```shell
pipenv lock
```

**Display dependency in the requirement.txt fashion**
```shell
pipenv lock -r
```

### Install common packages

```shell
# Jupyter
pipenv install jupyter --dev
# ipykernel
pipenv install ipykernel --dev
# notebook
pipenv install notebook --dev
```

## Restructure code with Formatting
### Black
[`black`](https://github.com/psf/black) is code formatter which does not require configuration. It integrates with vscode as well.

```shell
pipenv install black --dev --pre
```
Using the `--pre` switch was mandatory because black is not currently released. See [issue](https://github.com/Microsoft/vscode-python/issues/5171) for more information.

To make it work with vscode I added this configuration lines in `settings.json`
```json
"[python]": {
   "editor.formatOnSave": true
},
"python.formatting.provider": "black"
```

It's also possible to sort Python import definitions alphabetically with `isort`.
Another useful way to share coding styles accross IDEs is using the `.editorconfig`, see [this](https://editorconfig.org/) for more info.

## Enforce coding rules with Linting
### Pylint
Linting highlights syntactical and stylistic problems in your Python source code, which oftentimes helps you identify and correct subtle programming errors or unconventional coding practices that can lead to errors. For example, linting detects use of an uninitialized or undefined variable, calls to undefined functions, missing parentheses, and even more subtle issues such as attempting to redefine built-in types or functions. Linting is thus distinct from Formatting because linting analyzes how the code runs and detects errors whereas formatting only restructures how code appears.

```shell
pipenv install pylint --dev
```
```json
"python.linting.enabled": true
```
https://code.visualstudio.com/docs/python/linting

[flake8](https://pypi.org/project/flake8/), [pep8](https://pypi.org/project/pycodestyle/)

### Mypy
```shell
pipenv install mypy --dev
```

```json
"python.linting.mypyEnabled": true
```

Configuration: `mypy.ini`

## Testing
### Pytest
linters other than the default PyLint
```shell
pipenv install pytest --dev
```
https://docs.pytest.org/en/latest/goodpractices.html

```json
"python.testing.pytestEnabled": true,
"python.testing.pytestArgs": [
   "tests"
]
```

Configuration: `pytest.ini`

Test discovery alternatives:
https://docs.pytest.org/en/latest/unittest.html#unittest-testcase

## Documentation
Sphinx
https://www.sphinx-doc.org/en/master/
Latex ?

Model design with **GraphViz** (integrated to Sphinx ?)

```shell
pipenv install sphinx --dev
pipenv install sphinx_rtd_theme --dev
```

also `mkdocs`
https://www.mkdocs.org/

### Documentation: initialization
```shell
cd docs
sphinx-quickstart
```

The quickstart will ask you a few questions and you are almost ready. As for now, version 3.0.4, tt creates 4 files `conf.py`, `index.rst`, `Makefile`, `make.bat`
You should now populate your master file `index.rst` and create other documentation source files. Use the Makefile to build the docs, like so: `make builder`
where "builder" is one of the supported builders, e.g. html, latex or linkcheck.

Edit `conf.py` file like so
```python
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

extensions = ["sphinx.ext.autodoc"]

html_theme = "sphinx_rtd_theme"
```

And then edit the `index.rst` file
```rst
   modules
```

You can the run the following commands to build a basic documentation
```shell
cd docs
# might delete *.rst (exept index.rst) files before
sphinx-apidoc -o . ../src/climbingboard --ext-autodoc
make html
```

## Python package
`setup.py`
* https://github.com/akaihola/darker/blob/master/setup.cfg
* https://setuptools.readthedocs.io/en/latest/setuptools.html#configuring-setup-using-setup-cfg-files

## Git
* Download and install the [latest version of git](https://git-scm.com/downloads).
* Configure git with your [username](https://help.github.com/en/articles/setting-your-username-in-git) and [email](https://help.github.com/en/articles/setting-your-commit-email-address-in-git).
```shell
git config --global user.name 'your name'
git config --global user.email 'your email'
```
* Clone the git repository ..
```shell
git clone <repository>
```
* .. or add origin
```shell
git remote add origin <repository>
git branch -u origin/master
```

Be sure to create a `.gitignore` file and set it properly.

It's always good to have issue and pull request templates. These are located in
```
├── .github
    └── ISSUE_TEMPLATE.md
    └── PULL_REQUEST_TEMPLATE.md
```

## Code coverage
https://coverage.readthedocs.io/en/coverage-5.1/

```shell
pipenv install coverage --dev
pipenv shell
coverage erase  # clears previous data if any
coverage run --source='.src' -m pytest
coverage report  # prints to stdout
coverage html  # creates ./htmlcov/*.html including annotated source
```

https://help.github.com/en/actions/configuring-and-managing-workflows/persisting-workflow-data-using-artifacts

https://codecov.io/
https://github.com/marketplace/codecov

Add comment on a pull request
https://github.com/thollander/actions-comment-pull-request

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

## Zen of Python
```python
import this
```

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

**Sphinx**
* https://www.youtube.com/watch?v=b4iFyrLQQh4
