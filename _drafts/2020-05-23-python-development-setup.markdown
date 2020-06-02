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

This post is not a complete walk through tutorial, its aim is to give you a starter point, if you are relatively new to python and you look for good practices on how to structure a python project. I also give a bunch of link if you want to dig deeper or know more about alternatives.

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

As you might suppose, none of the files or directories are choosen randomly. You'll know more about these choices reading the post. It's worth noting that this structure might be familiar for most programers working with github and python. Major inspiration for the structure comes from [pytest good practices](https://docs.pytest.org/en/latest/goodpractices.html), other from famous repositories previously listed.

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

For the vscode interactive window to be active, you need these three packages: `jupyter`, `ipykernel` and `notebook`.
Here are the installation commands with pipenv:
```shell
# Jupyter
pipenv install jupyter --dev
# ipykernel
pipenv install ipykernel --dev
# notebook
pipenv install notebook --dev
```

### Visual Studio Code: Workspace settings
Workspace settings makes settings specific to a project, they make the development process easier and easily shareable with others.
Configurations is made through file located in the .vscode folder at root. Here we present two
* settings.json
* launch.json

`settings.json` gather all general settings specific to the current project.
`launch.json` specify the type of debugging scenarios. One cool thing about `launch.json` is that it has [Platform-specific properties](https://code.visualstudio.com/docs/editor/debugging#_platformspecific-properties) which means you can have specific launch commands depending on your OS.

## Pipenv
[Pipenv](https://github.com/pypa/pipenv) automatically creates and manages a virtualenv for your projects, as well as adds/removes packages from your Pipfile as you install/uninstall packages. It also generates the ever-important Pipfile.lock, which is used to produce deterministic builds.

### Pipenv: Installation
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

### Pipenv: some basic commands
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

### Pipenv: manage dependencies
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

linters other than the default PyLint [flake8](https://pypi.org/project/flake8/), [pep8](https://pypi.org/project/pycodestyle/)

### Mypy
[mypy](http://mypy-lang.org/) is static type checker. Since [type hint](https://docs.python.org/3/library/typing.html) were released in version `3.5` but as the Python runtime does not enforce function and variable type annotations, a type checker is needed if you want to enable type checking.
Installation via pipenv:
```shell
pipenv install mypy --dev
```
Update the vscode `settings.json` file:
```json
"python.linting.mypyEnabled": true
```

For the configuration of `mypy`, it uses by default the `mypy.ini` file with fallback to `setup.cfg`.

## Testing
### Pytest
`pytest` full-featured Python testing tool. It is already used by a lot of repositories.
Installation with pipenv: 
```shell
pipenv install pytest --dev
```

Then update vscode `settings.json` file with these lines:
```json
"python.testing.pytestEnabled": true,
"python.testing.pytestArgs": [
   "tests"
]
```

To [customize pytest](https://docs.pytest.org/en/latest/customize.html), your configuration must go in either one of these files: `pytest.ini`, `tox.ini` and `setup.cfg`.

For discovery, `pytest` usually search for file called like `test_*.py` or `*_test.py` and then looks for function and methods prefixed by test. See [this](https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery) for the full explanation. As an alternative, `pytest` also discover natively [unittest](https://docs.pytest.org/en/latest/unittest.html#unittest-testcase) and nosetest.

I could not omit to talk about `tox`, a tool that automates and standardizes testing in Python. It integrates easily with `pytest`. What does `tox` do ? Basically it will creates a virtual environment a run the tests for you, as well as checking the package installation, and it make your life easier when you go in continuous integration workflow.  

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
By default, in python terminology, a folder is package, a file is a module, and that module contains definitions and statements. The file name is the module name with the suffix .py appended.
__init__.py is required to import the directory as a regular package, and can simply be an empty file. More information [here](https://docs.python.org/3/reference/import.html#regular-packages).

We then need a build script for [setuptools](https://packaging.python.org/key_projects/#setuptools). It tells setuptools about your package (such as the name and version) as well as which code files to include. It's commonly done in a `setup.py` located at the root of the repository. I personnaly prefer the configuration way, with a `setup.cfg` [file](https://setuptools.readthedocs.io/en/latest/setuptools.html#configuring-setup-using-setup-cfg-files).

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
Coverage measurement is used to gauge the effectiveness of tests. It can show which parts of your code are being exercised by tests, and which are not.

For this task we use [Coverage.py](https://coverage.readthedocs.io/en/coverage-5.1/).
Here is how you can install it with `pipenv`.

```shell
pipenv install coverage --dev
pipenv shell
coverage erase  # clears previous data if any
coverage run --source='.src' -m pytest
coverage report  # prints to stdout
coverage html  # creates ./htmlcov/*.html including annotated source
```

We can then upload it as an [artifact](https://help.github.com/en/actions/configuring-and-managing-workflows/persisting-workflow-data-using-artifacts) with github action. It enables us to download the coverage report in github action tab.

> Github integrates also with [codecov](https://github.com/marketplace/codecov), and make it easier to vizualise report.

> You can also add comment on a pull request with this [action](https://github.com/thollander/actions-comment-pull-request).

## Continous integration
From [october 2018](https://github.blog/2018-10-17-action-demos/) GitHub Actions enables developers to automate, customize, and execute workflows directly in their repositories. By workflow, I mean build, test, package, release, or deploy your software.
Besides the complete built-in continuous integration service within github, it has another two intersting features:
* Built in secret store
* Multi-container testing, to play with `docker-compose`

There are two well known service: [Travis](https://travis-ci.org/) and [Jenkins](https://www.jenkins.io/), but as my requirements are not that high, and for the sake of simplicity I choose github action.

> To go further: artifact and publishing package are common tasks, as if you want to register a Docker container image to a package provider, and are supported by github action. Go [there](https://help.github.com/en/actions/publishing-packages-with-github-actions) to know more about packaging.

### Github Action: Add a badge
You can easily add a [workflow status badge](https://help.github.com/en/actions/configuring-and-managing-workflows/configuring-a-workflow#adding-a-workflow-status-badge-to-your-repository) associated with a github workflow.

For example to show on your readme:
```markdown
![](https://github.com/<OWNER>/<REPOSITORY>/workflows/<WORKFLOW_FILE_PATH>/badge.svg)
```

## Zen of Python
```python
import this
```

```
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
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

**Package**
* https://www.youtube.com/watch?v=UK97NoQK23k
* http://python-notes.curiousefficiency.org/en/latest/python_concepts/import_traps.html
* https://docs.python.org/3/reference/import.html
* https://docs.python.org/3/tutorial/modules.html
* https://packaging.python.org/tutorials/packaging-projects/
* https://docs.python.org/3/glossary.html#term-portion
* https://stackoverflow.com/questions/26667490/change-cwd-before-running-tests