---
layout: post
title:  "Automatically check for broken links on your jekyll website"
excerpt: "Automate jekyll website links checking with github action CI"
date:   2020-05-31
categories: [github action, jekyll, link checking]
---
In this post, I'll show a way to automatically validate links of your jekyll website with `html-proofer` (another Ruby Gem) and `Github Action CI`.

## Check for broken links
First, install html-proofer: add `gem 'html-proofer` to your `Gemfile`, then run `bundle install`.

Then, let's build the website including drafts with the following command `jekyll build --drafts`.

And finally `htmlproofer --log-level :debug ./_site` will show you a detailled report of the link check.

`html-proofer` has another nice features, just type `htmlproofer --help` to see all of them.

## Integration with Github
Let's say we have a static website running with jekyll on a github page. We would like to automate the check of broken links. Here, we describe a way to do that with Github Action.

Github gives a tool to execute workflows right into github repositories. Under the hood, a `Github Action` runs within a Docker container, which is given a bunch of contexts from Github (environment variables, the repository, and more).

If don't have yet configured a workflow, create a directory called `.github/workflows` at the root of your repository. Then create a yaml configuration file. I called mine `checklinks.yaml`, but it can be renamed as long as the extension remains correct.

Then copy and paste this, scroll down to see the explanation 
```yaml
name: CheckLinks
on:
  push:
    branches:
      - master
  pull_request:
    branches:
      - master

jobs:
  checklinks:
    name: Linux
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
    steps:
      - uses: actions/checkout@v2
      - name: Ruby
        uses: actions/setup-ruby@v1
        with:
          ruby-version: 2.6.x
      - name: Setup Rubygems, Bundler, jekyll
        run: | 
          gem update --system --no-document
          gem update bundler --no-document
          gem install jekyll bundler
          bundle install
      - name: Build jekyll website with drafts
        run: bundle exec jekyll build --drafts
      - name: Check for broken links
        run: |
          bundle exec htmlproofer --log-level :debug ./_site &> links.log
        continue-on-error: true
      - name: Archive log links
        uses: actions/upload-artifact@v1
        with:
          name: links-check.log
          path: links.log
```
### Workflow configuration
For this specific task, one important thing is the workflow trigger. It launchs on `push` and `pull_request` on the `master` branch.
```yaml
on:
  push:
    branches:
      - master
  pull_request:
    branches:
      - master
```
:clock1: you may want to run the task periodically rather than on `push` events. For that you can add a [on.schedule](https://help.github.com/en/actions/reference/workflow-syntax-for-github-actions#onschedule) configuration.

### Dependencies setup
```shell
gem update --system --no-document # Update Rubygems
gem update bundler --no-document # Update Bundler
gem install jekyll bundler # Install jekyll bundler
bundle install # Bundle install
```

### Build website and execute the check
```shell
bundle exec jekyll build --drafts
bundle exec htmlproofer --log-level :debug ./_site &> links.log
```
The first line builds the website including drafts. The second executes the check and logs it into a file.

:warning: You may prefix `jekyll` and `html-proofer` action with `bundle exec`, otherwise you may get some errors
```shell
You have already activated mercenary 0.4.0, but your Gemfile requires mercenary 0.3.6
```

### Avoid stop on error
I had to set the parameter `continue-on-error: true` otherwise the step fails and stops the worflow. But there's maybe a better way, perhaps with [`problem matchers`](https://github.com/actions/toolkit/blob/master/docs/problem-matchers.md) ?

### Artifacts
To keep the log accessible for further investigation, just upload it as an artifact.
```yaml
- name: Archive log links
  uses: actions/upload-artifact@v1
  with:
    name: links-check.log
    path: links.log
```

# Sources
* [Html Proofer](https://www.supertechcrew.com/jekyll-check-for-broken-links/)
* [Problem matcher](https://github.com/actions/toolkit/blob/master/docs/commands.md#problem-matchers)
* [Github Action](https://tech.gadventures.com/things-i-learned-making-my-first-github-action-84f528a97015)
* [Github Action: exit code](https://help.github.com/en/actions/creating-actions/setting-exit-codes-for-actions)
* [Github Action: continue-on-error](https://help.github.com/en/actions/reference/workflow-syntax-for-github-actions)