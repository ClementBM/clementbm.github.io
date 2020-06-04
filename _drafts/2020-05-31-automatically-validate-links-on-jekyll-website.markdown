---
layout: post
title:  "Automatically check jekyll website links"
excerpt: "Automate jekyll website links checking with github action CI"
date:   2020-05-31
categories: [github action, jekyll, link checking]
---
In this post, I'll show a way to automatically validate links of your jekyll website with `html-proofer` (another Ruby Gem) and Github Action CI.

## Check for broken links
FIrst, we install html-proofer. Add `gem 'html-proofer` to your `Gemfile`. Then run `bundle install` to install it.

Then, let's build the website including drafts with the following command `jekyll build --drafts`.

And finally `htmlproofer --log-level :debug ./_site` will show you a detailled report of the link check.

`html-proofer` has another nice features, just type `htmlproofer --help` to see all of them.

## Integration with Github
Let's say we have a static website running with jekyll on a github page. We would like to automate this task of checking for broken links. Here, we describe a way to do that with Github Action.

Github gives a tool to execute workflows right into github repositories. Under the hood, a `Github Action` runs within a Docker container, which is given a bunch of contexts from Github (environment variables, the repository, and more).

If don't have yet configured a workflows, create a directory called `.github/workflows` at the root of your directory. Then create a yaml configuration file. I called mine `checklinks.yaml`, but it can be changed as long as you keep the extension correct.

Then copy and paste this:
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
      - name: Update Rubygems
        run: 'gem update --system --no-document'
      - name: Update Bundler
        run: 'gem update bundler --no-document'
      - name: Install jekyll bundler
        run: 'gem install jekyll bundler'
      - name: Bundle install
        run: bundle install
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

You may prefix jekyll and html-proofer action with bundle exec, otherwise you may get some error
```shell
You have already activated mercenary 0.4.0, but your Gemfile requires mercenary 0.3.6
```

### Setup

### Avoid stop on error

### Artifacts

# Sources
**Jekyll Action**
https://jekyllrb.com/docs/continuous-integration/github-actions/
https://github.com/marketplace/actions/jekyll-actions

**Html Proofer**
https://digitaldrummerj.me/jekyll-validating-links-and-images/
https://www.supertechcrew.com/jekyll-check-for-broken-links/

**Problem matcher**
https://github.com/actions/toolkit/blob/master/docs/commands.md#problem-matchers
https://github.com/actions/toolkit/blob/master/docs/problem-matchers.md
https://github.com/actions/setup-python/blob/master/.github/python.json

**On github action**
https://tech.gadventures.com/things-i-learned-making-my-first-github-action-84f528a97015
https://help.github.com/en/actions/creating-actions/setting-exit-codes-for-actions

**Continue on error**
https://help.github.com/en/actions/reference/workflow-syntax-for-github-actions