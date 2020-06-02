---
layout: post
title:  "Automatically check jekyll website links"
excerpt: "Automate jekyll website links checking with github action CI"
date:   2020-05-31
categories: [github action, jekyll, link checking]
---


The action needs permissions to push to your gh-pages branch. So you need to create a GitHub authentication token on your GitHub profile, then set it as an environment variable in your build using Secrets:

    On your GitHub profile, under Developer Settings, go to the Personal Access Tokens section.
    Create a token. Give it a name like “GitHub Actions” and ensure it has permissions to public_repos (or the entire repo scope for private repository) — necessary for the action to commit to the gh-pages branch.
    Copy the token value.
    Go to your repository’s Settings and then the Secrets tab.
    Create a token named JEKYLL_PAT (important). Give it a value using the value copied above.


# Sources
https://jekyllrb.com/docs/continuous-integration/github-actions/

https://github.com/marketplace/actions/jekyll-actions

[link](https://digitaldrummerj.me/jekyll-validating-links-and-images/)
[link](https://www.supertechcrew.com/jekyll-check-for-broken-links/)

https://help.github.com/en/actions/creating-actions/setting-exit-codes-for-actions

**Problem matcher**
https://github.com/actions/toolkit/blob/master/docs/commands.md#problem-matchers
https://github.com/actions/toolkit/blob/master/docs/problem-matchers.md
https://github.com/actions/setup-python/blob/master/.github/python.json