---
layout: post
title:  "Climate Cases by Juridiction Graph"
excerpt: "A quick view into the jurisdictions involved in climate change litigation"
date:   2023-08-30
categories: [project]
tags: [bokeh, graph, climatecases, visualization]
---

[Sabin Center for climate change law](https://climate.law.columbia.edu/) maintains a database that gather litigation cases across the global.

Created in 2011 the `Global Climate Change Litigation database` records cases that have a material issue of climate change law, policy, or science.

Currently, about 800 cases are recorded. You can find the database and informations at this [link](https://climatecasechart.com/about/).

In this post, I wanted to create a quick visualization tool to overview the hierachy of juridictions and their associated cases.

You can filter the below datatable by clicking on a node of the graph, thus selecting a juridiction.
The wheel zoom on the right side bar of the chart enables you to zoom in and zoom out.

Have fun!

> Apologies for not display all the country flags, working on it...

{% include sabincenter-cases-graph.html %}
