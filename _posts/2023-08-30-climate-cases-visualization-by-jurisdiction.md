---
layout: post
title:  "Explore Global Climate Cases by Jurisdiction"
excerpt: "A brief glance at the jurisdictions that are involved in climate change litigation, based on the cases recorded in the Sabin Center database"
date:   2023-08-30
categories: [project]
tags: [bokeh, graph, climate cases, visualization]
---

![Corn photo](/assets/2023-08-30/corn.jpg){: width="100%" style="margin-left: auto;margin-right: auto;display: block;"  }

# Context and Data
The [Sabin Center for climate change law](https://climate.law.columbia.edu/) manages a database that compiles legal cases from around the world related to climate change litigation.

Created in 2011, the **Global Climate Change Litigation Database** records cases involving material issue of climate change law, policy, or science.

Currently, there are approximately 800 cases documented in the database. All the informations regarding the database can be located through this [link](https://climatecasechart.com/about/).

In this post, my aim was to develop a rapid visualization tool for providing an overview of jurisdictional hierarchies and their corresponding cases.

By clicking on a graph node (selecting a jurisdiction), you have the ability to filter the case's datatable below.
Tap on the `Wheel Zoom` icon located in the sidebar on the right-hand side of the chart to activate the zoom in and zoom out functionalities.

Wishing you a joyful legal exploration!

> Apologies for not displaying all the country flags, working on it...

# Interactive Graph

{% include sabincenter-cases-graph.html %}

# Resources
* [Climate Case Chart by the Sabin Center of Climate Change Law](https://climatecasechart.com)
* [Interactive visualizations with the Bokeh python library](https://bokeh.org/)