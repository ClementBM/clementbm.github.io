---
layout: post
title:  "Sentiment analysis on twitter data with ELK"
excerpt: "A quick sentiment analysis on twitter data with ELK (Elasticsearch, Logstash, Kibana)"
date:   2020-03-02
categories: [Elasticsearch, Kibana, Logstash, ELK, Sentiment analysis]
---
In this post we'll perform a very quick installation of ELK, then get some tweets filtered by a certain keyword and finally add a sentiment analysis plugin into the logstash pipeline.

> Prerequisite: you may already be familiar with Docker and Docker Compose.

ELK stands for:
1. **Elasticsearch**: a search engine providing a distributed, multitenant capable full-text search
2. **Logstash**: a service used to collect, process and forward events and log messages
3. **Kibana**: a browser-based analytics and search interface for Elasticsearch

Below, a diagram of the ELK stack. 

![elk stack diagram](/assets/2020-03-02/elk-stack-elkb-diagram.svg)

## Setup with Docker
For convenience, we'll use Docker for the setup. Indeed, the repository [docker-elk](https://github.com/deviantony/docker-elk) gives a ready to use Elastic stack (ELK) infrastructure based on Docker and Docker Compose. All the installation process is well defined in the [README](https://github.com/deviantony/docker-elk) so we won't go into more details here. 

## Twitter logstash plugin
Logstash uses [configurable input plugin](https://www.elastic.co/guide/en/logstash/current/input-plugins.html) to retreive data, [filters](https://www.elastic.co/guide/en/logstash/current/filter-plugins.html) to process it and [ouptuts](https://www.elastic.co/guide/en/logstash/current/output-plugins.html) to define where to write all the aggregated data.

![logstash diagram from https://codeblog.dotsandbrackets.com/processing-logs-logstash/](/assets/2020-03-02/logstash-diagram.jpg)

Tweets can be collected via the [twitter input plugin](https://www.elastic.co/guide/en/logstash/current/plugins-inputs-twitter.html). Plugins are defined in logstash configuration files. 
So, retreiving twitter messages based on keyword can be achieved by adding a configuration file:

```json
input {
    twitter {
        consumer_key => "<your_consumer_key>"
        consumer_secret => "<your_consumer_secret>"
        oauth_token => "your_oauth_token"
        oauth_token_secret => "your_oauth_token_secret"
        keywords => [ "Coronavirus", "COVID-19" ]
        languages => [ "en-US" ]
    }
}
```
For this post, we override the `logstash.conf` in [kibana/logstash/config/logstash.yml](https://github.com/deviantony/docker-elk/blob/master/kibana/config/kibana.yml).

If you don't already have a twitter developper account, you may get one by following the twitter [documentation](https://developer.twitter.com/en/docs/basics/getting-started).

We added `languages` property to the twitter configuration to get only twitter message in english. `languages` take an array of string of bcp-47 language code, see [BCP-47 Code List](https://appmakers.dev/bcp-47-language-codes-list/). However, after experimentation it doesn't seem to filter as expected so it will require further investigations.

## Add sentiment analysis filter plugin
Once the input plugin has collected data it can be processed by any number of filters which then modify and annotate the event data. 

We use a filter to perform a sentiment analysis on twitter messages. I found this one [logstash-filter-sentimentalizer](https://github.com/tylerjl/logstash-filter-sentimentalizer) and add it to the logstash [Dockerfile](https://github.com/deviantony/docker-elk/blob/master/logstash/Dockerfile)

```dockerfile
ARG ELK_VERSION

# https://github.com/elastic/logstash-docker
FROM docker.elastic.co/logstash/logstash:${ELK_VERSION}

# Add your logstash plugins setup here
RUN logstash-plugin install logstash-filter-sentimentalizer
```

> After this change you must run `docker-compose build`

And then, we add a filter in the logstash configuration file
```json
filter {
  sentimentalizer {
    source => "message"
  }
}
```

## Output to Elasticsearch
Finally logstash routes events to output plugins which can forward the events to a variety of external programs including Elasticsearch, local files and several message bus implementations.

In this post, we simply pass the output to Elacticsearch. So we add the following to the configuration file
```json
output {
    elasticsearch { 
        hosts => "elasticsearch:9200"
        user => "elastic"
        password => "changeme"
        index => "twitter"
        document_type => "tweet"
    }
    stdout {codec => rubydebug }
}
```
Here I let the default login/password from docker-elk which might not be a good practice.

## Final configuration file
The configuration file might look like
```json
input {
    twitter {
        consumer_key => "<your_consumer_key>"
        consumer_secret => "<your_consumer_secret>"
        oauth_token => "your_oauth_token"
        oauth_token_secret => "your_oauth_token_secret"
        keywords => [ "Coronavirus", "COVID-19" ]
        languages => [ "en-US" ]
    }
}
filter {
  sentimentalizer {
    source => "message"
  }
}
output {
    elasticsearch { 
        hosts => "elasticsearch:9200"
        user => "elastic"
        password => "changeme"
        index => "twitter"
        document_type => "tweet"
    }
    stdout {codec => rubydebug }
}
```

## Visualisation with Kibana
We can now access Kibana interface to get some insights.

First we need to add the `twitter` index pattern by clicking on "Created index pattern" and writting `twitter` (defined in the output).
![index pattern](/assets/2020-03-02/index-pattern.png)

Then we can add some visualization chart by clicking on "Create visualization"
![visualization list](/assets/2020-03-02/visualization-list.png)

### Barchart
In the following barchart we can see the evolution of the sentiment on the keyword "Coronavirus" by hour.
![bar chart](/assets/2020-03-02/bar-chart.png)

### Piechart
An overall view of sentiment polarity on the same keyword
![pie chart](/assets/2020-03-02/pie-chart.png)

### Using data visualizer
We can also have a quick insight of the data by using "Data Visualizer". We are some screenshots of the indicators we have
![visualization count](/assets/2020-03-02/visualization-count.png)
![visualization overall](/assets/2020-03-02/visualization-overall.png)

# Sources
* [Docker ELK](https://github.com/deviantony/docker-elk)
* [Docker Compose](https://docs.docker.com/compose)
* [A Simple Sentiment Analysis Prototype Using Elasticsearch](https://qbox.io/blog/sentiment-analysis-prototype-using-elasticsearch)
* [ELK](https://wikitech.wikimedia.org/wiki/Logstash)
* [Logstash](https://codeblog.dotsandbrackets.com/processing-logs-logstash/)