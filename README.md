# Youtube and the myth of political neutrality
---

## Abstract :writing_hand:
American electoral politics have been dominated by two major political parties: the Democratic Party and the Republican Party. While democrats are liberal and left-leaning, republicans adopted a conservative right-leaning ideology. Each year, the world watches how these two powerful parties dispute power and picks sides. In this context, does Youtube have a political orientation? Is youtube strictly politically unbiased or does it have its own preferences?

For the purpose of finding the answer to these questions, we use a natural language processing model to categorise the political affiliation of a video based on title and text description. We try to understand if one political affiliation is more prevalent than the other and also correlated with a higher engagement rate. We perform our analysis over time to see whether or not these trends are changing and if they reflect the US political reality. In order to  deepen our understanding about the political phenomenom, we use a sentiment analysis model to label the sentiment of a video, based again on text description and title. Our analysis is performed at video granularity and also channel granularity. 

## Research questions :question:
Our research aims to find the answer to a series of question that would help us understand the political polarization of Youtube:
* Is YouTube strictly politically unbiased or does it have its own preferences?
* Do democrats and republicans live in partisan bubbles or is there any conversation between the two sides?
* Is it a bad thing for a channel to change the direction of the videos that they are producing? Does this affects the overall engagement?

## Additional Datasets :mechanical_arm:
Given the fact that our analysis is more profund and requires a deep understanding of the content published on Youtube, we decided to use additional datasets:
1. In order to correctly label the setiment of each video we trained our political-flavoured BERT model on [Twitter and Reddit Sentimental analysis Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset ). We particularly used this dataset since we wanted to capture how people are talking on social platforms.
2. For political affiliation labelling we trained our political-flavoured BERT model on [Democrat Vs. Republican Tweets](https://www.kaggle.com/datasets/kapastor/democratvsrepublicantweets). We again particularly used this dataset because we wanted our model to understand how people are expressing political ideas on social platforms.
3. In order to be able to target our analysis only on US content, we needed extra channel information like location, which was optained from [Youtube Data API](https://developers.google.com/youtube/v3), ussing the `channel_id`. In addition to location, we also retrieved channel topic (since we found some differences between the label from the dataset and the actual topic of the channel) and also channel description. For Youtube API, there is a quota limit of 10,000 requests per day and a limit of maximum 50 channels per request. Since we had 136,470 channels, we send aprox. 2600 requests to Youtube API of 50 channels each, thus not passing the quata limit per day. 

## Methods

## Proposed deadline

## Organization within the team
