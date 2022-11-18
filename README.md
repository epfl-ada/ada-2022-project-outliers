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

The first and most important aspect of our analysis was to be able to produce the political labels and sentiment labels. For this purpose, we trained a BERT model (more exactly [PolitBERT](https://huggingface.co/maurice/PolitBERT), in order to have a political-flavoured model) on Democrat Vs. Republican Tweets and Twitter and Reddit Sentimental analysis Dataset. After the models were trained, we were able to inpher for each video whici political and sentiment affilition it had, based on its text description and title. 

Since the discussion is relevant for the US population, we chose to only consider channels from US. In order to correctly identify the country of each channels, we used Youtube Data API to retrieve this additional information, based on channel id.

We found some inconsistencies among the categories for each channel, so we decided to retrieve this information form Youtube API. We filtred only channels from 'Politics' and 'Society', but our further analysis could focus on multiple categories.

#### :one: Is YouTube strictly politically unbiased or does it have its own preferences?
To investigate this, we tried to understand the distribution of republican and democratic content. We performend our analysis on video level and aggregated the results by month, in order to understand the trend revolution. 

![alt text](https://github.com/epfl-ada/ada-2022-project-outliers/blob/main/img/evolution_upload_density.png "Logo Title Text 1")

We can see that there is a clear tendency towards more republican content posted each month. However, in terms of reach, the results are surprising. Democratic content is able to gather definitely more engagement after Trump announces his candidancy for US president elections in June 2015.

![alt text](https://github.com/epfl-ada/ada-2022-project-outliers/blob/main/img/timeseries_likes.png "Logo Title Text 1")
![alt text](https://github.com/epfl-ada/ada-2022-project-outliers/blob/main/img/timeseries_likes.png "Logo Title Text 1")

#### :two: Do democrats and republicans live in partisan bubbles or is there any conversation between the two sides?

For this, we want to see how users interract. Do they only comment on a single type of video, or do they try to have a conversation with the other party? Are democrats more likely to comment to republicans videos? We can inpher if a user is republican or democrat by analysing the majority of the videos he/she commented on. 

We can also try to cluster videos. We can add weights between video that depends on whether or not the video was published by the same channel and the number of common commenters for both videos. In this way, we can see wheter or not we can cluster videos and what will describe these clusters.

## Proposed timeline

## Organization within the team
