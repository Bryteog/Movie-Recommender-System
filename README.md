## Overview
Content-based and Collaborative Recommender systems are built using the MovieLens small [dataset](https://grouplens.org/datasets/movielens/latest/). These systems were built using the Pandas libraries and Neural Networks. Movie suggestion can be made based on user similarity, movie popularity, and user ratings for specific movie and genres.


The original dataset has roughly 9000 movies rated by 600 users with ratings on a scale of 0.5 to 5 in 0.5 step increments. The dataset has been reduced in size to focus on movies from the years since 2000 and popular genres. The reduced dataset has $n_u = 397$ users, $n_m= 847$ movies and 25521 ratings. 
For each movie, the dataset provides a movie title, release date, and one or more genres. For example "Toy Story 3" was released in 2010 and has several genres: "Adventure | Animation | Children | Comedy | Fantasy". This dataset contains little information about users other than their ratings.


#### Collaborative filtering Systems
These systems generate recommendations using only information about rating profiles for different users or items. The items recommended are done so due to users previous behaviour and choices. The model recommends items in close proximity to those the user has already expressed interest in.


#### Content-based Systems
These systems take into consideration the features of the item and items with similar properties in addition to the users preferences. Features are extracted from an item or from the user history and then a decision on what to recommend is made.

