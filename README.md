##Context
nflfastR, and it’s predecessor nflscrapR, opened NFL play-by-play (PBP) data to the analytics community. This data enabled foundational models like Expected Points Added (EPA) and Win Probability Added (WPA), which are now used to assess teams and players through a more quantitative lens.

These models are completely open source, which means they can be augmented and extended to new applications. This post will use the work of nflfastR’s Win Probability (WP) model to answer a new question--what is the probability that a team covers their pre-game spread given the current game state?

##A Very Simple Cover Probability Model
Like the WP model, this very simple Cover Probability (CP) model uses XGBoost and nearly all the same features. In fact, no additional work went into finding a potentially better classifier, and little work was done to create new features to tailor the model for predicting cover probability.

The model uses these features...

* spread_time
* score_differential
* diff_time_ratio
* posteam_is_home
* half_seconds_remaining
* game_seconds_remaining
* down
* ydstogo
* yardline_100
* posteam_timeouts_remaining
* defteam_timeouts_remaining
* receive_2h_ko
* is_pat [new]
* spread_line_differential [new]

To predict…

* cover_result

The notable difference between the WP model and CP model are the substitution of ‘cover_result’ for ‘result’ as the dependent variable, and the addition of ‘spread_line_differential’ as a feature.

Cover_result is a binary value with 1 representing a cover and 0 representing a failure to cover. Spread_line_differential is the difference between a team’s current margin and the spread--ie how many points the team needs to cover the spread, or, if the team is already covering, how many points they are currently covering the spread by.

Some additional work was done to tune hyperparameters, which can be seen in the code posted here, but otherwise, that’s it. That’s the model.

Model Evaluation
The model performed reasonably well across training, validation, and test sets, showing little to no overfitting. It predicted the correct team to cover 74% of the time, which is slightly lower than the WP’s accuracy of 79%. Given that this model is meant to be a simple extension of the existing WP model to a new application, this falls very much within the territory of “good enough.”

<img width="798" alt="Screen Shot 2021-05-01 at 11 11 34 PM" src="https://user-images.githubusercontent.com/70054621/116805748-5c6be000-aadd-11eb-99ec-a9e5f9f8bb2f.png">

Visually, it’s also easy to see that the CP model gives a reasonably good estimation of a team’s chances to cover the spread:

<img width="1564" alt="Screen Shot 2021-05-01 at 9 20 58 PM" src="https://user-images.githubusercontent.com/70054621/116805760-6c83bf80-aadd-11eb-992a-3bfb7a65c71f.png">

There is certainly room for improvement, especially around the extreme values which overestimate cover probability of the leading team, but again, this is good enough.

Using the Model
For those interested in exploring or expanding the model, all code is posted on Github here. It uses standard play-by-play files and shouldn’t need any custom setup outside of the existing code.

For those interested in seeing graphs for individual games, all game pages on [nfeloapp.com](http://www.nfeloapp.com) now have a section for both Win Probability and Cover Probability that look like this:

<img width="642" alt="Screen Shot 2021-05-09 at 1 30 17 AM" src="https://user-images.githubusercontent.com/70054621/117596062-54242e00-b0f7-11eb-8535-ec6066c99b65.png">

<img width="639" alt="Screen Shot 2021-05-09 at 1 30 25 AM" src="https://user-images.githubusercontent.com/70054621/117596070-58e8e200-b0f7-11eb-9ed1-1b2ba291034f.png">


