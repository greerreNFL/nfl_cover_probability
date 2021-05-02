# nfl_cover_probability
Build off of nflfastR's Win Probability model to create a model for Cover Probability

A Very Simple Cover Probability Model
Like the Win Probability model, this very simple Cover Probability (CP) model uses XGBoost and nearly all the same features. In fact, no additional work went into finding a potentially better classifier or creating new features that might make the model more accurate.

The model uses these features...

spread_line
score_differential
spread_line_differential
posteam_is_home
half_seconds_remaining
game_seconds_remaining
ep
down
ydstogo
yardline_100
posteam_timeouts_remaining
defteam_timeouts_remaining
receive_2h_ko
Is_pat


To predict…

cover_result

The notable difference between the WP model and CP model are the substitution of ‘cover_result’ for ‘result’ as the dependent variable, and the addition of ‘spread_line_differential’ as a feature.

Cover_result is a binary value with 1 representing a cover and 0 representing a failure to cover. Spread_line_differential is the difference between a teams current margin and the spread--ie how many more points the team needs to cover the spread, or, if the team is already covering, how many points they are currently covering the spread by.

Some additional work was done to tune hyperparameters, which can be seen in the code posted here, but otherwise, that’s it. That’s the model.


Model Evaluation
The model performed reasonably well across training, test, and validation sets, showing little to no overfitting. It predicted the correct team to cover 74% of the time, which is slightly lower than the WP’s accuracy of 79%. Given that this model is meant to be a simple extension of the existing WP model to a new application, this falls very much within the territory of “good enough.”

<img width="798" alt="Screen Shot 2021-05-01 at 11 11 34 PM" src="https://user-images.githubusercontent.com/70054621/116805748-5c6be000-aadd-11eb-99ec-a9e5f9f8bb2f.png">

Visually, it’s also easy to see that the CP model gives a reasonably good estimation of a team’s chances to cover the spread:

<img width="1564" alt="Screen Shot 2021-05-01 at 9 20 58 PM" src="https://user-images.githubusercontent.com/70054621/116805760-6c83bf80-aadd-11eb-992a-3bfb7a65c71f.png">

There is certainly room for improvement, especially around the extreme values which overestimate cover probability of the leading team, but again, this is good enough.



