# Yelp Recommender System using Xgboost

### Method Description
I have worked on my Model-Based Recommender System, leveraging XGBoost to predict user ratings for businesses using the Yelp Reviews Dataset.

To enhance the model, I increased the number of features extracted from user.json, business.json, tip.json, and photo.json. While I considered using features from review_train.json, I ultimately decided against it since the validation set lacked corresponding review-related features. To address the cold-start problem, I assigned average rating values of 3.76 for new users and 3.75 for new businesses, based on overall dataset averages. This adjustment reduced the RMSE by 0.0035. Additionally, including the state feature led to a further RMSE reduction of 0.0007. After consolidating these features into a NumPy array, I applied MinMax normalization to scale them effectively. These improvements brought the RMSE down from 0.9852 to 0.9810 in this project.

To optimize the model further, I employed Optuna for hyperparameter tuning with an extensive search space and 2-fold cross-validation. This approach reduced the RMSE to 0.9753 on the validation set. I also experimented with a Gradient Boosting Regressor model using the same preprocessing steps and explored hybrid approaches by combining item-based collaborative filtering with the XGBoost model. Despite these attempts, the XGBoost solution provided the best results for the final submission.

### Error Distribution
| **Error Range** | **Total Records** |
|------------------|-------------------|
| >=0 and <1       | 102,682           |
| >=1 and <2       | 32,446            |
| >=2 and <3       | 6,090             |
| >=3 and <4       | 824               |
| >=4              | 2                 |


### RMSE Scores
| **Test Set** | **RMSE Value** |
|------------------|-------------------|
| Validation RMSE  | 0.9753088256169121|
| Test RMSE        | 0.973560847436848 |
