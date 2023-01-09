import  streamlit as st

import pickle
import pandas as pd
import requests
import surprise
import pandas as pd
import numpy as np
import heapq



st.title('Movie Recommender System')



def recommend(movie):
    movie_index = movies[movies['Title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

    recommended_movies = []
 

    for i in movies_list[1:11]:
        recommended_movies.append(movies.iloc[i[0]].Title)

    return recommended_movies

movies_dict = pickle.load(open('movies.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl','rb'))






######################### Collaborative #############################

users_dict = pickle.load(open('users1.pkl','rb'))
ratings = pd.DataFrame(users_dict)


def for_user(users):
    lowest_rating = ratings['rating'].min()
    highest_rating = ratings['rating'].max()

    reader = surprise.Reader(rating_scale = (lowest_rating,highest_rating))
    data = surprise.Dataset.load_from_df(ratings,reader)

    similarity_options = {'name': 'cosine', 'user_based': True}
    algo = surprise.KNNBasic(sim_options = similarity_options)
    output = algo.fit(data.build_full_trainset())
    
    df_50 = ratings[ratings['uid']==u_Id]
    items_50 = df_50['iid']

    all_items = ratings['iid'].unique()
    
    iids_to_predict = np.setdiff1d(all_items,items_50)
# Finding the expected ratings
    testset = [[u_Id,movieId,0.] for movieId in iids_to_predict]

    predictions = algo.test(testset)
    
    pred_ratings = np.array([pred.est for pred in predictions])
    i_max = np.argmax(pred_ratings)
    i_sorted = heapq.nlargest(10,range(len(pred_ratings)),
                          pred_ratings.take)

    rec=iids_to_predict[i_sorted]
    recommended=[]
    for i in rec:
        recommended.append(movies.loc[movies['Movie ID']==i].Title)
    m_list=[]
    for j in recommended:    
         m_list.append(j.values[0])
    return m_list
    
    
################################# Hybrid ##########################################

dff_df = pickle.load(open('dff.pkl','rb'))
dff = pd.DataFrame(dff_df)


merged = pickle.load(open('merged_df.pkl','rb'))
merged_df = pd.DataFrame(merged)

hybrid_df = pickle.load(open('hybrid.pkl','rb'))
user_movie_df = pd.DataFrame(hybrid_df)

def for_hybrid(user_id):
    user_df = user_movie_df[user_movie_df.index == user_id]
    movies_watched = user_df.columns[user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    m_count = movies_watched_df.shape[1]
    
    users_same_movies=user_movie_count[user_movie_count["movie_count"]/m_count > 0.6].sort_values("movie_count", ascending=False)
    
    final_df2 = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      user_df[movies_watched]])
    
    corr_df = final_df2.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()
    
    top_users = corr_df[(corr_df["user_id_1"] == user_id) & (corr_df["corr"] >= 0.60)][
    ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)

# Users who have more than 0.60 corr with the user
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
##    
    top_users_ratings = top_users.merge(dff[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    top_users_ratings['weighted_rating'] = top_users_ratings['weighted_rating'].astype(float)
    
    temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
    temp.columns = ['sum_corr', 'sum_weighted_rating']
    
    recommendation_df = pd.DataFrame()
    recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
    recommendation_df['movieId'] = temp.index
    recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)
##    
    final=merged_df.loc[merged_df['movieId'].isin(recommendation_df.head(10)['movieId'])]
    
    final_recommendations=final.groupby("title").agg({"rating": "max"}).sort_values("rating", ascending=False)
    
    final_recommendations.reset_index(inplace=True)
    
    m=[]
    for i in final_recommendations['title']:
        m.append(i)
    final_list = m[:11]
        
    return final_list


    





    
########################### Streamlit-UI ###############################       
tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative", "Hybrid"])

with tab1:
   st.header("Content-Based")
   selected_movie_name = st.selectbox(
       "Type or select a movie from the dropdown",
       movies['Title'].values
   )
   if st.button('Show Recommendation'):
       names = recommend(selected_movie_name)
       st.dataframe(names)

with tab2:
   st.header("Collaborative")
   u_Id = st.selectbox(
       "Type or select a user id from the dropdown",
       ratings['uid'].unique()
   )
   if st.button('Show User Based Recommendation'):
       names1 = for_user(u_Id)
       st.dataframe(names1)

with tab3:
   st.header("Hybrid")
   hybrid_u = st.selectbox(
       "Type or select a user id from the dropdown",
       user_movie_df.index
   )
   if st.button('Show Hybrid Recommendation'):
       names2 = for_hybrid(hybrid_u)
       st.dataframe(names2)
        
        
        
        