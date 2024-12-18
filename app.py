from flask import Flask, render_template, request
import pandas as pd

# Assuming your hybrid recommendation function and data are already loaded
# You can either load a DataFrame here or use the existing one.

app= Flask(__name__)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_filtering_recommendations(df1, target_user_id, top_n=10):
    # Create the user-item matrix
    user_item_matrix = df1.pivot_table(index='ID',columns='ProdID',values='Rating',aggfunc='mean').fillna(0).astype(int)
    # Calculate the user similarity matrix using cosine similarity
    user_similarities = cosine_similarity(user_item_matrix)

    # Find the index of the target user in the matrix
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    # Get the similarity scores for the target user
    target_user_similarity = user_similarities[target_user_index]

    # Sort the users by similarity in descending order (excluding the target user)
    similar_user_index = target_user_similarity.argsort()[::-1][1:]

    # Generate recommendations based on similar users
    recommended_items = []

    for user_index in similar_user_index:
        # Get items rated by the similar user but not by the target user
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)

        # Extract the item IDs of recommended items
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    # Get the details of recommended items
    recommended_items_details = df1[df1['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details.head(10)

def content_based_recommendations(df1,item_name , top_n = 10):
    #check if the item name dont exist in the training data
    if item_name not in df1['Name'].values:
        print(f"item {item_name} not found in training data")

    tfidf_vectorizer = TfidfVectorizer(stop_words='english') # tfidf will not apply on stop_words
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(df1['Tags']) # converted text data in to vectors in numerical form

    # step2 calculate cosine similarity of tag column  between each and every row/document of tag column
    cosine_sim_content = cosine_similarity(tfidf_matrix_content,tfidf_matrix_content)

    # testing the recommendations are coming proper

    #find the index of the searched item
    item_index = df1[df1['Name']==item_name].index[0]

    #get the cosine similarity score of item
    similar_items = list(enumerate(cosine_sim_content[item_index]))

    #sort the similar items by similarity score in descending order to get higher similar scores
    similar_items = sorted(similar_items,key = lambda x: x[1],reverse=True)

    # get the top n most similar items (excluding itself)
    top_sim_items = similar_items[1:top_n+1]

    #getting the index of top most similar items
    recommend_item_index = [x[0] for x in top_sim_items]

    # get the details of the similar items
    recommend_item_details = df1.iloc[recommend_item_index][['Name','ReviewCount','Brand','ImageURL', 'Rating']]
    return recommend_item_details.head(10)

# Example function for hybrid recommendations
def hybrid_recommendations(df1, target_user_id, item_name, top_n=10):
    # Get content-based recommendations (dummy data for illustration)
    content_based_rec = content_based_recommendations(df1,item_name , top_n)

    # Get collaborative filtering recommendations (dummy data for illustration)
    collaborative_filtering_rec = collaborative_filtering_recommendations(df1,target_user_id,top_n)

    # Merge both recommendation lists and remove duplicates
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()

    return hybrid_rec.head(top_n)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        target_user_id = int(request.form['user_id'])
        item_name = request.form['item_name']

        df1 = pd.read_csv("models/df1.csv")

        # Get hybrid recommendations
        hybrid_rec = hybrid_recommendations(df1, target_user_id, item_name, top_n=10)

        # Convert recommendations to a list for display
        recommendations = hybrid_rec['Name'].tolist()

        return render_template('index.html', recommendations=recommendations, target_user_id=target_user_id,
                               item_name=item_name)

    return render_template('index.html', recommendations=None)


if __name__ == '__main__':
    app.run(debug=True , port = 5002)
