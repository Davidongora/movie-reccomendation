from surprise import Dataset, Reader, SVD
import pandas as pd

df = pd.read_csv('data.txt')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['User', 'Movie', 'Rating']], reader)

algo = SVD()

trainset = data.build_full_trainset()
algo.fit(trainset)

def get_top_n_recommendations(user_id, n=5):
    user_movies = df[df['User'] == user_id]['Movie'].tolist()
    movies_to_predict = list(set(df['Movie']) - set(user_movies))
    
    testset = [[user_id, movie, 4] for movie in movies_to_predict] 
    predictions = algo.test(testset)
    
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    top_movies = [pred.iid for pred in top_n]
    return top_movies

while True:
    user_input = input("Enter your name (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    else:
        recommendations = get_top_n_recommendations(user_input)
        print(f"Hello {user_input}! Here are your top movie recommendations: {recommendations}")
