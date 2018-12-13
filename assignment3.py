import pandas as pd
import numpy as np
import warnings
from flask import Flask, render_template, request
warnings.filterwarnings('ignore')
app = Flask(__name__)
movieRating_variable_one = []

#Reading users file:
userCols = ['user id','movie id','rating','timestamp']
users = pd.read_csv('ml-100k/u.data', sep='\t', names=userCols,encoding='latin-1')

movieCols = ['movie id','movie title','release date','video release date',
              'IMDb URL', 'unknown','Action','Adventure','Animation',
              'Childrens', 'Comedy','Crime','Documentary','Drama','Fantasy',
              'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi',
              'Thriller','War','Western']
movieTitles = pd.read_csv('ml-100k/u.item', sep='|', names=movieCols,encoding='latin-1')


movieTitles.drop(["video release date","IMDb URL","unknown","Action","Adventure","Animation","Childrens","Comedy","Crime","Documentary","Drama","Fantasy",
              "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
              "Thriller","War","Western"], axis = 1, inplace = True) 


users = pd.merge(users, movieTitles, on='movie id')
users.head()
matrix = users.pivot_table(index='user id', columns='movie title', values='rating',aggfunc=np.mean)


ratings = pd.DataFrame(users.groupby('movie title')['rating'].mean())
ratings.head()
ratings['number_of_ratings'] = users.groupby('movie title')['rating'].count()
ratings.head()

sort=ratings.sort_values('number_of_ratings', ascending=False).head(10)

@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template ("index.html")

@app.route('/result', methods=['GET', 'POST'])
def result():
	if request.form.get('submit') == 'submit':
		movie = request.form.get('movie')
		print(movie)
		movieRating_variable = movieRating(movie)
		a = []
		for index,row in movieRating_variable.iterrows():
			temp = []
			temp.append(index)
			temp.append(str(round(int(row['correlation']), 3)))
			temp.append(int(row['number_of_ratings']))
			a.append(temp)

		return render_template('index.html', data = a)

def movieRating(movie):
	userMovieRating = matrix[movie]
	similarity=matrix.corrwith(userMovieRating)

	correlate = pd.DataFrame(similarity, columns=['correlation'])
	correlate.dropna(inplace=True)
	correlate.head()

	correlate = correlate.join(ratings['number_of_ratings'])

	correlateMax = correlate[correlate['number_of_ratings'] > 50].sort_values(by='correlation', ascending=False).head(10)

	return correlateMax

if __name__ == '__main__':
	app.run(host='127.0.0.1',port =5000,debug=True)





