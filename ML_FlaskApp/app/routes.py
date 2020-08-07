import flask
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from .forms import LoginForm
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User, Post, Ecommerce
from app import db
from app.forms import RegistrationForm, EditProfileForm
from datetime import datetime
from app.forms import EmptyForm, PostForm
from sqlalchemy.orm import load_only
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import math
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import WhitespaceTokenizer
lemmatizer = WordNetLemmatizer()
w_tokenizer = WhitespaceTokenizer()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report
import colored 
import pickle
from app import app

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('base.html')

@app.route('/index', methods=['GET', 'POST'])
@login_required
def home():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(body=form.post.data, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your post is now live!')
        return redirect(url_for('home'))
    page = request.args.get('page', 1, type=int)
    posts = current_user.followed_posts().paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('home', page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('home', page=posts.prev_num) \
        if posts.has_prev else None
    return render_template('index.html', title='Home', form=form,
                           posts=posts.items, next_url=next_url,
                           prev_url=prev_url)
    
@app.route('/products', methods=['GET', 'POST'])
def products():
    ecommerce_title = Ecommerce.query.with_entities(Ecommerce.title)
    ecommerce_content = Ecommerce.query.with_entities(Ecommerce.content)
    ecommerce_image = Ecommerce.query.with_entities(Ecommerce.image)
    ecommerce_title
    ecommerce_titles = [j for sub in ecommerce_title for j in sub]
    i = 0
    
    sentiment_products = []
    for y in range(9):
        ecommerce_title1 = ''
        ecommerce_title1 = ecommerce_titles[y]
        products_sentiment = Post.query.filter_by(ecommerce_title=ecommerce_title1)
        sentiment_sc = 0
        x = 0   
        for review_score in products_sentiment:
            x += 1
            sentiment_sc += review_score.sentiment
        if (x==0):
            sentiment_products.append(int(round(sentiment_sc,0)))
        else:    
            sentiment_products.append(int(round(sentiment_sc/x,0)))

        

    

    return render_template('products.html', ecommerce_title=ecommerce_title, ecommerce_content=ecommerce_content, ecommerce_image=ecommerce_image, sentiment_products=sentiment_products, ecommerce_title1=ecommerce_title1)


@app.route('/product/<title>', methods=['GET', 'POST'])
@login_required
def product(title):
    global ecommerce_title,sentiment_sc
    product = Ecommerce.query.filter_by(title=title).first_or_404()
    product_sentiment = Post.query.filter_by(ecommerce_title=title)
    ecommerce_title = product.title
    ecommerce_content = product.content
    ecommerce_image = product.image
    form = PostForm()
    sentiment_sc = 0
    x = 0
    for review_score in product_sentiment:
        x += 1
        sentiment_sc += review_score.sentiment
    if (x==0):
        sentiment_sc = int(round(sentiment_sc,0))
    else:    
        sentiment_sc = int(round(sentiment_sc/x,0))
    review_count = x    

    if form.validate_on_submit():
        post = Post(body=form.post.data, ecommerce_title=ecommerce_title, author=current_user)
        db.session.add(post)
        db.session.commit()
        
        return redirect(url_for('product', title=ecommerce_title))
    page = request.args.get('page', 1, type=int)
    reviews = product.reviews.order_by(Post.timestamp.desc()).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('product', title=ecommerce_title, page=reviews.next_num) \
        if reviews.has_next else None
    prev_url = url_for('product', title=ecommerce_title, page=reviews.prev_num) \
        if reviews.has_prev else None
    
    return render_template('product.html', ecommerce_title=ecommerce_title, ecommerce_content=ecommerce_content, ecommerce_image=ecommerce_image, form=form, reviews=reviews.items, product=product, sentiment_sc=sentiment_sc, review_count=review_count, next_url=next_url, prev_url=prev_url)
    


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('products'))        
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('products')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('products')
        return redirect(url_for('products'))
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

# @app.route('/next')
# def next():
#   form = PostForm()
#   if form.validate_on_submit():
#       post = Post(body=form.post.data, author=current_user)
#       db.session.add(post)
#       db.session.commit()
#       flash('Your post is now live!')
#       return redirect(url_for('next'))      
        

#   posts = current_user.followed_posts().all()
#   return render_template('next.html', posts=posts, form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    page = request.args.get('page', 1, type=int)
    posts = user.posts.order_by(Post.timestamp.desc()).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('user', username=user.username, page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('user', username=user.username, page=posts.prev_num) \
        if posts.has_prev else None
    form = EmptyForm()
    
    return render_template('user.html', user=user, posts=posts.items,
                           next_url=next_url, prev_url=prev_url, form=form)
    

@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.now()
        db.session.commit()

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm()
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title='Edit Profile',
                           form=form)


@app.route('/follow/<username>', methods=['POST'])
@login_required
def follow(username):
    form = EmptyForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=username).first()
        if user is None:
            flash('User {} not found.'.format(username))
            return redirect(url_for('index'))
        if user == current_user:
            flash('You cannot follow yourself!')
            return redirect(url_for('user', username=username))
        current_user.follow(user)
        db.session.commit()
        
        return redirect(url_for('user', username=username))
    else:
        return redirect(url_for('index'))


@app.route('/unfollow/<username>', methods=['POST'])
@login_required
def unfollow(username):
    form = EmptyForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=username).first()
        if user is None:
            flash('User {} not found.'.format(username))
            return redirect(url_for('index'))
        if user == current_user:
            flash('You cannot unfollow yourself!')
            return redirect(url_for('user', username=username))
        current_user.unfollow(user)
        db.session.commit()
        
        return redirect(url_for('user', username=username))
    else:
        return redirect(url_for('index'))


@app.route('/explore')
@login_required
def explore():

    page = request.args.get('page', 1, type=int)
    posts = current_user.followed_posts().paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('explore', page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('explore', page=posts.prev_num) \
        if posts.has_prev else None
    
    return render_template('index.html', title='Explore',
                           posts=posts.items, next_url=next_url,
                           prev_url=prev_url)
    
   
    
       

@app.route('/predict', methods=['POST'])
@login_required
def predict():

    data_file = 'reviews_final.csv'
    data = pd.read_csv(data_file)

    stopwords_eng = stopwords.words('english')
    stopwords_eng2 = stopwords_eng
    stopwords_eng2 = [x.capitalize() for x in stopwords_eng2]
    stopwords_final = stopwords_eng + stopwords_eng2




    class ApplyRegex(BaseEstimator, TransformerMixin):
    
        def __init__(self, break_line=True, carriage_return=True, numbers=True, number_replacing='', special_char=True, additional_spaces=True):
            self.break_line = break_line
            self.carriage_return = carriage_return
            self.numbers = numbers
            self.number_replacing = number_replacing
            self.special_char = special_char
            self.additional_spaces = additional_spaces
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            X_transformed = []
            for c in X:
                if self.break_line:
                    c = re.sub('\n', ' ', c)
                if self.carriage_return:
                    c = re.sub('\r', ' ', c)
                if self.numbers:
                    c = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', self.number_replacing , c)
                if self.special_char:
                    c = re.sub(r'R\$', ' ', c)
                    c = re.sub(r'\W', ' ', c)
                if self.additional_spaces:
                    c = re.sub(r'\s+', ' ', c)
                X_transformed.append(c)
            return X_transformed
    
    class StopWordsRemoval(BaseEstimator, TransformerMixin):
    
        def fit(self, X, y=None):
            return self
        
        def stopword_removal(self):
            y=[]
            review_no_stopword = []
            for idx, review in enumerate(self) :
                try:
                    y=''
                    for word in review.split():
                        if word not in stopwords_final:
                            y+= word + ' '
                    review_no_stopword.append(y)
                except:
                    print(idx)
            return review_no_stopword
            
        def transform(self, X, y=None): 
            X_transformed = StopWordsRemoval.stopword_removal(X)
            return X_transformed


    
    class TextLemmatization(BaseEstimator, TransformerMixin):

        def fit(self, X, y=None):
            return self
        
        def lemmatize_text(text):
            return ' '.join(lemmatizer.lemmatize(w, pos="v") for w in w_tokenizer.tokenize(text))


        def transform(self, X, y=None):
            X_transformed = list(map(lambda c: TextLemmatization.lemmatize_text(c), X))
            return X_transformed

    preprocess_pipeline = Pipeline([
    ('regex_cleaner', ApplyRegex()),
    ('stopwords_remover', StopWordsRemoval()),
    ('lemmatization', TextLemmatization()),
    ])

    X = data['reviews.text']
    y = data['reviews.rating'].values
    y = y.astype(int)

    X_preprocessed = preprocess_pipeline.fit_transform(X)
    reviews_vector = list(map(lambda c: nltk.word_tokenize(c), X_preprocessed))
    vectorizer = CountVectorizer(max_features=300)
    X_transformed = vectorizer.fit_transform(X_preprocessed).toarray()

    review_txt = request.form.get('review')
    review = request.form.values()
    preprocessed_review = preprocess_pipeline.fit_transform(review)
    final_review = vectorizer.transform(preprocessed_review).toarray()
    prediction = model.predict(final_review)


    # Sentiment Score Predictor
    # text_preprocessed = preprocess_pipeline.fit_transform([review_txt])
    # text_transformed = vectorizer.transform(text_preprocessed).toarray()
    # review_proba = model.predict_proba(text_transformed)
    # sentiment_score = round(review_proba[0,2]*100,2)
    # sentiment_score = sentiment_score.item()

    form = PostForm()
    if form.validate_on_submit():
        
        text_preprocessed = preprocess_pipeline.fit_transform([form.post.data])
        text_transformed = vectorizer.transform(text_preprocessed).toarray()
        review_proba = model.predict_proba(text_transformed)
        sentiment_score = round(review_proba[0,2]*100,2)
        sentiment_score = sentiment_score.item()
        post = Post(body=form.post.data, ecommerce_title=ecommerce_title, author=current_user, sentiment=sentiment_score)
        db.session.add(post)
        db.session.commit()
        return redirect(url_for('product', title=ecommerce_title))
    
    page = request.args.get('page', 1, type=int)
    reviews = product.reviews.order_by(Post.timestamp.desc()).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('product', title=ecommerce_title, page=reviews.next_num) \
        if reviews.has_next else None
    prev_url = url_for('product', title=ecommerce_title, page=reviews.prev_num) \
        if reviews.has_prev else None


    return render_template('product.html', prediction_text='Sentiment Score : ', prediction_score='{}%'.format(sentiment_score), review_text='Your review: {}'.format(review_txt), reviews=reviews.items)





if __name__ == "__main__":
    app.run(debug=True)