from flask import Flask, render_template, request, jsonify, send_from_directory, current_app
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import os
import logging
import pickle
import numpy as np
from functools import lru_cache
import time
from typing import List, Tuple, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MovieRecommendationEngine:
    """Movie recommendation engine with clustering and similarity calculations"""
    
    def __init__(self, data_path: str, cache_dir: str = 'cache'):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.df = None
        self.tfidf = None
        self.X = None
        self.X_reduced = None
        self.kmeans = None
        self.title_mapping = {}
        
        os.makedirs(cache_dir, exist_ok=True)
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load cached model or create new one"""
        cache_file = os.path.join(self.cache_dir, 'recommendation_model.pkl')
        
        try:
            # Check if cache exists and is newer than data file
            if (os.path.exists(cache_file) and 
                os.path.getmtime(cache_file) > os.path.getmtime(self.data_path)):
                logger.info("Loading cached model...")
                self._load_cached_model(cache_file)
            else:
                logger.info("Creating new model...")
                self._create_model()
                self._cache_model(cache_file)
        except Exception as e:
            logger.error(f"Error with model loading/creation: {e}")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            self._create_model()
    
    def _create_model(self):
        """Create the recommendation model from scratch"""
        logger.info("Loading and preprocessing data...")
        
        # Load and clean data
        self.df = pd.read_csv(self.data_path)
        original_count = len(self.df)
        
        # Handle missing values more intelligently
        self.df = self.df.fillna('')
        self.df = self.df[self.df['title'].str.strip() != '']  # Remove empty titles
        
        # Netflix-specific data cleaning
        self.df['cast'] = self.df['cast'].str.replace('â€™', "'", regex=False)  # Fix encoding issues
        self.df['description'] = self.df['description'].str.replace('â€™', "'", regex=False)
        
        logger.info(f"Loaded {len(self.df)} titles (filtered from {original_count})")
        
        # Create feature combinations with Netflix-specific fields
        feature_cols = ['title', 'director', 'cast', 'country', 'listed_in', 'description', 'type']
        self.df['combined'] = self.df[feature_cols].agg(' '.join, axis=1)
        
        # Create title mapping for fuzzy matching
        self.title_mapping = {
            title.lower().strip(): idx 
            for idx, title in enumerate(self.df['title']) 
            if title.strip()
        }
        
        # TF-IDF Vectorization with optimized parameters
        logger.info("Creating TF-IDF vectors...")
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.X = self.tfidf.fit_transform(self.df['combined'])
        
        # Dimensionality reduction
        logger.info("Performing dimensionality reduction...")
        n_components = min(300, self.X.shape[1], self.X.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.X_reduced = self.svd.fit_transform(self.X)
        
        # Clustering
        logger.info("Performing clustering...")
        k = min(15, len(self.df) // 10)  # Adaptive cluster count
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(self.X_reduced)
        
        logger.info(f"Model created with {k} clusters")
    
    def _cache_model(self, cache_file: str):
        """Cache the trained model"""
        try:
            cache_data = {
                'df': self.df,
                'tfidf': self.tfidf,
                'X_reduced': self.X_reduced,
                'svd': self.svd,
                'kmeans': self.kmeans,
                'title_mapping': self.title_mapping
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info("Model cached successfully")
        except Exception as e:
            logger.error(f"Failed to cache model: {e}")
    
    def _load_cached_model(self, cache_file: str):
        """Load cached model"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.df = cache_data['df']
        self.tfidf = cache_data['tfidf']
        self.X_reduced = cache_data['X_reduced']
        self.svd = cache_data['svd']
        self.kmeans = cache_data['kmeans']
        self.title_mapping = cache_data['title_mapping']
        
        logger.info("Cached model loaded successfully")
    
    def _find_best_title_match(self, title: str) -> Optional[int]:
        """Find best matching title using fuzzy matching"""
        title_lower = title.lower().strip()
        
        # Exact match first
        if title_lower in self.title_mapping:
            return self.title_mapping[title_lower]
        
        # Fuzzy matching
        best_match = None
        best_ratio = 0.6  # Minimum similarity threshold
        
        for stored_title, idx in self.title_mapping.items():
            ratio = SequenceMatcher(None, title_lower, stored_title).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = idx
        
        return best_match
    
    @lru_cache(maxsize=1000)
    def get_recommendations(self, title: str, n: int = 5) -> List[dict]:
        """Get movie recommendations with caching"""
        start_time = time.time()
        
        try:
            # Find matching movie
            idx = self._find_best_title_match(title)
            if idx is None:
                logger.warning(f"No match found for title: {title}")
                return []
            
            # Get cluster and similar movies
            cluster = self.df.at[idx, 'cluster']
            cluster_members = self.df[self.df['cluster'] == cluster].index.tolist()
            
            # Calculate similarities within cluster
            target_vector = self.X_reduced[idx].reshape(1, -1)
            cluster_vectors = self.X_reduced[cluster_members]
            similarities = cosine_similarity(target_vector, cluster_vectors)[0]
            
            # Rank by similarity
            ranked_indices = [
                cluster_members[i] for i in np.argsort(similarities)[::-1]
                if cluster_members[i] != idx
            ][:n]
            
            # Prepare recommendations with Netflix-specific metadata
            recommendations = []
            for rec_idx in ranked_indices:
                movie = self.df.loc[rec_idx]
                rec = {
                    'title': movie['title'],
                    'year': int(movie['release_year']) if movie['release_year'] and str(movie['release_year']).isdigit() else None,
                    'type': movie['type'],
                    'rating': movie.get('rating', ''),
                    'duration': movie.get('duration', ''),
                    'genre': movie.get('listed_in', '').split(',')[0].strip() if movie.get('listed_in') else '',
                    'director': movie.get('director', '').split(',')[0].strip() if movie.get('director') else '',
                    'country': movie.get('country', '').split(',')[0].strip() if movie.get('country') else '',
                    'description': movie.get('description', '')[:100] + '...' if len(movie.get('description', '')) > 100 else movie.get('description', ''),
                    'similarity': round(float(similarities[cluster_members.index(rec_idx)]), 3)
                }
                recommendations.append(rec)
            
            processing_time = time.time() - start_time
            logger.info(f"Generated {len(recommendations)} recommendations for '{title}' in {processing_time:.3f}s")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for '{title}': {e}")
            return []
    
    def get_stats(self) -> dict:
        """Get Netflix dataset statistics"""
        if self.df is None:
            return {}
        
        # Get content type breakdown
        type_counts = self.df['type'].value_counts().to_dict()
        
        # Get rating distribution
        rating_counts = self.df['rating'].value_counts().head(5).to_dict()
        
        # Get top genres
        all_genres = []
        for genres in self.df['listed_in'].fillna(''):
            all_genres.extend([g.strip() for g in genres.split(',') if g.strip()])
        genre_counts = pd.Series(all_genres).value_counts().head(5).to_dict()
        
        # Get top countries
        all_countries = []
        for countries in self.df['country'].fillna(''):
            all_countries.extend([c.strip() for c in countries.split(',') if c.strip()])
        country_counts = pd.Series(all_countries).value_counts().head(5).to_dict()
        
        return {
            'total_titles': len(self.df),
            'content_types': type_counts,
            'clusters': len(self.df['cluster'].unique()) if 'cluster' in self.df.columns else 0,
            'top_genres': genre_counts,
            'top_countries': country_counts,
            'top_ratings': rating_counts,
            'years_range': {
                'min': int(self.df['release_year'].min()) if self.df['release_year'].notna().any() else None,
                'max': int(self.df['release_year'].max()) if self.df['release_year'].notna().any() else None
            } if 'release_year' in self.df.columns else None
        }

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize recommendation engine
try:
    # Try relative path first (for your dataset)
    data_path = os.path.join(os.path.dirname(__file__), 'netflix_titles.csv')
    if not os.path.exists(data_path):
        # Try other common locations
        possible_paths = [
            'netflix_titles.csv',
            os.path.join(os.path.dirname(__file__), 'netflix_titles.csv'),
            r'C:\Users\krish\OneDrive\Desktop\python\MRS\netflix_titles.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        else:
            raise FileNotFoundError("Netflix dataset not found in any expected location")
    
    recommendation_engine = MovieRecommendationEngine(data_path)
    logger.info("Netflix recommendation engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize recommendation engine: {e}")
    recommendation_engine = None

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(current_app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get dataset statistics"""
    if recommendation_engine is None:
        return jsonify({'error': 'Recommendation engine not available'}), 503
    
    stats = recommendation_engine.get_stats()
    return jsonify(stats)

@app.route('/api/search')
def search_movies():
    """Search for movies by partial title"""
    if recommendation_engine is None:
        return jsonify({'error': 'Recommendation engine not available'}), 503
    
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'movies': []})
    
    query_lower = query.lower()
    matches = recommendation_engine.df[
        recommendation_engine.df['title'].str.lower().str.contains(query_lower, na=False)
    ]['title'].head(10).tolist()
    
    return jsonify({'movies': matches})

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get movie recommendations"""
    if recommendation_engine is None:
        return jsonify({
            'error': 'Recommendation engine not available',
            'recommendations': []
        }), 503
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'recommendations': []
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Invalid JSON data',
                'recommendations': []
            }), 400
        
        title = data.get('title', '').strip()
        if not title:
            return jsonify({
                'error': 'Title is required',
                'recommendations': []
            }), 400
        
        n_recommendations = min(int(data.get('count', 5)), 20)  # Max 20 recommendations
        
        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(title, n_recommendations)
        
        # Format response
        response = {
            'recommendations': recommendations,
            'query': title,
            'count': len(recommendations)
        }
        
        if not recommendations:
            response['message'] = f"No recommendations found for '{title}'. Try a different movie title."
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({
            'error': f'Invalid input: {str(e)}',
            'recommendations': []
        }), 400
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {e}")
        return jsonify({
            'error': 'Failed to generate recommendations',
            'recommendations': []
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'engine_available': recommendation_engine is not None,
        'timestamp': time.time()
    }
    
    if recommendation_engine:
        status['dataset_size'] = len(recommendation_engine.df)
    
    return jsonify(status)

if __name__ == '__main__':
    # Development server
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )