from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from geoalchemy2 import Geometry
from sqlalchemy.dialects.postgresql import JSONB
import os

# load env variables
load_dotenv()

# get absolute path to the templates directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

# initialize Flask
app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

# load configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# initialize database
db = SQLAlchemy(app)

################### DATABASE #################

# Define models
class SatelliteImageStats(db.Model):
    __tablename__ = 'satellite_image_stats'
    
    stats_id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255), nullable=False)
    acquisition_date = db.Column(db.DateTime, nullable=False)
    sensor_type = db.Column(db.String(50), nullable=False)
    resolution = db.Column(db.Float, nullable=False)
    cloud_cover_percentage = db.Column(db.Float)
    region_geometry = db.Column(Geometry('POLYGON', srid=4326))
    image_metadata = db.Column(JSONB)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    
    # Relationships
    spectral_statistics = db.relationship('SpectralStatistics', backref='satellite_image_stats', lazy=True, cascade="all, delete-orphan")
    spectral_index_statistics = db.relationship('SpectralIndexStatistics', backref='satellite_image_stats', lazy=True, cascade="all, delete-orphan")
    risk_assessments = db.relationship('RiskAssessment', backref='satellite_image_stats', lazy=True)

class SpectralStatistics(db.Model):
    __tablename__ = 'spectral_statistics'
    
    stat_id = db.Column(db.Integer, primary_key=True)
    stats_id = db.Column(db.Integer, db.ForeignKey('satellite_image_stats.stats_id', ondelete='CASCADE'), nullable=False)
    band_name = db.Column(db.String(50), nullable=False)
    band_number = db.Column(db.Integer)
    wavelength_nm = db.Column(db.Float)
    min_value = db.Column(db.Float)
    max_value = db.Column(db.Float)
    mean_value = db.Column(db.Float)
    median_value = db.Column(db.Float)
    std_dev = db.Column(db.Float)
    histogram_data = db.Column(JSONB)  # Store histogram as JSON
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

class SpectralIndexStatistics(db.Model):
    __tablename__ = 'spectral_index_statistics'
    
    index_stat_id = db.Column(db.Integer, primary_key=True)
    stats_id = db.Column(db.Integer, db.ForeignKey('satellite_image_stats.stats_id', ondelete='CASCADE'), nullable=False)
    index_name = db.Column(db.String(50), nullable=False)
    formula = db.Column(db.String(255))
    min_value = db.Column(db.Float)
    max_value = db.Column(db.Float)
    mean_value = db.Column(db.Float)
    median_value = db.Column(db.Float)
    std_dev = db.Column(db.Float)
    percentile_data = db.Column(JSONB)  # Store percentiles as JSON
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

class AnalysisZone(db.Model):
    __tablename__ = 'analysis_zones'
    
    zone_id = db.Column(db.Integer, primary_key=True)
    zone_name = db.Column(db.String(100), nullable=False)
    zone_type = db.Column(db.String(50))
    geometry = db.Column(Geometry('POLYGON', srid=4326))
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    
    # Relationships
    risk_assessments = db.relationship('RiskAssessment', backref='analysis_zone', lazy=True)

class RiskAssessment(db.Model):
    __tablename__ = 'risk_assessments'
    
    assessment_id = db.Column(db.Integer, primary_key=True)
    stats_id = db.Column(db.Integer, db.ForeignKey('satellite_image_stats.stats_id'))
    zone_id = db.Column(db.Integer, db.ForeignKey('analysis_zones.zone_id'))
    assessment_date = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    risk_level = db.Column(db.String(20), nullable=False)
    risk_score = db.Column(db.Float)
    confidence_level = db.Column(db.Float)
    weather_conditions = db.Column(JSONB)
    model_version = db.Column(db.String(50))
    summary = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    
    # Relationships
    simulations = db.relationship('FireSimulation', backref='risk_assessment', lazy=True, cascade="all, delete-orphan")

class FireSimulation(db.Model):
    __tablename__ = 'fire_simulations'
    
    simulation_id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey('risk_assessments.assessment_id'))
    simulation_parameters = db.Column(JSONB)
    simulation_results = db.Column(JSONB)  # Store simulation results directly as JSON
    time_horizon = db.Column(db.Integer)  # in hours
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

class ModelConfiguration(db.Model):
    __tablename__ = 'model_configurations'
    
    config_id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    model_version = db.Column(db.String(50), nullable=False)
    parameters = db.Column(JSONB)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

################### ROUTES ###################
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Handle the statistics data upload
        stats_data = request.json
        new_stats = SatelliteImageStats(
            image_name=stats_data['image_name'],
            acquisition_date=datetime.fromisoformat(stats_data['acquisition_date']),
            sensor_type=stats_data['sensor_type'],
            resolution=stats_data['resolution'],
            cloud_cover_percentage=stats_data.get('cloud_cover_percentage'),
            region_geometry=stats_data.get('region_geometry'),
            image_metadata=stats_data.get('metadata')
        )
        db.session.add(new_stats)
        db.session.commit()
        
        # Process spectral band statistics
        for band_data in stats_data.get('spectral_bands', []):
            band_stats = SpectralStatistics(
                stats_id=new_stats.stats_id,
                band_name=band_data['band_name'],
                band_number=band_data.get('band_number'),
                wavelength_nm=band_data.get('wavelength_nm'),
                min_value=band_data.get('min_value'),
                max_value=band_data.get('max_value'),
                mean_value=band_data.get('mean_value'),
                median_value=band_data.get('median_value'),
                std_dev=band_data.get('std_dev'),
                histogram_data=band_data.get('histogram_data')
            )
            db.session.add(band_stats)
        
        # Process spectral index statistics
        for index_data in stats_data.get('spectral_indices', []):
            index_stats = SpectralIndexStatistics(
                stats_id=new_stats.stats_id,
                index_name=index_data['index_name'],
                formula=index_data.get('formula'),
                min_value=index_data.get('min_value'),
                max_value=index_data.get('max_value'),
                mean_value=index_data.get('mean_value'),
                median_value=index_data.get('median_value'),
                std_dev=index_data.get('std_dev'),
                percentile_data=index_data.get('percentile_data')
            )
            db.session.add(index_stats)
        
        db.session.commit()
        return jsonify({'message': 'Statistics upload successful', 'stats_id': new_stats.stats_id})
    
    return render_template('upload.html')

@app.route('/analyze/<int:stats_id>')
def analyze(stats_id):
    stats = SatelliteImageStats.query.get_or_404(stats_id)
    return render_template('analysis.html', stats=stats)

# API Routes
@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = SatelliteImageStats.query.order_by(SatelliteImageStats.acquisition_date.desc()).all()
    return jsonify([{
        'stats_id': s.stats_id,
        'image_name': s.image_name,
        'acquisition_date': s.acquisition_date.isoformat(),
        'sensor_type': s.sensor_type,
        'risk_assessments': len(s.risk_assessments)
    } for s in stats])

@app.route('/api/stats/<int:stats_id>', methods=['GET'])
def get_stat_details(stats_id):
    stats = SatelliteImageStats.query.get_or_404(stats_id)
    spectral_stats = SpectralStatistics.query.filter_by(stats_id=stats_id).all()
    index_stats = SpectralIndexStatistics.query.filter_by(stats_id=stats_id).all()
    
    return jsonify({
        'stats_id': stats.stats_id,
        'image_name': stats.image_name,
        'acquisition_date': stats.acquisition_date.isoformat(),
        'sensor_type': stats.sensor_type,
        'resolution': stats.resolution,
        'cloud_cover_percentage': stats.cloud_cover_percentage,
        'spectral_bands': [{
            'band_name': s.band_name,
            'band_number': s.band_number,
            'min_value': s.min_value,
            'max_value': s.max_value,
            'mean_value': s.mean_value
        } for s in spectral_stats],
        'spectral_indices': [{
            'index_name': i.index_name,
            'min_value': i.min_value,
            'max_value': i.max_value,
            'mean_value': i.mean_value
        } for i in index_stats]
    })

@app.route('/api/chat', methods=["POST"])
def chat():
    # Chat implementation goes here
    query = request.json.get('query', '')
    # Use your NLP model to process the query
    response = {"answer": f"Processed query: {query}"}
    return jsonify(response)

# initialize database
def init_db():
    # enable postGIS
    with db.engine.connect() as conn:
        conn.execute(db.text('CREATE EXTENSION IF NOT EXISTS postgis'))
        conn.commit()
    # create tables
    db.create_all()

if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(debug=True)