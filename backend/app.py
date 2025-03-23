from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from geoalchemy2 import Geometry
from sqlalchemy.dialects.postgresql import JSONB
from gpt4all import GPT4All
import numpy as np
import json
import os

# load env variables
load_dotenv()

# get absolute path to the templates directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

# initialize Flask
app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

# Initialize GPT4ALL
model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# load configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://hwey:1234@localhost:5432/wildfire_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# initialize database
db = SQLAlchemy(app)

################### FUNCTIONS ################
# Function to generate chat responses
def generate_chat_response(query, context=None):
    # Create a system prompt with FireSight-specific knowledge
    system_prompt = """You are FireSight AI Assistant, an expert in wildfire risk assessment and management.
    You help analyze satellite imagery, vegetation indices, weather patterns, and terrain factors to assess wildfire risks.
    Provide concise, actionable information about wildfire risks and prevention strategies."""
    
    # Include context in the prompt if available
    if context:
        context_str = json.dumps(context)
        user_prompt = f"Context: {context_str}\n\nUser query: {query}"
    else:
        user_prompt = query
    
    # Full prompt with system and user parts
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    # Generate response using GPT4ALL
    response = model.generate(full_prompt, max_tokens=1024)
    
    return response.strip()

# Function to summarize risk assessments
def summarize_risk_data(risk_assessments):
    if not risk_assessments:
        return "No risk assessment data available."
    
    # Extract key information
    risk_levels = [assessment.risk_level for assessment in risk_assessments]
    risk_scores = [assessment.risk_score for assessment in risk_assessments if assessment.risk_score]
    
    # Count risk levels
    level_counts = {
        'Critical': risk_levels.count('Critical'),
        'High': risk_levels.count('High'),
        'Medium': risk_levels.count('Medium'),
        'Low': risk_levels.count('Low')
    }
    
    # Calculate averages
    avg_score = np.mean(risk_scores) if risk_scores else None
    
    # Get the most recent assessment
    recent = max(risk_assessments, key=lambda x: x.assessment_date)
    
    # Create a summary
    summary = (
        f"Summary of {len(risk_assessments)} risk assessments:\n"
        f"- Critical alerts: {level_counts['Critical']}\n"
        f"- High alerts: {level_counts['High']}\n"
        f"- Medium alerts: {level_counts['Medium']}\n"
        f"- Low alerts: {level_counts['Low']}\n"
    )
    
    if avg_score:
        summary += f"- Average risk score: {avg_score:.2f}\n"
    
    summary += (
        f"\nMost recent assessment ({recent.assessment_date.strftime('%Y-%m-%d')}):\n"
        f"- Risk level: {recent.risk_level}\n"
        f"- Risk score: {recent.risk_score:.2f}\n"
        f"- Area: {recent.analysis_zone.zone_name}\n"
    )
    
    return summary

# Function to summarize vegetation indices
def summarize_vegetation_indices(indices_data):
    if not indices_data:
        return "No vegetation index data available."
    
    # Extract key data
    ndvi_data = [idx for idx in indices_data if idx.index_name == 'NDVI']
    nbr_data = [idx for idx in indices_data if idx.index_name == 'NBR']
    ndwi_data = [idx for idx in indices_data if idx.index_name == 'NDWI']
    
    summary = "Vegetation Index Summary:\n"
    
    # Add NDVI summary
    if ndvi_data:
        ndvi = ndvi_data[0]
        health_status = "healthy" if ndvi.mean_value > 0.6 else "stressed" if ndvi.mean_value > 0.4 else "severely stressed"
        summary += (
            f"NDVI (vegetation health):\n"
            f"- Average: {ndvi.mean_value:.2f} ({health_status})\n"
            f"- Range: {ndvi.min_value:.2f} to {ndvi.max_value:.2f}\n"
        )
    
    # Add NBR summary
    if nbr_data:
        nbr = nbr_data[0]
        burn_status = "no burn" if nbr.mean_value > 0.3 else "moderate burn" if nbr.mean_value > 0.1 else "severe burn"
        summary += (
            f"NBR (burn ratio):\n"
            f"- Average: {nbr.mean_value:.2f} ({burn_status})\n"
            f"- Range: {nbr.min_value:.2f} to {nbr.max_value:.2f}\n"
        )
    
    # Add NDWI summary
    if ndwi_data:
        ndwi = ndwi_data[0]
        moisture_status = "adequate" if ndwi.mean_value > 0.4 else "moderate" if ndwi.mean_value > 0.2 else "low"
        summary += (
            f"NDWI (moisture):\n"
            f"- Average: {ndwi.mean_value:.2f} ({moisture_status} moisture)\n"
            f"- Range: {ndwi.min_value:.2f} to {ndwi.max_value:.2f}\n"
        )
    
    # Add recommendation
    if ndvi_data and nbr_data and ndwi_data:
        combined_risk = (1 - ndvi.mean_value) * 0.4 + (1 - nbr.mean_value) * 0.3 + (1 - ndwi.mean_value) * 0.3
        risk_level = "high" if combined_risk > 0.6 else "moderate" if combined_risk > 0.4 else "low"
        summary += f"\nOverall vegetation risk level: {risk_level.upper()}\n"
        
        if risk_level == "high":
            summary += "Recommendation: Immediate monitoring and preventive measures advised."
        elif risk_level == "moderate":
            summary += "Recommendation: Regular monitoring and preparedness advised."
        else:
            summary += "Recommendation: Standard monitoring protocols sufficient."
    
    return summary

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
        # Check if the request is a file upload or JSON stats
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle file upload
            if 'satellite_image' not in request.files:
                return jsonify({'error': 'No file part'}), 400
                
            file = request.files['satellite_image']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
                
            # Get form data
            acquisition_date = request.form.get('acquisition_date')
            sensor_type = request.form.get('sensor_type')
            resolution = request.form.get('resolution')
            region = request.form.get('region')
            
            # Save the file
            filename = file.filename
            file_path = os.path.join('backend', 'static', 'uploads', filename)
            os.makedirs(os.path.join('backend', 'static', 'uploads'), exist_ok=True)
            file.save(file_path)
            
            # Create a new satellite image stats record
            new_stats = SatelliteImageStats(
                image_name=filename,
                acquisition_date=datetime.fromisoformat(acquisition_date) if acquisition_date else datetime.now(timezone.utc),
                sensor_type=sensor_type or 'Unknown',
                resolution=float(resolution) if resolution else 0.0,
                cloud_cover_percentage=0.0,  # Would be calculated from the image
                image_metadata={'region': region}
            )
            db.session.add(new_stats)
            db.session.commit()
            
            return jsonify({'message': 'File upload successful', 'stats_id': new_stats.stats_id})
        else:
            # Handle JSON stats upload (existing code)
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
    query = request.json.get('query', '')
    context_type = request.json.get('context_type', None)
    context_id = request.json.get('context_id', None)
    
    # Get appropriate context based on request
    context = None
    if context_type == 'risk' and context_id:
        assessment = RiskAssessment.query.get(context_id)
        if assessment:
            context = {
                'risk_level': assessment.risk_level,
                'risk_score': assessment.risk_score,
                'assessment_date': assessment.assessment_date.isoformat(),
                'zone_name': assessment.analysis_zone.zone_name if assessment.analysis_zone else 'Unknown',
                'weather': assessment.weather_conditions
            }
    elif context_type == 'vegetation' and context_id:
        indices = SpectralIndexStatistics.query.filter_by(stats_id=context_id).all()
        if indices:
            context = {
                'indices': [
                    {
                        'name': idx.index_name,
                        'mean': idx.mean_value,
                        'min': idx.min_value,
                        'max': idx.max_value
                    } for idx in indices
                ]
            }
    
    # Handle special commands for summarization
    if query.lower().startswith('summarize'):
        if 'risk' in query.lower() or 'alerts' in query.lower():
            # Summarize risk assessments
            zone_id = context_id if context_type == 'zone' else None
            
            if zone_id:
                assessments = RiskAssessment.query.filter_by(zone_id=zone_id).all()
            else:
                assessments = RiskAssessment.query.order_by(RiskAssessment.assessment_date.desc()).limit(10).all()
            
            response = summarize_risk_data(assessments)
        
        elif 'vegetation' in query.lower() or 'indices' in query.lower() or 'ndvi' in query.lower():
            # Summarize vegetation indices
            stats_id = context_id if context_type == 'stats' else None
            
            if stats_id:
                indices = SpectralIndexStatistics.query.filter_by(stats_id=stats_id).all()
            else:
                # Get the most recent satellite image stats
                recent_stats = SatelliteImageStats.query.order_by(SatelliteImageStats.acquisition_date.desc()).first()
                indices = SpectralIndexStatistics.query.filter_by(stats_id=recent_stats.stats_id).all() if recent_stats else []
            
            response = summarize_vegetation_indices(indices)
        
        else:
            # General summary - combine multiple summaries
            recent_stats = SatelliteImageStats.query.order_by(SatelliteImageStats.acquisition_date.desc()).first()
            recent_assessments = RiskAssessment.query.order_by(RiskAssessment.assessment_date.desc()).limit(5).all()
            
            risk_summary = summarize_risk_data(recent_assessments)
            
            indices = []
            if recent_stats:
                indices = SpectralIndexStatistics.query.filter_by(stats_id=recent_stats.stats_id).all()
            veg_summary = summarize_vegetation_indices(indices)
            
            response = f"FireSight Dashboard Summary\n\n{risk_summary}\n\n{veg_summary}"
    
    else:
        # Regular chat response
        response = generate_chat_response(query, context)
    
    return jsonify({"answer": response})

@app.route('/api/summary', methods=["GET"])
def get_summary():
    summary_type = request.args.get('type', 'general')
    entity_id = request.args.get('id')
    
    if summary_type == 'risk':
        if entity_id:
            assessments = [RiskAssessment.query.get_or_404(entity_id)]
        else:
            assessments = RiskAssessment.query.order_by(RiskAssessment.assessment_date.desc()).limit(5).all()
        
        summary = summarize_risk_data(assessments)
    
    elif summary_type == 'vegetation':
        if entity_id:
            indices = SpectralIndexStatistics.query.filter_by(stats_id=entity_id).all()
        else:
            recent_stats = SatelliteImageStats.query.order_by(SatelliteImageStats.acquisition_date.desc()).first()
            indices = SpectralIndexStatistics.query.filter_by(stats_id=recent_stats.stats_id).all() if recent_stats else []
        
        summary = summarize_vegetation_indices(indices)
    
    else:  # general summary
        recent_stats = SatelliteImageStats.query.order_by(SatelliteImageStats.acquisition_date.desc()).first()
        recent_assessments = RiskAssessment.query.order_by(RiskAssessment.assessment_date.desc()).limit(5).all()
        
        risk_summary = summarize_risk_data(recent_assessments)
        
        indices = []
        if recent_stats:
            indices = SpectralIndexStatistics.query.filter_by(stats_id=recent_stats.stats_id).all()
        veg_summary = summarize_vegetation_indices(indices)
        
        summary = f"FireSight Dashboard Summary\n\n{risk_summary}\n\n{veg_summary}"
    
    return jsonify({"summary": summary})

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