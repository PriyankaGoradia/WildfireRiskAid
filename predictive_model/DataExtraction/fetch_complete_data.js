// ------------------------------------------
// Wildfire Dataset Feature Extraction Script (Alberta)
// ------------------------------------------

// 1. Define Region of Interest (Alberta wildfire zone)
var roi = ee.Geometry.Rectangle([-117.5, 52.0, -113.0, 55.0]);

// 2. Define Time Range (Wildfire season)
var startDate = '2023-05-01';
var endDate = '2023-09-30';

// 3. Sentinel-2 for NDVI, NBR, NDWI
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(function(img) {
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    var nbr  = img.normalizedDifference(['B8', 'B12']).rename('NBR');
    var ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI');
    return img.addBands([ndvi, nbr, ndwi])
              .select(['NDVI', 'NBR', 'NDWI'])
              .copyProperties(img, ['system:time_start']);
  });

var medianSpectral = s2.median();

// 4. ERA5-Land Hourly Weather Data (Temperature, Wind, Humidity)
var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .select(['temperature_2m', 'u_component_of_wind_10m', 'v_component_of_wind_10m', 'dewpoint_temperature_2m'])
  .mean();  // mean over time

// Convert temperature from Kelvin to Celsius
var tempC = era5.select('temperature_2m').subtract(273.15).rename('Temp');

// Wind Speed & Direction
var u = era5.select('u_component_of_wind_10m');
var v = era5.select('v_component_of_wind_10m');
var windSpeed = u.pow(2).add(v.pow(2)).sqrt().rename('Wind_Spd');
var windDir = u.atan2(v).multiply(180 / Math.PI).rename('Wind_Dir');

// Humidity (approximate using dew point and temp)
var dewC = era5.select('dewpoint_temperature_2m').subtract(273.15);
var humidity = dewC.subtract(tempC).multiply(-1).rename('Humidity'); // Simplified version

var weather = tempC.addBands([windSpeed, windDir, humidity]);

// 5. Elevation & Slope
var elev = ee.Image('USGS/SRTMGL1_003').clip(roi);
var slope = ee.Terrain.slope(elev);

// 6. Merge all features
var combined = medianSpectral
  .addBands(weather)
  .addBands(elev.rename('Elev'))
  .addBands(slope.rename('Slope'));

// 7. Sample points
var samplePoints = combined.sample({
  region: roi,
  scale: 30,
  numPixels: 10000,
  seed: 42,
  geometries: true
});

// 8. Export table as CSV to Drive
Export.table.toDrive({
  collection: samplePoints,
  description: 'Alberta_Complete_Wildfire_Features',
  fileFormat: 'CSV'
});
