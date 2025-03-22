// Set your region of interest (example: part of Alberta)
var roi = ee.Geometry.Rectangle([-117.5, 52.0, -113.0, 55.0]); // adjust coordinates as needed

// Define date range (e.g., summer fire season)
var startDate = '2023-05-01';
var endDate = '2023-09-30';

// Sentinel-2 NDVI & NBR
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(function(img) {
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    var nbr = img.normalizedDifference(['B8', 'B12']).rename('NBR');
    return img.addBands([ndvi, nbr])
              .select(['NDVI', 'NBR'])
              .copyProperties(img, ['system:time_start']);
  });

// Landsat 8 NDVI & NBR
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUD_COVER', 20))
  .map(function(img) {
    var sr = img.multiply(0.0000275).add(-0.2); // Apply scaling
    var ndvi = sr.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
    var nbr = sr.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR');
    return img.addBands([ndvi, nbr])
              .select(['NDVI', 'NBR'])
              .copyProperties(img, ['system:time_start']);
  });

// Merge collections
var merged = s2.merge(l8);

// Create median composite
var medianComposite = merged.median();

// Display on map
Map.centerObject(roi, 7);
Map.addLayer(medianComposite.select('NDVI'), {min: 0, max: 1, palette: ['white', 'green']}, 'NDVI Composite');
Map.addLayer(medianComposite.select('NBR'), {min: -1, max: 1, palette: ['white', 'black', 'red']}, 'NBR Composite');

// Optional: Export to Google Drive
Export.image.toDrive({
  image: medianComposite,
  description: 'Alberta_NDVI_NBR',
  scale: 30,
  region: roi,
  maxPixels: 1e13
});
