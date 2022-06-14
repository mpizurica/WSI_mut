import qupath.lib.regions.RegionRequest
import qupath.lib.gui.scripting.QPEx

def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()
def hierarchy = imageData.getHierarchy()
pixelfactor = server.getPixelCalibration().getPixelHeightMicrons()

// whether you want to restrict the number of tiles which are exported
// note: better to restrict later by random undersampling (here tiles processed in order)
restrict = false

// Define output resolution
double requestedPixelSize = 0.5

// Convert to downsample
double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()
print(downsample)

//create tiles: set desired output resolution
tile_px = 512 
tile_mic = tile_px * pixelfactor * downsample
print(tile_mic)

// select annotation objects -> these will be tiled
selectAnnotations();

// create tiles
runPlugin('qupath.lib.algorithms.TilerPlugin', '{"tileSizeMicrons": '+tile_mic+',  "trimToROI": false,  "makeAnnotations": true,  "removeParentAnnotation": false}');

// set class of tiles to class of parent annotation
tiles = hierarchy.getFlattenedObjectList(null)
tiles.retainAll { it.getName() != null}
tiles.each {it.setPathClass(it.getParent().getPathClass())}
print(tiles)
print('created tiles')

// get filename of slide image
def filename = server.getMetadata().getName() 

// create path to which tiles will be written
def path = buildFilePath(PROJECT_BASE_DIR, '../../thesis_2021/PRAD_dx/tiles_512px/', filename)
print(path)
mkdirs(path)

//tiles = hierarchy.getFlattenedObjectList(null)
//tiles.retainAll { it.getName() != null}

print(getAnnotationObjects())

// if we don't want more than 500 tiles of each class per patient, keep a dict to track this
Map<String, Integer> pathClasses = [:]
pathClasses['GS3'] = 0
pathClasses['GS3+3'] = 0
pathClasses['GS3+4'] = 0
pathClasses['GS4+3'] = 0
pathClasses['GS4+5'] = 0
pathClasses['GS4+4'] = 0
pathClasses['GS5+4'] = 0
pathClasses['GS5+5'] = 0
pathClasses['GS3+5'] = 0
pathClasses['GS5+3'] = 0
pathClasses['GS5+3MUCINOUSCARCINOMA'] = 0
pathClasses['GS4+3INTRADUCTALCARCINOMA'] = 0
pathClasses['Tumor'] = 0

if (tiles.size() > 0) {
    
	for (annotation in getAnnotationObjects()) {
		roi = annotation.getROI()
		def request = RegionRequest.createInstance(imageData.getServerPath(), downsample, roi)

		x = annotation.getROI().getCentroidX()
		y = annotation.getROI().getCentroidY()

		String tiletype = annotation.getParent().getPathClass()
		tiletype = tiletype.replaceAll("\\s","")
		
		// only export square tiles, and no more than 500 per class (if desired)
		if (restrict == true){
		condition = pathClasses[tiletype]<500
		}

		else {
		condition = true
		}
			
		// watch out: getArea is defined in pixels
		if ((annotation.getROI().getArea()/(tile_px*downsample*tile_px*downsample) >= 0.99)&&(annotation.getROI().getArea()/(tile_px*downsample*tile_px*downsample) <= 1.05)&&condition) {
		String tilename = String.format("%s_"+x+";"+y+'_%s_.jpg', filename, tiletype);
		
		def path2 = buildFilePath(PROJECT_BASE_DIR, '../../thesis_2021/PRAD_dx/tiles_512px/', filename+'/', tiletype)
		mkdirs(path2)

		writeImageRegion(server, request, path +'/'+ tiletype + '/' + tilename);
		print("wrote " + tilename);
		pathClasses[tiletype] = pathClasses[tiletype] + 1;

		} 
        
        }
    
    
}
else {
    print('slide has no tiles')
}

print("Wrote tiles")

//Remove tiles
// want to remove all tiles but keep annotations
remove = hierarchy.getFlattenedObjectList(null)

remove.retainAll { it.getName() != null}

if (remove.size() > 0) {
    removeObjects(remove, true)
}


print('Done')



