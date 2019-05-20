# dijkstraRoadProcess

This folder contains the development of dijkstraRoadProcess. This process computes a shortest path from the vanishing point in the image to the botom line of the frame, this path has the purpose of delimiting the drivable road from the point of view of the camera to the vanishing point. Paper followed can ve viewed here: 
https://drive.google.com/file/d/1ZMG_8rpvHtK4SdWMdcKoXMNaaTW5WLTU/view?usp=sharing


*dijkstraRoadEngine* runs the pipeline to process each frame retrieved from ROS nodes publishing the disparity frame, the camera image and the vdisp line fitter. The pipeline is as follows:
1. Get disparity image and color image from ros publishers.
2. Get mocked vanishing point. (_current develop on another branch_)
3. Get weightedSumMatrix upoin with shortest path will be calculated. 
4. Run dijsktra shortest path over weightedSumMatrix

*computeCostUtility* computes the following costs:
	computeGradientDirectionCost
	computeDisparityFeatureCost
	computeFlatnessCost
	computeOrientationLinkCost*
	computeGradientCost

*dijkstraUtility* performs dijkstra shortests path from vanishing point to point P1 and P2. These two points is where the shortest path encounters botom row of the image from the weightedSumMatrix

