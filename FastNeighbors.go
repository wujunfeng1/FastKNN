package FastNeighbors

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"sort"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type Interval struct {
	LBound, UBound float64
}

type Point struct {
	ID  int
	Vec []float64
}

type KDLeaf struct {
	BoundingBox []Interval
	Points      []Point
}

type KDTree struct {
	BoundingBox []Interval
	DivisionDim int
	NumPoints   int
	LSubTree    *KDTree
	RSubTree    *KDTree
	LLeaf       *KDLeaf
	RLeaf       *KDLeaf
}

func computeBoundingBox(points []Point) []Interval {
	m := len(points[0].Vec)
	n := len(points)
	boundingBox := make([]Interval, m)
	for i := 0; i < m; i++ {
		boundingBox[i] = Interval{points[0].Vec[i], points[0].Vec[i]}
	}
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			value := points[j].Vec[i]
			if value < boundingBox[i].LBound {
				boundingBox[i].LBound = value
			} else if value > boundingBox[i].UBound {
				boundingBox[i].UBound = value
			}
		}
	}
	return boundingBox
}

func newKDLeaf(points []Point) *KDLeaf {
	return &KDLeaf{
		BoundingBox: computeBoundingBox(points),
		Points:      points,
	}
}

func NewKDTree(pointLocations [][]float64, numPointsPerLeaf int, relaxation float64) *KDTree {
	// get dims of data
	n := len(pointLocations)
	if n == 0 {
		log.Fatalln("no input points passed to NewKDTree")
	}
	m := len(pointLocations[0])
	if m == 0 {
		log.Fatalln("input point of 0 dims is not allowed")
	}

	// check data
	for j := 1; j < n; j++ {
		if len(pointLocations[j]) != m {
			log.Fatalln("dims of input points don't match")
		}
	}

	// check numPointsPerLeaf
	if numPointsPerLeaf < 1 {
		log.Fatalln("numPointsPerLeaf is not allowed to be < 1")
	}

	// create points
	points := make([]Point, n)
	for j, location := range pointLocations {
		points[j] = Point{ID: j, Vec: location}
	}

	// return a KD tree containing these points
	return createKDTree(points, numPointsPerLeaf, relaxation)
}

// relaxation == 0.0 means no relaxation, all splits will be exactly half
// relaxation == 1.0 means full relaxation, any split will be allowed.
func createKDTree(points []Point, numPointsPerLeaf int, relaxation float64) *KDTree {
	if relaxation < 0.0 || relaxation > 1.0 {
		log.Fatalln("relaxation should be within [0.0,1.0]")
	}

	// create a KD tree with all these points in one leaf
	result := &KDTree{
		BoundingBox: computeBoundingBox(points),
		DivisionDim: -1,
		NumPoints:   len(points),
		LSubTree:    nil,
		RSubTree:    nil,
		LLeaf:       newKDLeaf(points),
		RLeaf:       nil,
	}

	// split it to satisfy the numPointsPerLeaf condition
	result.split(numPointsPerLeaf, relaxation)

	// retur the result
	return result
}

func (kdtree *KDTree) split(numPointsPerLeaf int, relaxation float64) {
	if kdtree.DivisionDim != -1 {
		log.Fatalln("a KDTree is not allowed to split twice")
	}

	// if the tree is already small, just set its DivisionDim and return
	if len(kdtree.LLeaf.Points) <= numPointsPerLeaf {
		kdtree.DivisionDim = 0
		return
	}

	// find the dim with the largest variance
	m := len(kdtree.BoundingBox)
	points := kdtree.LLeaf.Points
	n := len(points)
	means := make([]float64, m)
	vars := make([]float64, m)
	for i := 0; i < m; i++ {
		means[i] = 0.0
		vars[i] = 0.0
	}
	for j := 0; j < n; j++ {
		point := points[j]
		for i := 0; i < m; i++ {
			means[i] += point.Vec[i]
			vars[i] += point.Vec[i] * point.Vec[i]
		}
	}
	for i := 0; i < m; i++ {
		means[i] /= float64(n)
		vars[i] -= float64(n) * means[i] * means[i]
	}

	dimWithMaxVar := 0
	for i := 1; i < m; i++ {
		if vars[i] > vars[dimWithMaxVar] {
			dimWithMaxVar = i
		}
	}

	// sort along dim with the largest variance
	kdtree.DivisionDim = dimWithMaxVar
	sort.Slice(points, func(i, j int) bool {
		return points[i].Vec[dimWithMaxVar] < points[j].Vec[dimWithMaxVar]
	})

	// compute the density along the dim
	// reflect elements at boundary
	densities := make([]float64, n)
	densities[0] = 0.5 / (1e-10 + points[1].Vec[dimWithMaxVar] - points[0].Vec[dimWithMaxVar])
	densities[n-1] = 0.5 / (1e-10 + points[n-1].Vec[dimWithMaxVar] - points[n-2].Vec[dimWithMaxVar])
	for i := 1; i < n-1; i++ {
		densities[i] = 1.0 / (1e-10 + points[i+1].Vec[dimWithMaxVar] - points[i-1].Vec[dimWithMaxVar])
	}
	prevDensities := densities
	densities = make([]float64, n)
	for iter := 0; iter < 3; iter++ {
		densities[0] = 0.5 * (prevDensities[0] + prevDensities[1])
		densities[n-1] = 0.5 * (prevDensities[n-2] + prevDensities[n-1])
		for i := 1; i < n-1; i++ {
			densities[i] = 0.25*(prevDensities[i-1]+prevDensities[i+1]) +
				0.5*prevDensities[i]
		}
		prevDensities, densities = densities, prevDensities
	}
	highestDensity := 0.0
	for i := 0; i < n; i++ {
		if densities[i] > highestDensity {
			highestDensity = densities[i]
		}
	}

	// determine the range of valid split
	leftmostSplit := int(math.Floor(0.5 * (1.0 - relaxation) * float64(n)))
	rightmostSplit := n - leftmostSplit
	if rightmostSplit < leftmostSplit {
		rightmostSplit = leftmostSplit
	} else if rightmostSplit >= n {
		rightmostSplit = n - 1
	}

	// scan through the range to find the best split
	bestSplit := n / 2
	splitDensity := highestDensity
	for i := leftmostSplit; i <= rightmostSplit; i++ {
		if densities[i] < splitDensity {
			bestSplit = i
			splitDensity = densities[i]
		}
	}

	leftPoints := points[0:bestSplit]
	rightPoints := points[bestSplit:n]
	kdtree.LLeaf = nil

	// for each part, split it if it is still too large
	var waitGroup sync.WaitGroup
	waitGroup.Add(1)
	go func(wg *sync.WaitGroup) {
		if len(rightPoints) > numPointsPerLeaf {
			kdtree.RSubTree = createKDTree(rightPoints, numPointsPerLeaf, relaxation)
		} else {
			kdtree.RLeaf = newKDLeaf(rightPoints)
		}
		wg.Done()
	}(&waitGroup)

	if len(leftPoints) > numPointsPerLeaf {
		kdtree.LSubTree = createKDTree(leftPoints, numPointsPerLeaf, relaxation)
	} else {
		kdtree.LLeaf = newKDLeaf(leftPoints)
	}
	waitGroup.Wait()
}

func (kdtree KDTree) ForEachLeaf(f func(leaf *KDLeaf)) {
	if kdtree.LLeaf != nil {
		f(kdtree.LLeaf)
	} else if kdtree.LSubTree != nil {
		kdtree.LSubTree.ForEachLeaf(f)
	}

	if kdtree.RLeaf != nil {
		f(kdtree.RLeaf)
	} else if kdtree.RSubTree != nil {
		kdtree.RSubTree.ForEachLeaf(f)
	}
}

func findNearestPointInBoundingBox(center []float64, boundingBox []Interval,
) []float64 {
	m := len(boundingBox)
	result := make([]float64, m)
	for i := 0; i < m; i++ {
		if center[i] < boundingBox[i].LBound {
			result[i] = boundingBox[i].LBound
		} else if center[i] > boundingBox[i].UBound {
			result[i] = boundingBox[i].UBound
		} else {
			result[i] = center[i]
		}
	}
	return result
}

func computeDistanceToBoundingBox(center []float64, boundingBox []Interval,
) float64 {
	nearestPoint := findNearestPointInBoundingBox(center, boundingBox)
	return computeDistanceToPoint(center, nearestPoint)
}

func computeDistanceToPoint(center []float64, point []float64) float64 {
	result := 0.0
	for i, value := range center {
		dOfDim := value - point[i]
		result += dOfDim * dOfDim
	}
	result = math.Sqrt(result)
	return result
}

func (kdtree KDTree) ComputeDensity(center []float64, radius float64) int {
	m := len(kdtree.BoundingBox)
	if len(center) != m {
		log.Fatalln("Dims of center does not match dims of kd-tree")
	}

	result := 0
	distance := computeDistanceToBoundingBox(center, kdtree.BoundingBox)
	if distance > radius {
		return result
	}

	if kdtree.LSubTree != nil {
		childrenResult := kdtree.LSubTree.ComputeDensity(center, radius)
		result += childrenResult
	} else if kdtree.LLeaf != nil {
		childrenResult := kdtree.LLeaf.computeDensity(center, radius)
		result += childrenResult
	}
	if kdtree.RSubTree != nil {
		childrenResult := kdtree.RSubTree.ComputeDensity(center, radius)
		result += childrenResult
	} else if kdtree.RLeaf != nil {
		childrenResult := kdtree.RLeaf.computeDensity(center, radius)
		result += childrenResult
	}

	return result
}

func (kdleaf KDLeaf) computeDensity(center []float64, radius float64) int {
	m := len(kdleaf.BoundingBox)
	if len(center) != m {
		log.Fatalln("Dims of center does not match dims of kd-leaf")
	}

	result := 0
	distance := computeDistanceToBoundingBox(center, kdleaf.BoundingBox)
	if distance > radius {
		return result
	}

	n := len(kdleaf.Points)
	for j := 0; j < n; j++ {
		distance = computeDistanceToPoint(center, kdleaf.Points[j].Vec)
		if distance <= radius {
			result++
		}
	}

	return result
}

func (kdtree KDTree) getPointsTo(points []Point) {
	if kdtree.LSubTree != nil {
		kdtree.LSubTree.getPointsTo(points)
	} else if kdtree.LLeaf != nil {
		for _, point := range kdtree.LLeaf.Points {
			points[point.ID] = point
		}
	}

	if kdtree.RSubTree != nil {
		kdtree.RSubTree.getPointsTo(points)
	} else if kdtree.RLeaf != nil {
		for _, point := range kdtree.RLeaf.Points {
			points[point.ID] = point
		}
	}
}

func (kdtree KDTree) GetLeafClusters() [][]int {
	result := [][]int{}
	if kdtree.LSubTree != nil {
		result = kdtree.LSubTree.GetLeafClusters()
	} else if kdtree.LLeaf != nil {
		cluster := make([]int, len(kdtree.LLeaf.Points))
		for i, point := range kdtree.LLeaf.Points {
			cluster[i] = point.ID
		}
		result = [][]int{cluster}
	}

	if kdtree.RSubTree != nil {
		result = append(result, kdtree.RSubTree.GetLeafClusters()...)
	} else if kdtree.RLeaf != nil {
		cluster := make([]int, len(kdtree.RLeaf.Points))
		for i, point := range kdtree.RLeaf.Points {
			cluster[i] = point.ID
		}
		result = append(result, cluster)
	}

	return result
}

func (kdleaf KDLeaf) testDensityPeak(center []float64, radius float64, idxCenter int,
	densities []int) bool {
	m := len(kdleaf.BoundingBox)
	if len(center) != m {
		log.Fatalln("Dims of center does not match dims of kd-leaf")
	}

	distance := computeDistanceToBoundingBox(center, kdleaf.BoundingBox)
	if distance > radius {
		return true
	}

	n := len(kdleaf.Points)
	for j := 0; j < n; j++ {
		distance = computeDistanceToPoint(center, kdleaf.Points[j].Vec)
		if distance <= radius {
			if densities[kdleaf.Points[j].ID] > densities[idxCenter] {
				return false
			}
		}
	}

	return true
}

func (kdtree KDTree) testDensityPeak(center []float64, radius float64, idxCenter int,
	densities []int) bool {
	m := len(kdtree.BoundingBox)
	if len(center) != m {
		log.Fatalln("Dims of center does not match dims of kd-tree")
	}

	distance := computeDistanceToBoundingBox(center, kdtree.BoundingBox)
	if distance > radius {
		return true
	}

	if kdtree.LSubTree != nil {
		if !kdtree.LSubTree.testDensityPeak(center, radius, idxCenter, densities) {
			return false
		}
	} else if kdtree.LLeaf != nil {
		if !kdtree.LLeaf.testDensityPeak(center, radius, idxCenter, densities) {
			return false
		}
	}
	if kdtree.RSubTree != nil {
		if !kdtree.RSubTree.testDensityPeak(center, radius, idxCenter, densities) {
			return false
		}
	} else if kdtree.RLeaf != nil {
		if !kdtree.RLeaf.testDensityPeak(center, radius, idxCenter, densities) {
			return false
		}
	}

	return true
}

func (kdtree KDTree) FindDensityPeaks(radius float64) []int {
	points := make([]Point, kdtree.NumPoints)
	densities := make([]int, kdtree.NumPoints)
	isDensityPeak := make([]bool, kdtree.NumPoints)
	kdtree.getPointsTo(points)

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	wg.Add(numCPUs)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func(idxCPU int) {
			i0 := idxCPU * kdtree.NumPoints / numCPUs
			i1 := (idxCPU + 1) * kdtree.NumPoints / numCPUs
			for i := i0; i < i1; i++ {
				densities[i] = kdtree.ComputeDensity(points[i].Vec, radius)
			}
			wg.Done()
		}(idxCPU)
	}
	wg.Wait()

	wg.Add(numCPUs)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func(idxCPU int) {
			i0 := idxCPU * kdtree.NumPoints / numCPUs
			i1 := (idxCPU + 1) * kdtree.NumPoints / numCPUs
			for i := i0; i < i1; i++ {
				isDensityPeak[i] = kdtree.testDensityPeak(points[i].Vec, radius, i, densities)
			}
			wg.Done()
		}(idxCPU)
	}
	wg.Wait()

	numDensityPeaks := 0
	for i := 0; i < kdtree.NumPoints; i++ {
		if isDensityPeak[i] {
			numDensityPeaks++
		}
	}
	result := make([]int, numDensityPeaks)
	idxPeak := 0
	for i := 0; i < kdtree.NumPoints; i++ {
		if isDensityPeak[i] {
			result[idxPeak] = i
			idxPeak++
		}
	}
	return result
}

func (kdtree KDTree) FindAdaptiveDensityPeaks(scale float64) []int {
	points := make([]Point, kdtree.NumPoints)
	densities := make([]int, kdtree.NumPoints)
	isDensityPeak := make([]bool, kdtree.NumPoints)
	kdtree.getPointsTo(points)

	ar := NewAdaptiveRadius(kdtree.getLeafInfo())

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	wg.Add(numCPUs)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func(idxCPU int) {
			i0 := idxCPU * kdtree.NumPoints / numCPUs
			i1 := (idxCPU + 1) * kdtree.NumPoints / numCPUs
			for i := i0; i < i1; i++ {
				radius := math.Min(ar.RadiusAt(points[i].Vec)*scale, 1.0)
				densities[i] = kdtree.ComputeDensity(points[i].Vec, radius)
				//fmt.Printf("%d, %d: radius = %f, density = %d\n", idxCPU, i, radius, densities[i])
			}
			wg.Done()
		}(idxCPU)
	}
	wg.Wait()

	wg.Add(numCPUs)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func(idxCPU int) {
			i0 := idxCPU * kdtree.NumPoints / numCPUs
			i1 := (idxCPU + 1) * kdtree.NumPoints / numCPUs
			for i := i0; i < i1; i++ {
				radius := ar.RadiusAt(points[i].Vec) * scale
				isDensityPeak[i] = kdtree.testDensityPeak(points[i].Vec, radius, i, densities)
			}
			wg.Done()
		}(idxCPU)
	}
	wg.Wait()

	numDensityPeaks := 0
	for i := 0; i < kdtree.NumPoints; i++ {
		if isDensityPeak[i] {
			numDensityPeaks++
		}
	}
	result := make([]int, numDensityPeaks)
	idxPeak := 0
	for i := 0; i < kdtree.NumPoints; i++ {
		if isDensityPeak[i] {
			result[idxPeak] = i
			idxPeak++
		}
	}
	return result
}

func (kdtree KDTree) FindNeighbors(center []float64, radius float64) []Point {
	m := len(kdtree.BoundingBox)
	if len(center) != m {
		log.Fatalln("Dims of center does not match dims of kd-tree")
	}

	result := []Point{}
	distance := computeDistanceToBoundingBox(center, kdtree.BoundingBox)
	if distance > radius {
		return result
	}

	if kdtree.LSubTree != nil {
		childrenResult := kdtree.LSubTree.FindNeighbors(center, radius)
		result = append(result, childrenResult...)
	} else if kdtree.LLeaf != nil {
		childrenResult := kdtree.LLeaf.findNeighbors(center, radius)
		result = append(result, childrenResult...)
	}
	if kdtree.RSubTree != nil {
		childrenResult := kdtree.RSubTree.FindNeighbors(center, radius)
		result = append(result, childrenResult...)
	} else if kdtree.RLeaf != nil {
		childrenResult := kdtree.RLeaf.findNeighbors(center, radius)
		result = append(result, childrenResult...)
	}

	return result
}

func (kdleaf KDLeaf) findNeighbors(center []float64, radius float64) []Point {
	m := len(kdleaf.BoundingBox)
	if len(center) != m {
		log.Fatalln("Dims of center does not match dims of kd-leaf")
	}

	result := []Point{}
	distance := computeDistanceToBoundingBox(center, kdleaf.BoundingBox)
	if distance > radius {
		return result
	}

	n := len(kdleaf.Points)
	for j := 0; j < n; j++ {
		distance = computeDistanceToPoint(center, kdleaf.Points[j].Vec)
		if distance <= radius {
			result = append(result, kdleaf.Points[j])
		}
	}

	return result
}

func (kdtree KDTree) FindAllNeighbors(radius float64) [][]Point {
	result := make([][]Point, kdtree.NumPoints)
	kdtree.findAllNeighborsTo(result, radius, kdtree)
	return result
}

func (kdtree KDTree) findAllNeighborsTo(neighbors [][]Point, radius float64,
	root KDTree) {
	var waitGroup sync.WaitGroup
	if kdtree.RSubTree != nil {
		waitGroup.Add(1)
		go func(wg *sync.WaitGroup) {
			kdtree.RSubTree.findAllNeighborsTo(neighbors, radius, root)
			wg.Done()
		}(&waitGroup)
	} else if kdtree.RLeaf != nil {
		for _, point := range kdtree.RLeaf.Points {
			neighbors[point.ID] = root.FindNeighbors(point.Vec, radius)
		}
	}

	if kdtree.LSubTree != nil {
		kdtree.LSubTree.findAllNeighborsTo(neighbors, radius, root)
	} else if kdtree.LLeaf != nil {
		for _, point := range kdtree.LLeaf.Points {
			neighbors[point.ID] = root.FindNeighbors(point.Vec, radius)
		}
	}
	waitGroup.Wait()
}

type CircularRegion struct {
	Center []float64
	Radius float64
}

func (kdleaf KDLeaf) getLeafInfo() CircularRegion {
	numDims := len(kdleaf.BoundingBox)
	center := make([]float64, numDims)
	for i := 0; i < numDims; i++ {
		center[i] = 0.0
	}
	for _, point := range kdleaf.Points {
		for i := 0; i < numDims; i++ {
			center[i] += point.Vec[i]
		}
	}
	numPoints := float64(len(kdleaf.Points))
	if numPoints > 0.0 {
		w := 1.0 / numPoints
		for i := 0; i < numDims; i++ {
			center[i] *= w
		}
	}
	radius := 0.0
	for _, point := range kdleaf.Points {
		myRadius := 0.0
		for i := 0; i < numDims; i++ {
			d := center[i] - point.Vec[i]
			myRadius += d * d
		}
		myRadius = math.Sqrt(myRadius)
		if myRadius > radius {
			radius = myRadius
		}
	}

	return CircularRegion{Center: center, Radius: radius}
}

func (kdtree KDTree) getLeafInfo() []CircularRegion {
	result := []CircularRegion{}
	if kdtree.LSubTree != nil {
		result = kdtree.LSubTree.getLeafInfo()
	} else if kdtree.LLeaf != nil {
		result = []CircularRegion{kdtree.LLeaf.getLeafInfo()}
	}
	if kdtree.RSubTree != nil {
		result = append(result, kdtree.RSubTree.getLeafInfo()...)
	} else if kdtree.RLeaf != nil {
		result = append(result, kdtree.RLeaf.getLeafInfo())
	}
	return result
}

type AdaptiveRadius struct {
	Regions  []CircularRegion
	RBFCoefs []float64
	RBFVar   float64
}

func NewAdaptiveRadius(regions []CircularRegion) AdaptiveRadius {
	sumRadius := 0.0
	numRegions := len(regions)
	for _, region := range regions {
		sumRadius += region.Radius
	}
	meanRadius := sumRadius / float64(numRegions)
	rbfVar := meanRadius

	numDims := len(regions[0].Center)
	A := mat.NewDense(numRegions, numRegions, nil)
	b := mat.NewVecDense(numRegions, nil)
	x := mat.NewVecDense(numRegions, nil)
	w := 0.5 / (rbfVar * rbfVar)
	for i := 0; i < numRegions; i++ {
		b.SetVec(i, regions[i].Radius)
		for j := i; j < numRegions; j++ {
			cdIJ := 0.0
			for k := 0; k < numDims; k++ {
				d := regions[i].Center[k] - regions[j].Center[k]
				cdIJ += d * d
			}
			wij := math.Exp(-cdIJ * w)
			A.Set(i, j, wij)
			if i != j {
				A.Set(j, i, wij)
			}
		}
	}

	var qr mat.QR
	qr.Factorize(A)

	err := qr.SolveVecTo(x, false, b)
	if err != nil {
		log.Fatalln(err)
	}

	coefs := make([]float64, numRegions)
	for i := 0; i < numRegions; i++ {
		coefs[i] = x.AtVec(i)
	}

	return AdaptiveRadius{Regions: regions, RBFCoefs: coefs, RBFVar: rbfVar}
}

func (ar AdaptiveRadius) RadiusAt(center []float64) float64 {
	numDims := len(ar.Regions[0].Center)
	if len(center) != numDims {
		err := fmt.Sprintf(
			"Atempting to get adaptive radius of %d-dimensions at center of %d-dimensions",
			numDims, len(center))
		log.Fatalln(err)
	}

	result := 0.0
	numRegions := len(ar.Regions)
	w := 0.5 / (ar.RBFVar * ar.RBFVar)
	for i := 0; i < numRegions; i++ {
		cdI := 0.0
		for k := 0; k < numDims; k++ {
			d := ar.Regions[i].Center[k] - center[k]
			cdI += d * d
		}
		wi := math.Exp(-cdI * w)
		result += wi * ar.RBFCoefs[i]
	}

	return result
}
