using System;
using System.Collections.Generic;
using OpenCvSharp;
using Accord.Statistics.Kernels;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using System.Runtime.InteropServices;

namespace Sudoku_Finder
{
    public class SudokuFinder {

        Mat gray = new Mat();
        public Mat unwarpedSudoku;
        private Mat kernelHorz = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(27, 1));
        private Mat kernelVert = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(1, 27));
        private Mat horzSubtract = new Mat();
        private Mat vertSubtract = new Mat();
        private Point[] likelyCandidate;
        public MulticlassSupportVectorMachine<IKernel> saveKern;


        Mat[] unwarpedBoxCentres = new Mat[81];
        Point[] warpedInfo = new Point[85];
        Point[][] contours;
        HierarchyIndex[] hierarchyIndex;


        public SudokuFinder(byte[] savedKern = null, string savedKernPath = null) {
            if (savedKern!= null)
            {
                Accord.IO.Serializer.Load(savedKern, out saveKern);
            }
            else if (savedKernPath != null)
            {
                Accord.IO.Serializer.Load(savedKernPath, out saveKern);
            }
            unwarpedSudoku = new Mat(216, 216, MatType.CV_8UC1);
            int ox = 12, oy = 12, ind = 0;
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    var t = new Mat(3, 1, MatType.CV_64FC1);
                    t.Set(0, ox + i * 24);
                    t.Set(1, oy + j * 24);
                    t.Set(2, 1);
                    unwarpedBoxCentres[ind] = t;
                    ind++;
                }
            }
        }
        
        public bool FindSudoku(Mat src)
        {
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGRA2GRAY);
            Cv2.AdaptiveThreshold(gray, gray, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.Binary, 101, 5);
            Cv2.BitwiseNot(gray, gray);

            Cv2.FindContours(gray, out contours, out hierarchyIndex, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            if (!findContourOfPuzzle())
                return false;
            
            unwarpSudokuSquare();

            return true;
            //Cv2.DrawContours(src, contour_list, largestContourIndex, Scalar.Green,30);
        }

        public Point[] getSudokuLocations()
        {
            warpedInfo[0] = likelyCandidate[0];
            warpedInfo[1] = likelyCandidate[1];
            warpedInfo[2] = likelyCandidate[2];
            warpedInfo[3] = likelyCandidate[3];

            return warpedInfo;
        }

        public List<Mat> ProcessPuzzle(Mat unwarpedPuzzle)
        {
            Cv2.Erode(unwarpedPuzzle, horzSubtract, kernelHorz);
            Cv2.Dilate(horzSubtract, horzSubtract, kernelHorz);

            Cv2.Erode(unwarpedPuzzle, vertSubtract, kernelVert);
            Cv2.Dilate(vertSubtract, vertSubtract, kernelVert);

            unwarpedSudoku = unwarpedSudoku - (horzSubtract + vertSubtract);

            return getIndividualBoxes(unwarpedSudoku);
        }

        private List<Mat> getIndividualBoxes(Mat unwarped)
        {
            int pixelnum = 3;
            List<Mat> boxes = new List<Mat>();
            Point[][] contours;
            HierarchyIndex[] hierarchyIndex;
            int inc = unwarped.Rows / 9;
            for (int x = 0; x < 9; x += 1)
            {
                for (int y = 0; y < 9; y += 1)
                {
                    int xind = x * inc;
                    int yind = y * inc;
                    var num = unwarped.SubMat(xind + pixelnum, xind - pixelnum + inc, yind + pixelnum, yind - pixelnum + inc);
                    num.FindContours(out contours, out hierarchyIndex, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

                    Point[] LargestContour = new Point[1];
                    double maxArea = 0;
                    foreach (var c in contours)
                    {
                        double tempArea = Cv2.ContourArea(c);
                        if (tempArea > maxArea)
                        {
                            maxArea = tempArea;
                            LargestContour = c;
                        }
                    }
                    int w = 10;
                    Rect rec = Cv2.BoundingRect(LargestContour);
                    Mat numbox = new Mat(w, w, MatType.CV_8UC1);

                    var dest = new Point2f[] { new Point2f(0, 0), new Point2f(0, w), new Point2f(w, w), new Point2f(w, 0) };

                    var transform = Cv2.GetPerspectiveTransform(rect2Contour(rec), dest);
                    Cv2.WarpPerspective(num, numbox, transform, new Size(w, w));

                    boxes.Add(numbox);
                }
            }
            return boxes;
        }

        private List<Point2f> rect2Contour(Rect r)
        {
            var cont = new List<Point2f>();
            int ox = r.Location.X;
            int oy = r.Location.Y;
            cont.Add(new Point2f(ox, oy));
            cont.Add(new Point2f(ox, oy + r.Height));
            cont.Add(new Point2f(ox + r.Width, oy + r.Height));
            cont.Add(new Point2f(ox + r.Width, oy));
            return cont;
        }


        private void unwarpSudokuSquare()
        {
            var contour = orderFourPoints(likelyCandidate);

            var src = new Point2f[] { contour[0], contour[1], contour[2], contour[3] };
            var dest = new Point2f[] { new Point2f(0, 0), new Point2f(0, 216), new Point2f(216, 216), new Point2f(216, 0) };

            var transform = Cv2.GetPerspectiveTransform(src, dest);
            var fs1 = transform.Inv();

            Cv2.WarpPerspective(gray, unwarpedSudoku, transform, new Size(216, 216));

            for (int i =4; i < 85; i++)
            {
                Mat newpt = (fs1 * unwarpedBoxCentres[0]);
                warpedInfo[i] = new Point(newpt.At<int>(0), newpt.At<int>(1));
            }
            
        }

        private Point[] orderFourPoints(Point[] points)
        {
            Point[] sorted = new Point[4];

            int largestSum = 0, largestDiff = 0;
            int smallestSum = 100000, smallestDiff = 100000;

            foreach (Point p in points)
            {
                int sum = p.X + p.Y;
                int diff = p.X - p.Y;
                if (sum > largestSum)
                {
                    sorted[2] = p;
                    largestSum = sum;
                }
                if (sum < smallestSum)
                {
                    sorted[0] = p;
                    smallestSum = sum;
                }
                if (diff > largestDiff)
                {
                    sorted[3] = p;
                    largestDiff = diff;
                }
                if (diff < smallestDiff)
                {
                    sorted[1] = p;
                    smallestDiff = diff;
                }
            }
            return sorted;
        }

        private bool findContourOfPuzzle()
        {
            Point[] simple;
            double maxArea = 0;
            double area;
            bool found = false;

            foreach (Point[] contour in contours)
            {                
                var curve = new List<Point>(contour);
                double eps = Cv2.ArcLength(curve, true) * .08;
                simple = Cv2.ApproxPolyDP(curve, eps, true);
                if (simple.Length == 4 && Cv2.IsContourConvex(simple))
                {
                    area = Cv2.ContourArea(contour);
                    if (area > maxArea)
                    {
                        likelyCandidate = simple;
                        maxArea = area;
                        found = true;
                    }
                }
            }
            return found;
        }

        public byte[] trainSVM(double[][] inputs, int[] outputs)
        {
            IKernel kernel = Gaussian.Estimate(inputs, inputs.Length / 4);

            var numComplexity = kernel.EstimateComplexity(inputs);

            double complexity = numComplexity;
            double tolerance = (double)0.2;
            int cacheSize = (int)1000;
            SelectionStrategy strategy = SelectionStrategy.SecondOrder;

            // Create the learning algorithm using the machine and the training data
            var ml = new MulticlassSupportVectorLearning<IKernel>()
            {
                // Configure the learning algorithm
                Learner = (param) => new SequentialMinimalOptimization<IKernel>()
                {
                    Complexity = complexity,
                    Tolerance = tolerance,
                    CacheSize = cacheSize,
                    Strategy = strategy,
                    Kernel = kernel
                }
            };

            var ksvm = ml.Learn(inputs, outputs);
            byte[] saved;
            Accord.IO.Serializer.Save(ksvm, out saved);
            return saved;
        }

        public void loadSVM(byte[] kernelbyteArray)
        {
            Accord.IO.Serializer.Load(kernelbyteArray, out saveKern);
        }
        public void loadSVM(string kernelPath)
        {
            Accord.IO.Serializer.Load(kernelPath, out saveKern);
        }

        public int[] OCR(List<Mat> numbers)
        {
            double[][] testdata = new double[81][];
            int i = 0;
            foreach (var d in numbers)
            {
                int size = d.Size().Height * d.Size().Height;
                byte[] managedArray = new byte[size];
                Marshal.Copy(d.Data, managedArray, 0, size);

                testdata[i] = (Array.ConvertAll(managedArray, c => c != 0 ? (double)1 : 0));
                i++;
            }
            var a = saveKern.Decide(testdata);
            var b = saveKern.Probabilities(testdata);
            var de = saveKern.Probability(testdata);
            return saveKern.Decide(testdata);
        }

    }

}