using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using OpenCvSharp;
using System.Runtime.InteropServices;


namespace Sudoku_Finder
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            List<Mat> grayImages = new List<Mat>();
            List<Mat> unwarpedImages = new List<Mat>();
            List<Mat> filtered = new List<Mat>();
            List<Mat> processedSquares = new List<Mat>();
            List<Mat> trainingData = new List<Mat>();

            string[] imgNames =
            {
                "7_a.jpg",
                "7_b.jpg",
                "1_a.jpg",
                "1_b.jpg",
                "2_a.jpg",
                "2_b.jpg",
                "3_a.jpg",
                "3_b.jpg",
                "4_a.jpg",
                "4_b.jpg",
                "5_a.jpg",
                "5_b.jpg",
                "6_a.jpg",
                "6_b.jpg",
                "8_a.jpg",
                "8_b.jpg",
                "9_a.jpg",
                "9_b.jpg"
            };

            var imganalyzer = new SudokuFinder();
            var gray = new Mat();

            List<int> intList = new List<int>();

            foreach (var s in imgNames)
            {
                var src = Cv2.ImRead(@".\Resources\" + s);
                imganalyzer.FindSudoku(src);
                var data = imganalyzer.ProcessPuzzle(imganalyzer.unwarpedSudoku);

                grayImages.Add(src);
                filtered.Add(imganalyzer.unwarpedSudoku.Clone());
                processedSquares.Add(concatPUzzle(data));
                trainingData.AddRange(data);
                int x = Int32.Parse(s[0].ToString());
                var myArray = new List<int>();
                for (int i = 0; i < 81; i++)
                {
                    myArray.Add(x);
                }
                intList.AddRange(myArray);
            }
            intList.Add(0);

            List<double[]> TrainginData = new List<double[]>();

            int size = 0;
            foreach (var d in trainingData)
            {
                size = d.Size().Height * d.Size().Height;
                byte[] managedArray = new byte[size];
                Marshal.Copy(d.Data, managedArray, 0, size);

                TrainginData.Add(Array.ConvertAll(managedArray, c => c != 0 ? (double)1 : 0));
            }

            TrainginData.Add(new double[size]);

            var saved = imganalyzer.trainSVM(TrainginData.ToArray(), intList.ToArray());
            
            double sizeb = ((float) saved.Count() ) / 1000000.0;

            //Test
            imganalyzer = new SudokuFinder(saved);

            var src2 = Cv2.ImRead(@".\Resources\2.jpg");
            imganalyzer.FindSudoku(src2);
            var data2 = imganalyzer.ProcessPuzzle(imganalyzer.unwarpedSudoku);

            var ans2 = imganalyzer.OCR(data2);
            Cv2.ImShow("Source", imganalyzer.unwarpedSudoku);
            Cv2.ImShow("newbox", src2);
            Cv2.ImShow("data2", concatPUzzle(data2));

            Application.Run();
            Console.ReadKey();
        }

        static Mat concatPUzzle(List<Mat> data)
        {
            Mat output = new Mat();
            Mat r1 = new Mat(), r2 = new Mat(), r3 = new Mat(), r4 = new Mat(), r5 = new Mat(), r6 = new Mat(), r7 = new Mat(), r8 = new Mat(), r9 = new Mat();
            Cv2.HConcat(data.Take(9).ToArray(), r1);
            Cv2.HConcat(data.Skip(9).Take(9).ToArray(), r2);
            Cv2.HConcat(data.Skip(18).Take(9).ToArray(), r3);
            Cv2.HConcat(data.Skip(27).Take(9).ToArray(), r4);
            Cv2.HConcat(data.Skip(36).Take(9).ToArray(), r5);
            Cv2.HConcat(data.Skip(45).Take(9).ToArray(), r6);
            Cv2.HConcat(data.Skip(54).Take(9).ToArray(), r7);
            Cv2.HConcat(data.Skip(63).Take(9).ToArray(), r8);
            Cv2.HConcat(data.Skip(72).Take(9).ToArray(), r9);

            Cv2.VConcat(new Mat[] { r1, r2, r3, r4, r5, r6, r7, r8, r9 }, output);
            return output;
        }

    }
}
