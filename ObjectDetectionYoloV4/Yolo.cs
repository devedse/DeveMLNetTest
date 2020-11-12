using Microsoft.ML;
using Microsoft.ML.Data;
using ObjectDetectionYoloV4.DataStructures;
using ObjectDetectionYoloV4.YoloParser;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace ObjectDetectionYoloV4
{
    public class Yolo
    {
        private readonly MLContext _mlContext;
        private readonly OnnxModelScorer _modelScorer;
        private readonly YoloOutputParser _yoloOutputParser;

        public Yolo(string modelPath)
        {
            // Initialize MLContext
            _mlContext = new MLContext();
            _modelScorer = new OnnxModelScorer(_mlContext, modelPath);

            _yoloOutputParser = new YoloOutputParser();
        }

        public List<YoloOutput> ProcessDirectory(string directoryPath)
        {
            return ProcessImages(ImageNetData.ReadFromFile(directoryPath).ToList());
        }

        public YoloOutput ProcessImage(string imagePath)
        {
            var toProcess = new List<ImageNetData>()
            {
                new ImageNetData() { ImagePath = imagePath }
            };
            return ProcessImages(toProcess).Single();
        }

        public List<YoloOutput> ProcessImages(List<ImageNetData> images)
        {
            var w = Stopwatch.StartNew();
            IDataView imageDataView = _mlContext.Data.LoadFromEnumerable(images);

            // Use model to score data
            var modelOutput = _modelScorer.Score(imageDataView);
            Console.WriteLine($">>> 1: {w.Elapsed}");

            var boxes = modelOutput.GetColumn<float[]>("boxes").ToList();
            var confs = modelOutput.GetColumn<float[]>("confs").ToList();

            Console.WriteLine($">>> 2: {w.Elapsed}");

            if (boxes.Count != confs.Count && images.Count != boxes.Count)
            {
                throw new InvalidOperationException("boxes.Count should be equal to confs.Count. These should be equal to the images count.");
            }

            List<YoloOutput> yoloOutput = new List<YoloOutput>();
            //List<IList<YoloBoundingBox>> outputBoxes = new List<IList<YoloBoundingBox>>();
            //Boxes.Count == images.count because the model can handle multiple images at once
            for (int i = 0; i < images.Count; i++)
            {
                var imageHere = images[i];
                var boxesHere = boxes[i];
                var confsHere = confs[i];

                var parsedBoxes = _yoloOutputParser.ParseOutputs(boxesHere, confsHere);
                parsedBoxes = _yoloOutputParser.FilterBoundingBoxes(parsedBoxes, 50, 0.6f);

                var yo = new YoloOutput(imageHere.ImagePath, parsedBoxes);
                yoloOutput.Add(yo);
            }

            Console.WriteLine($">>> 3: {w.Elapsed}");

            Console.WriteLine($"Total elapsed time for processing: {w.Elapsed}");
            return yoloOutput;
        }
    }
}
