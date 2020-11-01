using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using ObjectDetection.DataStructures;
using ObjectDetection.YoloParser;
using static Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace ObjectDetection
{
    class OnnxModelScorer
    {
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly MLContext mlContext;

        private IList<YoloBoundingBox> _boundingBoxes = new List<YoloBoundingBox>();

        public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
        {
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.mlContext = mlContext;
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 416;
            public const int imageWidth = 416;
        }


        private ITransformer LoadModel(string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");

            // Create IDataView from empty list to obtain input data schema
            var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "image", resizing: ResizingKind.Fill))
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", inputColumnName: "image", scaleImage: 1f / 255f, orderOfExtraction: ColorsOrder.ABGR))
                            .Append(mlContext.Transforms.ApplyOnnxModel(
                                modelFile: modelLocation,
                                outputColumnNames: new[] { "boxes", "confs" },
                                inputColumnNames: new[] { "input" }
                                ));

            // Fit scoring pipeline
            var w = Stopwatch.StartNew();
            var model = pipeline.Fit(data);
            Console.WriteLine($"Fit took: {w.Elapsed}");

            return model;
        }

        private IDataView PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            Console.WriteLine($"Images location: {imagesFolder}");
            Console.WriteLine("");
            Console.WriteLine("=====Identify the objects in the images=====");
            Console.WriteLine("");

            var w = Stopwatch.StartNew();
            IDataView scoredData = model.Transform(testData);
            Console.WriteLine($"Transform took: {w.Elapsed}");

            return scoredData;
        }

        public IDataView Score(IDataView data)
        {
            var model = LoadModel(modelLocation);

            return PredictDataUsingModel(data, model);
        }
    }
}

