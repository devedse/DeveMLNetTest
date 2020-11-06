using Microsoft.ML;
using ObjectDetection.DataStructures;
using ObjectDetection.YoloParser;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using static Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace ObjectDetection
{
    public class OnnxModelScorer
    {
        public const int ModelImageHeight = 416;
        public const int ModelImageWidth = 416;


        private readonly string _modelLocation;
        private readonly MLContext _mlContext;

        private readonly ITransformer _model;

        public OnnxModelScorer(MLContext mlContext, string modelLocation)
        {
            _modelLocation = modelLocation;
            _mlContext = mlContext;

            _model = LoadModel(modelLocation);
        }



        private ITransformer LoadModel(string modelLocation)
        {
            Console.WriteLine("===== Reading model =====");
            Console.WriteLine($"Model location: {modelLocation}");

            // Define scoring pipeline
            var pipeline = _mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                            .Append(_mlContext.Transforms.ResizeImages(outputColumnName: "image", imageWidth: ModelImageWidth, imageHeight: ModelImageHeight, inputColumnName: "image", resizing: ResizingKind.Fill))
                            .Append(_mlContext.Transforms.ExtractPixels(outputColumnName: "input", inputColumnName: "image", scaleImage: 1f / 255f, orderOfExtraction: ColorsOrder.ABGR))
                            .Append(_mlContext.Transforms.ApplyOnnxModel(
                                modelFile: modelLocation,
                                outputColumnNames: new[] { "boxes", "confs" },
                                inputColumnNames: new[] { "input" }
                                ));

            var w = Stopwatch.StartNew();
            //Train on empty data
            var model = pipeline.Fit(_mlContext.Data.LoadFromEnumerable(new List<ImageNetData>()));
            Console.WriteLine($"Fit took: {w.Elapsed}");

            return model;
        }

        public IDataView Score(IDataView data)
        {
            Console.WriteLine("===== Starting tranform... =====");

            var w = Stopwatch.StartNew();
            IDataView scoredData = _model.Transform(data);
            Console.WriteLine($"Transform took: {w.Elapsed}");

            return scoredData;
        }
    }
}

