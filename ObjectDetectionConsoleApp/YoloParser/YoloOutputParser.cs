﻿using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace ObjectDetection.YoloParser
{
    class YoloOutputParser
    {
        class CellDimensions : DimensionsBase { }

        public const int ROW_COUNT = 13;
        public const int COL_COUNT = 13;
        public const int CHANNEL_COUNT = 125;
        public const int BOXES_PER_CELL = 5;
        public const int BOX_INFO_FEATURE_COUNT = 5;
        public const int CLASS_COUNT = 20;
        public const float CELL_WIDTH = 32;
        public const float CELL_HEIGHT = 32;

        private int channelStride = ROW_COUNT * COL_COUNT;

        private float[] anchors = new float[]
        {
            1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
        };

        private string[] labels = File.ReadAllLines("labels.txt").Select(t => t.Trim()).Where(t => !string.IsNullOrWhiteSpace(t)).ToArray();

        private static Color[] classColors = new Color[]
        {
            Color.Khaki,
            Color.Fuchsia,
            Color.Silver,
            Color.RoyalBlue,
            Color.Green,
            Color.DarkOrange,
            Color.Purple,
            Color.Gold,
            Color.Red,
            Color.Aquamarine,
            Color.Lime,
            Color.AliceBlue,
            Color.Sienna,
            Color.Orchid,
            Color.Tan,
            Color.LightPink,
            Color.Yellow,
            Color.HotPink,
            Color.OliveDrab,
            Color.SandyBrown,
            Color.DarkTurquoise
        };

        private float Sigmoid(float value)
        {
            var k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        private float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        private int GetOffset(int x, int y, int channel)
        {
            // YOLO outputs a tensor that has a shape of 125x13x13, which 
            // WinML flattens into a 1D array.  To access a specific channel 
            // for a given (x,y) cell position, we need to calculate an offset
            // into the array
            return (channel * this.channelStride) + (y * COL_COUNT) + x;
        }

        private BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
        {
            return new BoundingBoxDimensions
            {
                X = modelOutput[GetOffset(x, y, channel)],
                Y = modelOutput[GetOffset(x, y, channel + 1)],
                Width = modelOutput[GetOffset(x, y, channel + 2)],
                Height = modelOutput[GetOffset(x, y, channel + 3)]
            };
        }

        private float GetConfidence(float[] modelOutput, int x, int y, int channel)
        {
            return Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]);
        }

        private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
        {
            return new CellDimensions
            {
                X = ((float)x + Sigmoid(boxDimensions.X)) * CELL_WIDTH,
                Y = ((float)y + Sigmoid(boxDimensions.Y)) * CELL_HEIGHT,
                Width = (float)Math.Exp(boxDimensions.Width) * CELL_WIDTH * anchors[box * 2],
                Height = (float)Math.Exp(boxDimensions.Height) * CELL_HEIGHT * anchors[box * 2 + 1],
            };
        }

        public float[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
        {
            float[] predictedClasses = new float[CLASS_COUNT];
            int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;
            for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
            }
            return Softmax(predictedClasses);
        }

        private ValueTuple<int, float> GetTopResult(float[] predictedClasses)
        {
            return predictedClasses
                .Select((predictedClass, index) => (Index: index, Value: predictedClass))
                .OrderByDescending(result => result.Value)
                .First();
        }

        private float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
        {
            var areaA = boundingBoxA.Width * boundingBoxA.Height;

            if (areaA <= 0)
                return 0;

            var areaB = boundingBoxB.Width * boundingBoxB.Height;

            if (areaB <= 0)
                return 0;

            var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
            var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
            var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
            var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }



        public IList<YoloBoundingBox> ParseOutputs(float[] boxes, float[] confs, float threshold = .6F)
        {
            var boxesUnflattened = new List<BoundingBoxDimensions>();
            for (int i = 0; i < boxes.Length; i += 4)
            {
                boxesUnflattened.Add(new BoundingBoxDimensions()
                {
                    X = boxes[i],
                    Y = boxes[i + 1],
                    Width = boxes[i + 2] - boxes[i],
                    Height = boxes[i + 3] - boxes[i + 1],
                    OriginalStuff = boxes.Skip(i).Take(4).ToArray()
                });
            }



            var confsUnflattened = new List<float[]>();
            for (int i = 0; i < confs.Length; i += 80)
            {
                confsUnflattened.Add(confs.Skip(i).Take(80).ToArray());
            }



            var minx = boxesUnflattened.Min(t => t.X);
            var maxx = boxesUnflattened.Max(t => t.X);
            var miny = boxesUnflattened.Min(t => t.Y);
            var maxy = boxesUnflattened.Max(t => t.Y);

            var minw = boxesUnflattened.Min(t => t.Width);
            var maxw = boxesUnflattened.Max(t => t.Width);
            var minh = boxesUnflattened.Min(t => t.Height);
            var maxh = boxesUnflattened.Max(t => t.Height);


            //var maxPerClass = Enumerable.Range(0, labels.Length).Select(i => confsUnflattened[i].Max()).ToList();
            var maxConfidencePerBox = confsUnflattened.Select(t => t.Select((n, i) => (Number: n, Index: i)).Max()).ToList();
            var boxesNumbered = boxesUnflattened.Select((b, i) => (Box: b, Index: i)).ToList();

            var boxesIndexDieHogeConfidenceHebben = maxConfidencePerBox.Where(t => t.Number > threshold).ToList();
            var alleBoxesDieHogeConfidenceHebben = boxesIndexDieHogeConfidenceHebben.Join(boxesNumbered, t => t.Index, t => t.Index, (l, r) => r).ToList();

            for (int j = 0; j < labels.Length; j++)
            {

            }




            int isnietgoedcount = 0;

            var boxesOutput = new List<YoloBoundingBox>();
            for (int i = 0; i < boxesUnflattened.Count; i++)
            {
                var bb = boxesUnflattened[i];
                var conf = confsUnflattened[i];

                var topResult = conf.Max();
                var topResultIndex = conf.ToList().IndexOf(topResult);

                if (bb.X > 0 && bb.Y > 0 && bb.Width > 0.01f && bb.Height > 0.01f)
                {
                    boxesOutput.Add(new YoloBoundingBox()
                    {
                        Dimensions = bb,
                        Confidence = topResult,
                        Label = labels[topResultIndex],
                        BoxColor = classColors[topResultIndex % classColors.Length]
                    });
                }
            }

            var dingetje = boxesOutput.OrderByDescending(t => t.Confidence).ToList();

            return boxesOutput;
        }

        public IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
        {
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];

            for (int i = 0; i < isActiveBoxes.Length; i++)
                isActiveBoxes[i] = true;

            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                                .OrderByDescending(b => b.Box.Confidence)
                                .ToList();

            var results = new List<YoloBoundingBox>();

            for (int i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    if (results.Count >= limit)
                        break;

                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;

                            if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }

                    if (activeCount <= 0)
                        break;
                }
            }
            return results;
        }

    }
}