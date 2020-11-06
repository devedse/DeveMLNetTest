using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;

namespace ObjectDetectionYoloV4.YoloParser
{
    public class YoloOutputParser
    {
        private readonly string[] _labels = File.ReadAllLines("labels.txt").Select(t => t.Trim()).Where(t => !string.IsNullOrWhiteSpace(t)).ToArray();

        private readonly Color[] _classColors = new Color[]
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

        public IList<YoloBoundingBox> ParseOutputs(float[] boxes, float[] confs, float threshold = .4F)
        {
            Console.WriteLine("===== Parsing boxes... =====");
            var w = Stopwatch.StartNew();
            var boxesUnflattened = new List<YoloBoxDimensions>();
            for (int i = 0, boxNumber = 0; i < boxes.Length; i += 4, boxNumber++)
            {
                boxesUnflattened.Add(new YoloBoxDimensions(boxNumber, boxes.Skip(i).Take(4).ToArray()));
            }



            var confsUnflattened = new List<YoloConf>();
            for (int i = 0, number = 0; i < confs.Length; i += _labels.Length, number++)
            {
                confsUnflattened.Add(new YoloConf(number, confs.Skip(i).Take(_labels.Length).ToArray()));
            }


            var boxesIndexWhichHaveHighConfidence = confsUnflattened.Where(t => t.MaxConf > threshold).ToList();
            var allBoxesThemselvesWithHighConfidence = boxesIndexWhichHaveHighConfidence.Join(boxesUnflattened, t => t.MatchingBoxNumber, t => t.BoxNumber, (conf, box) => (Box: box, Conf: conf)).ToList();


            var boxesOutput = new List<YoloBoundingBox>();

            foreach (var b in allBoxesThemselvesWithHighConfidence)
            {
                var startString = $"{b.Conf.MatchingBoxNumber}: {_labels[b.Conf.MaxConfIndex]}";
                //Console.WriteLine($"{startString.PadRight(30, ' ')}({string.Join(",", b.Box.OriginalStuff)})");

                boxesOutput.Add(new YoloBoundingBox()
                {
                    Dimensions = b.Box,
                    Confidence = b.Conf.MaxConf,
                    Label = _labels[b.Conf.MaxConfIndex],
                    BoxColor = _classColors[b.Conf.MaxConfIndex % _classColors.Length]
                });
            }

            Console.WriteLine($"Parsing boxes took: {w.Elapsed}");

            return boxesOutput;
        }

        public IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
        {
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];

            for (int i = 0; i < isActiveBoxes.Length; i++)
            {
                isActiveBoxes[i] = true;
            }

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
                    {
                        break;
                    }

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
                                {
                                    break;
                                }
                            }
                        }
                    }

                    if (activeCount <= 0)
                    {
                        break;
                    }
                }
            }
            return results;
        }

        private float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
        {
            var areaA = boundingBoxA.Width * boundingBoxA.Height;

            if (areaA <= 0)
            {
                return 0;
            }

            var areaB = boundingBoxB.Width * boundingBoxB.Height;

            if (areaB <= 0)
            {
                return 0;
            }

            var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
            var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
            var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
            var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }
    }
}