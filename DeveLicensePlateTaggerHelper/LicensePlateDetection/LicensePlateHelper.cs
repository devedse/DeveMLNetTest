using DeveCoolLib.OtherExtensions;
using DeveLicensePlateTaggerHelper.Helpers;
using Newtonsoft.Json;
using ObjectDetectionYoloV4;
using ObjectDetectionYoloV4.YoloParser;
using System;
using System.Collections.Generic;
using System.IO;

namespace DeveLicensePlateTaggerHelper.LicensePlateDetection
{
    public class LicensePlateHelper
    {
        public void TagVehiclesInExistingData(Yolo yolo)
        {
            var jsonFileDir = @"C:\XGitPrivate\DeveLicensePlateDataSet\Export";
            foreach (var file in Directory.GetFiles(jsonFileDir))
            {
                if (Path.GetExtension(file).Equals(".json", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine($"Processing: {file}");
                    var jsonData = File.ReadAllText(file);
                    var deserializedJson = JsonConvert.DeserializeObject<VottJsonAsset>(jsonData);

                    var filePath = deserializedJson.asset.path.TrimStartOnce("file:", StringComparison.OrdinalIgnoreCase);

                    var output = yolo.ProcessImage(filePath);

                    foreach (var box in output.Boxes)
                    {
                        if (box.Label is "car" or "truck" or "bus")
                        {
                            Console.WriteLine($"Found {box.Label}");
                            var newRegion = BoxToRegion(box, deserializedJson.asset.size);
                            deserializedJson.regions.Add(newRegion);
                        }
                    }

                    var newSerializedJson = JsonConvert.SerializeObject(deserializedJson, Formatting.Indented);
                    File.WriteAllText(file, newSerializedJson);

                    Console.WriteLine($"Written: {file}");
                }
            }
        }

        private static Region BoxToRegion(YoloBoundingBox box, Size size)
        {
            var capitalized = box.Label[0].ToString().ToUpper() + box.Label[1..];

            var bb = new BoundingBox()
            {
                left = box.Dimensions.X * size.width,
                top = box.Dimensions.Y * size.height,
                width = box.Dimensions.Width * size.width,
                height = box.Dimensions.Height * size.height
            };

            var region = new Region()
            {
                id = RandomStringHelper.RandomString(9),
                type = "RECTANGLE",
                tags = new List<string>() { capitalized },
                boundingBox = bb,
                points = new List<Point>()
                {
                    new Point()
                    {
                        x = bb.left,
                        y = bb.top
                    },
                    new Point()
                    {
                        x = bb.left + bb.width,
                        y = bb.top
                    },
                    new Point()
                    {
                        x = bb.left,
                        y = bb.top + bb.height
                    },
                    new Point()
                    {
                        x = bb.left + bb.width,
                        y = bb.top + bb.height
                    }
                }
            };
            return region;
        }
    }
}
