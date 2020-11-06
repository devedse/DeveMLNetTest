﻿using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;

namespace ObjectDetectionYoloV4.DataStructures
{
    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
        {
            return Directory
                .GetFiles(imageFolder)
                .Where(filePath => Path.GetExtension(filePath) != ".md")
                .Select(filePath => new ImageNetData { ImagePath = filePath });
        }
    }
}