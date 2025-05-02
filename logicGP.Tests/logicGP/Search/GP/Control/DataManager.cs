using System.Globalization;
using Italbytz.Adapters.Algorithms.AI.Search.GP.Fitness;
using Italbytz.Adapters.Algorithms.AI.Search.GP.SearchSpace;
using Italbytz.Adapters.Algorithms.AI.Util;
using Italbytz.ML;
using Italbytz.Ports.Algorithms.AI.Search.GP.SearchSpace;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.Adapters.Algorithms.AI.Search.GP.Control;

public class DataManager
{
    public required List<ILiteral<string>> Literals { get; set; }

    public required List<string> Labels { get; set; }

    public required string Label { get; set; }

    public void Initialize(IDataView gpTrainingData,
        string labelColumnName = DefaultColumnNames.Label)
    {
        // Determine the labels from the label column
        Label = labelColumnName;
        Labels = (gpTrainingData.GetOrderedUniqueColumnEntries(Label) ??
                  throw new InvalidOperationException(
                      "The label column could not be parsed into string labels."))
            .ToList();

        // Construct the literals for the feature columns
        Literals = [];

        var slotsNamesArray = gpTrainingData.GetFeaturesSlotNames();


        var values = gpTrainingData
            .GetColumn<float[]>(DefaultColumnNames.Features)
            .ToList();


        var feature = 0;

        foreach (var slotName in slotsNamesArray)
        {
            var columnData = GetFeatureColumnAsString(values, feature);

            var uniqueValues =
                new HashSet<string>(
                    columnData);
            var uniqueCount = uniqueValues.Count;

            var powerSetCount = 1 << uniqueCount;
            for (var i = 1; i < powerSetCount - 1; i++)
            {
                var literalType = uniqueValues.Count <= 3
                    ? LogicGpLiteralType.Dussault
                    : LogicGpLiteralType.Rudell;
                var literal = new LogicGpLiteral<string>(feature,
                    slotName.ToString(),
                    uniqueValues, i,
                    columnData, literalType);
                Literals.Add(literal);
            }

            feature++;
        }
    }

    private static List<string> GetFeatureColumnAsString(List<float[]> values,
        int i)
    {
        return values
            .Select(row => row[i].ToString(CultureInfo.InvariantCulture))
            .ToList();
    }

    public ILiteral<string> GetRandomLiteral()
    {
        var random = ThreadSafeRandomNetCore.LocalRandom;
        var index = random.Next(Literals.Count);
        return Literals[index];
    }
}