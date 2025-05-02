using System.Text.Json;
using System.Text.Json.Serialization;
using logicGP.Tests.Util.ML.ModelBuilder.Configuration;

namespace logicGP.Tests.Unit.Util;

[TestClass]
public class ModelBuilderConfigurationTests
{
    [TestMethod]
    public void TestModelBuilderConfiguration()
    {
        List<IColumnProperties> columnProperties =
        [
            new ColumnPropertiesV5
            {
                ColumnName = "age",
                ColumnPurpose = ColumnPurposeType.Feature,
                ColumnDataFormat = ColumnDataKind.Single,
                IsCategorical = false
            }
        ];

        ITrainingOption trainingOption = new ClassificationTrainingOptionV2
        {
            Subsampling = false,
            TrainingTime = 20,
            LabelColumn = "num",
            AvailableTrainers =
                ["LBFGS", "LGBM", "SDCA", "FASTTREE", "RANDOMFOREST"],
            ValidationOption = new TrainValidationSplitOptionV0
            {
                SplitRatio = 0.1f
            }
        };

        var config = new TrainingConfiguration
        {
            Scenario = ScenarioType.Classification,

            DataSource = new TabularFileDataSourceV3
            {
                EscapeCharacter = '\\',
                ReadMultiLines = false,
                AllowQuoting = false,
                FilePath = "Test",
                Delimiter = ",",
                DecimalMarker = '.',
                HasHeader = true,
                ColumnProperties = columnProperties
            },
            Environment = new LocalEnvironmentV1
            {
                Type = "LocalCPU",
                EnvironmentType = EnvironmentType.LocalCPU
            },
            TrainingOption = trainingOption
        };
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters =
            {
                new JsonStringEnumConverter()
            }
        };
        var jsonString = JsonSerializer.Serialize(config, options);
        Console.WriteLine(jsonString);
        Assert.IsNotNull(jsonString);
    }
}